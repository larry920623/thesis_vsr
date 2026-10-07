"""Synthetic tests; --confidence-only runs without importing MMCV/model."""
import argparse
import importlib.util
from pathlib import Path
import torch


def confidence_tests(device):
    # Load leaf module directly so package __init__ does not require MMCV.
    # Works both at project root and after moving to scripts/tests/.
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "archs/basicvsrpp/depth_confidence.py").is_file()), None)
    if root is None:
        raise FileNotFoundError("Cannot locate archs/basicvsrpp/depth_confidence.py")
    path = root / "archs/basicvsrpp/depth_confidence.py"
    spec = importlib.util.spec_from_file_location("c2_confidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    d = torch.full((1, 1, 8, 8), 0.5, device=device)
    flow = torch.zeros(1, 2, 8, 8, device=device)
    c, diag = module.depth_motion_confidence(d, d, flow)
    torch.testing.assert_close(c, torch.ones_like(c), atol=1e-6, rtol=0)
    mismatch = d.clone(); mismatch[:, :, :, 4:] = 0.6
    c, _ = module.depth_motion_confidence(mismatch, d, flow)
    assert c[:, :, :, 4:].max() < 1e-6
    flow[:, 0] = 1
    c, _ = module.depth_motion_confidence(d, d, flow)
    assert c[:, :, :, -1].max() == 0
    torch.testing.assert_close(c[:, :, :, :-1], torch.ones_like(c[:, :, :, :-1]), atol=1e-6, rtol=0)
    prev_conf = torch.ones_like(d); prev_conf[:, :, :, 2] = 0
    second = module.compose_second_order_confidence(c, prev_conf, flow, flow * 2)
    assert second[:, :, :, 1].max() == 0
    assert second[:, :, :, -2:].max() == 0
    # Current coordinates x=1 sample previous x=2; this catches wrong grid use.
    previous = d.clone(); previous[:, :, :, 2] = 0.3
    current = d.clone(); current[:, :, :, 1] = 0.3
    c, _ = module.depth_motion_confidence(current, previous, flow)
    torch.testing.assert_close(c[:, :, :, 1], torch.ones_like(c[:, :, :, 1]), atol=1e-6, rtol=0)
    current[:, :, 1, 1] = float("nan")
    c, _ = module.depth_motion_confidence(current, previous, flow)
    assert torch.isfinite(c).all() and c[:, :, 1, 1].max() == 0
    print("Confidence geometry tests: PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--confidence-only", action="store_true")
    p.add_argument("--checkpoint", default="/home/larry/ssd_data/sr_project/checkpoints/causal_basicvsrpp_mv/best_model.pth")
    args = p.parse_args()
    confidence_tests(args.device)
    if args.confidence_only:
        return
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full model tests require CUDA and MMCV DCN")
    from archs.basicvsrpp.causal_basicvsrpp_mv import CausalBasicVSRPlusPlusMV
    from archs.basicvsrpp.causal_basicvsrpp_mv_depth import CausalBasicVSRPlusPlusMVDepth
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = CausalBasicVSRPlusPlusMVDepth().to(args.device).eval()
    base = CausalBasicVSRPlusPlusMV().to(args.device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    weights = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    weights = {k.removeprefix("module.").removeprefix("generator."): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    base.load_state_dict(weights, strict=True)
    assert not any("spynet" in k.lower() for k in model.state_dict())
    assert model.state_dict().keys() == base.state_dict().keys()
    x = torch.rand(1, 4, 3, 64, 64, device=args.device)
    flow = torch.rand(1, 4, 2, 64, 64, device=args.device) * 2 - 1
    depth = torch.full((1, 4, 1, 64, 64), 0.5, device=args.device)
    depth[:, 2:, :, :, 32:] = 0.51
    with torch.no_grad():
        model.gate_strength = 0
        baseline = base(x, flow)
        off = model(x, flow, depth)
        torch.testing.assert_close(off, baseline, atol=1e-6, rtol=0)
        print("C1 / C2 gate-off equivalence:", (off - baseline).abs().max().item())
        model.gate_strength = 1
        clip = model(x, flow, depth)
        state = None; ys = []
        for i in range(4):
            y, state = model.forward_step(x[:, i], flow[:, i], depth[:, i], state)
            ys.append(y)
        stream = torch.stack(ys, dim=1)
        prefix = model(x[:, :2], flow[:, :2], depth[:, :2])
        x2, f2, d2 = x.clone(), flow.clone(), depth.clone()
        x2[:, 2:] = torch.rand_like(x2[:, 2:])
        f2[:, 2:] = torch.rand_like(f2[:, 2:]) * 8
        d2[:, 2:] = torch.rand_like(d2[:, 2:])
        altered = model(x2, f2, d2)
        first, state = model(x[:, :2], flow[:, :2], depth[:, :2], return_state=True)
        rest = model(x[:, 2:], flow[:, 2:], depth[:, 2:], state=state)
        for name, a, b in (("Clip/Streaming", clip, stream),
                           ("Prefix", clip[:, :2], prefix),
                           ("Future Leakage RGB/MV/Depth", clip[:, :2], altered[:, :2]),
                           ("Chunk continuation", clip, torch.cat([first, rest], dim=1))):
            err = (a - b).abs().max().item()
            print(f"{name}: {err:.3e}")
            torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
        assert clip.shape == (1, 4, 3, 256, 256) and torch.isfinite(clip).all()
    model.train(); model.zero_grad(set_to_none=True)
    y = model(x[:, :3], flow[:, :3], depth[:, :3])
    loss = y.square().mean(); loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print("Backward finite-gradient test: PASS")


if __name__ == "__main__":
    main()

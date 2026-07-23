import torch
import torch.nn as nn
import torch.nn.functional as F

def flow_warp(x, flow, padding_mode='zeros'):
    b, c, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)
    grid.requires_grad = False
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    
    return F.grid_sample(x, vgrid_scaled, mode='bilinear', padding_mode=padding_mode, align_corners=True)

class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 7, 1, 3)
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)

class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        for level in range(5):
            ref.append(F.avg_pool2d(input=ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(input=supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            
            warp_supp = flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode='border')
            flow = flow_up + self.basic_module[level](torch.cat([ref[level], warp_supp, flow_up], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_up, w_up), mode='bilinear', align_corners=False)
        
        flow = F.interpolate(input=self.compute_flow(ref, supp), size=(h, w), mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow

class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = []
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        for _ in range(num_blocks):
            main.append(ResidualBlockNoBN(mid_channels=out_channels))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)

class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.upconv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), upsample_kernel, 1, upsample_kernel // 2)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.upconv(x))

class BasicVSRNet(nn.Module):
    def __init__(self, mid_channels=64, num_blocks=30):
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        flows_forward, flows_backward = self.compute_flow(lrs)

        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out

        return torch.stack(outputs, dim=1)

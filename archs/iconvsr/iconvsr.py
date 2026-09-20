import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接沿用你之前寫好的純淨版 BasicVSR 與 EDVR 元件，避免重複造輪子！
from archs.basicvsr.basicvsr import SPyNet, ResidualBlocksWithInputConv, PixelShufflePack, flow_warp
from archs.edvr.edvr import PCDAlignment, TSAFusion, ConvModule, make_layer, ResidualBlockNoBN

class EDVRFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, num_frames=5, deform_groups=8, num_blocks_extraction=5, center_frame_idx=2, with_tsa=True):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        self.with_tsa = with_tsa
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_blocks_extraction, mid_channels=mid_channels)

        self.feat_l2_conv1 = ConvModule(mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l2_conv2 = ConvModule(mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.feat_l3_conv1 = ConvModule(mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l3_conv2 = ConvModule(mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        
        self.pcd_alignment = PCDAlignment(mid_channels=mid_channels, deform_groups=deform_groups)
        
        if self.with_tsa:
            self.fusion = TSAFusion(mid_channels=mid_channels, num_frames=num_frames, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1, 1)
            
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        n, t, c, h, w = x.size()
        l1_feat = self.feature_extraction(self.lrelu(self.conv_first(x.view(-1, c, h, w))))
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))

        l1_feat = l1_feat.view(n, t, -1, h, w)
        l2_feat = l2_feat.view(n, t, -1, h // 2, w // 2)
        l3_feat = l3_feat.view(n, t, -1, h // 4, w // 4)

        ref_feats = [l1_feat[:, self.center_frame_idx].clone(), l2_feat[:, self.center_frame_idx].clone(), l3_feat[:, self.center_frame_idx].clone()]
        aligned_feat = [self.pcd_alignment([l1_feat[:, i].clone(), l2_feat[:, i].clone(), l3_feat[:, i].clone()], ref_feats) for i in range(t)]
        aligned_feat = torch.stack(aligned_feat, dim=1)

        return self.fusion(aligned_feat) if self.with_tsa else self.fusion(aligned_feat.view(n, -1, h, w))

class IconVSRNet(nn.Module):
    def __init__(self, mid_channels=64, num_blocks=30, keyframe_stride=5, padding=2):
        super().__init__()
        self.mid_channels = mid_channels
        self.padding = padding
        self.keyframe_stride = keyframe_stride

        self.spynet = SPyNet()
        self.edvr = EDVRFeatureExtractor(num_frames=padding * 2 + 1, center_frame_idx=padding)
        
        self.backward_fusion = nn.Conv2d(2 * mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.forward_fusion = nn.Conv2d(2 * mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(2 * mid_channels + 3, mid_channels, num_blocks)

        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def spatial_padding(self, lrs):
        n, t, c, h, w = lrs.size()
        pad_h, pad_w = (4 - h % 4) % 4, (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            lrs = lrs.view(-1, c, h, w)
            lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')
            return lrs.view(n, t, c, h + pad_h, w + pad_w), pad_h, pad_w
        return lrs, 0, 0

    def compute_refill_features(self, lrs, keyframe_idx):
        if self.padding == 2: lrs_padded = [lrs[:, [4, 3]], lrs, lrs[:, [-4, -5]]]
        elif self.padding == 3: lrs_padded = [lrs[:, [6, 5, 4]], lrs, lrs[:, [-5, -6, -7]]]
        lrs_padded = torch.cat(lrs_padded, dim=1)
        num_frames = 2 * self.padding + 1
        return {i: self.edvr(lrs_padded[:, i:i + num_frames].contiguous()) for i in keyframe_idx}

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1, lrs_2 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w), lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, lrs):
        n, t, c, h_input, w_input = lrs.size()
        lrs, pad_h, pad_w = self.spatial_padding(lrs)
        h, w = lrs.size(3), lrs.size(4)

        keyframe_idx = list(range(0, t, self.keyframe_stride))
        if keyframe_idx[-1] != t - 1: keyframe_idx.append(t - 1)

        flows_forward, flows_backward = self.compute_flow(lrs)
        feats_refill = self.compute_refill_features(lrs, keyframe_idx)

        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            lr_curr = lrs[:, i, :, :, :]
            if i < t - 1:
                feat_prop = flow_warp(feat_prop, flows_backward[:, i, :, :, :].permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = self.backward_fusion(torch.cat([feat_prop, feats_refill[i]], dim=1))
            feat_prop = self.backward_resblocks(torch.cat([lr_curr, feat_prop], dim=1))
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:
                feat_prop = flow_warp(feat_prop, flows_forward[:, i - 1, :, :, :].permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = self.forward_fusion(torch.cat([feat_prop, feats_refill[i]], dim=1))
            
            feat_prop = self.forward_resblocks(torch.cat([lr_curr, outputs[i], feat_prop], dim=1))
            out = self.conv_last(self.lrelu(self.conv_hr(self.lrelu(self.upsample2(self.lrelu(self.upsample1(feat_prop)))))))
            outputs[i] = out + self.img_upsample(lr_curr)

        out_seq = torch.stack(outputs, dim=1)
        return out_seq[:, :, :, :4 * h_input, :4 * w_input] if pad_h > 0 or pad_w > 0 else out_seq

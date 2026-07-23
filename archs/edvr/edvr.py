import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

def make_layer(block, num_blocks, **kwarg):
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.upconv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), upsample_kernel, 1, upsample_kernel // 2)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.upconv(x))

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True) if act_cfg else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

class ModulatedDCNPack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, deform_groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.deform_groups = deform_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, 1, mask)

class PCDAlignment(nn.Module):
    def __init__(self, mid_channels=64, deform_groups=8):
        super().__init__()
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
            if i == 3:
                self.offset_conv2[level] = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            else:
                self.offset_conv2[level] = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
                self.offset_conv3[level] = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            self.dcn_pack[level] = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1, deform_groups=deform_groups)
            if i < 3:
                self.feat_conv[level] = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg if i == 2 else None)

        self.cas_offset_conv1 = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_offset_conv2 = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_dcnpack = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1, deform_groups=deform_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neighbor_feats, ref_feats):
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([neighbor_feats[i - 1], ref_feats[i - 1]], dim=1)
            offset = self.offset_conv1[level](offset)
            if i == 3:
                offset = self.offset_conv2[level](offset)
            else:
                offset = self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1))
                offset = self.offset_conv3[level](offset)

            feat = self.dcn_pack[level](neighbor_feats[i - 1], offset)
            if i == 3: feat = self.lrelu(feat)
            else: feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))

            if i > 1:
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        offset = torch.cat([feat, ref_feats[0]], dim=1)
        offset = self.cas_offset_conv2(self.cas_offset_conv1(offset))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

class TSAFusion(nn.Module):
    def __init__(self, mid_channels=64, num_frames=5, center_frame_idx=2):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)
        
        self.temporal_attn1 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.temporal_attn2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.feat_fusion = ConvModule(num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = ConvModule(num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn2 = ConvModule(mid_channels * 2, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn3 = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn4 = ConvModule(mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn5 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.spatial_attn_l1 = ConvModule(mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_l2 = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_l3 = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_add1 = ConvModule(mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_add2 = nn.Conv2d(mid_channels, mid_channels, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        n, t, c, h, w = aligned_feat.size()
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        emb = self.temporal_attn2(aligned_feat.view(-1, c, h, w)).view(n, t, -1, h, w)

        corr_l = []
        for i in range(t):
            corr = torch.sum(emb[:, i, :, :, :] * embedding_ref, 1)
            corr_l.append(corr.unsqueeze(1))
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1)).unsqueeze(2).expand(n, t, c, h, w).contiguous().view(n, -1, h, w)
        aligned_feat = aligned_feat.view(n, -1, h, w) * corr_prob
        feat = self.feat_fusion(aligned_feat)

        attn = self.spatial_attn1(aligned_feat)
        attn = self.spatial_attn2(torch.cat([self.max_pool(attn), self.avg_pool(attn)], dim=1))
        
        attn_level = self.spatial_attn_l1(attn)
        attn_level = self.spatial_attn_l2(torch.cat([self.max_pool(attn_level), self.avg_pool(attn_level)], dim=1))
        attn_level = self.upsample(self.spatial_attn_l3(attn_level))

        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn5(self.upsample(self.spatial_attn4(attn)))
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        
        return feat * torch.sigmoid(attn) * 2 + attn_add

class EDVRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, mid_channels=64, num_frames=5, deform_groups=8, center_frame_idx=2, with_tsa=True):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        self.with_tsa = with_tsa
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, mid_channels=mid_channels)
        
        self.feat_l2_conv1 = ConvModule(mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l2_conv2 = ConvModule(mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.feat_l3_conv1 = ConvModule(mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l3_conv2 = ConvModule(mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        
        self.pcd_alignment = PCDAlignment(mid_channels=mid_channels, deform_groups=deform_groups)
        self.fusion = TSAFusion(mid_channels=mid_channels, num_frames=num_frames, center_frame_idx=self.center_frame_idx) if with_tsa else nn.Conv2d(num_frames * mid_channels, mid_channels, 1, 1)
        
        self.reconstruction = make_layer(ResidualBlockNoBN, 10, mid_channels=mid_channels)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        n, t, c, h, w = x.size()
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        l1_feat = self.feature_extraction(self.lrelu(self.conv_first(x.view(-1, c, h, w))))
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))

        l1_feat, l2_feat, l3_feat = l1_feat.view(n, t, -1, h, w), l2_feat.view(n, t, -1, h // 2, w // 2), l3_feat.view(n, t, -1, h // 4, w // 4)

        ref_feats = [l1_feat[:, self.center_frame_idx].clone(), l2_feat[:, self.center_frame_idx].clone(), l3_feat[:, self.center_frame_idx].clone()]
        aligned_feat = [self.pcd_alignment([l1_feat[:, i].clone(), l2_feat[:, i].clone(), l3_feat[:, i].clone()], ref_feats) for i in range(t)]
        aligned_feat = torch.stack(aligned_feat, dim=1)

        feat = self.fusion(aligned_feat) if self.with_tsa else self.fusion(aligned_feat.view(n, -1, h, w))
        out = self.conv_last(self.lrelu(self.conv_hr(self.lrelu(self.upsample2(self.lrelu(self.upsample1(self.reconstruction(feat))))))))
        return out + self.img_upsample(x_center)

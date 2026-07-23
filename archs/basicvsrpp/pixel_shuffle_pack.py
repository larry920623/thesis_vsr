import torch.nn as nn

class PixelShufflePack(nn.Module):
    """純 PyTorch 版的 PixelShufflePack"""
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        return x

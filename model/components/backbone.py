from torch import nn
from typing import Literal

__all__ = ["nnBackbone", "SpodBackbone", "No_backbone_ST"]


class nnBackbone(nn.Module):
    def __init__(self):
        super(nnBackbone, self).__init__()


class SpodBackbone(nnBackbone):
    def __init__(self):
        super(SpodBackbone, self).__init__()


class AdaptivePadding(nn.Module):
    def __init__(
            self,
            kernel_size: tuple[int, int] = (1, 1),
            stride: tuple[int, int] = (1, 1),
            dilation: tuple[int, int] = (1, 1),
            padding: Literal["corner", "same"] = "corner"
    ):
        assert padding in ("corner", "same")
        super(AdaptivePadding, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == "same":
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 30, embed_dims: int = 256):
        super(PatchEmbed, self).__init__()
        self.embed_dims = embed_dims

        self.adap_padding = AdaptivePadding()
        self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dims)

        self.init_input_size = None
        self.init_out_size = None


class No_backbone_ST(nnBackbone):
    def __init__(
            self,
            in_channels=30,
            embed_dims=256,
            strides=(1, 2, 2, 4),
            patch_size=(1, 2, 2, 4),
            num_levels=2
    ):
        super(No_backbone_ST, self).__init__()
        assert strides[0] == patch_size[0], "Use non-overlapping patch embed."
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        super(No_backbone_ST, self).__init__()

        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dims=embed_dims)
        self.num_levels = num_levels
        self.conv = nn.Conv2d(in_channels, embed_dims, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, embed_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dims, embed_dims),
            nn.LeakyReLU(0.2)
        )
        self.norm = nn.LayerNorm(embed_dims)

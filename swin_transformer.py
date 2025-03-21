from typing import Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.utils.checkpoint as checkpoint
from mmengine.model import  constant_init,kaiming_init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import torch.nn.functional as F

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=3):  #reduction=16
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)

class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):

    def __init__(self, dim,  window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  #(64,49,96)>>(16,49,192)>>(4,49,384)>>(1,49,768)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)   #(1,56,56,96)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C  (64,49,96)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows,mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        #window_sizes = [window_size if not is_slide else input_resolution[0] for i in range(depth)]  #//
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

class MixerBlock(nn.Module):

    def __init__(
        self,
        dim,
        seq_len,
        mlp_ratio=(0.5, 4.0),
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        tokens_dim = int(mlp_ratio[0] * dim)
        channels_dim = int(mlp_ratio[1] * dim)
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len,
                                    tokens_dim,
                                    act_layer=act_layer,
                                    drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim,
                                      channels_dim,
                                      act_layer=act_layer,
                                      drop=drop)
        self.channelattention = SE_Block(96)

    # def forward(self, x):
    #     shortcut1 = x
    #     x = self.norm1(x).transpose(1, 2)  #(1,96,3136)
    #     x = self.mlp_tokens(x).transpose(1, 2)
    #     x = shortcut1 + self.drop_path(x)
    #     #x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    #     shortcut2 = x
    #     x = self.norm2(x)  #(1,3136,96)
    #     x = x.reshape(1,96,56,56)
    #     x = self.channelattention(x)
    #     x = x.reshape(1,3136,96)
    #     x = self.mlp_channels(x)
    #     x = shortcut2 + self.drop_path(x)
    #     #x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    #     return x

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        # shortcut2 = x
        # x = self.norm2(x)  #(1,3136,96)
        # x = x.reshape(1,96,56,56)
        # x = self.channelattention(x)
        # x = x.reshape(1,3136,96)
        # x = self.mlp_channels(x)
        # x = shortcut2 + self.drop_path(x)
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        # shortcut1 = x
        # x = self.norm1(x).transpose(1, 2)  #(1,96,3136)
        # x = self.mlp_tokens(x).transpose(1, 2)
        # x = shortcut1 + self.drop_path(x)
        return x

    def flops(self):
        flops = 0
        # norm1
        flops += self.dim * self.seq_len  # LayerNorm for tokens

        # token-mixing MLP
        # Assuming Mlp has a hidden layer with size `tokens_dim`
        tokens_dim = int(self.mlp_ratio[0] * self.dim)
        flops += self.seq_len * tokens_dim * 2  # Linear layers (input to hidden and hidden to output), times 2 for multiply-add
        flops += tokens_dim  # Bias addition for hidden layer
        flops += self.seq_len  # Bias addition for output layer

        # norm2
        flops += self.dim * self.seq_len  # LayerNorm for channels

        # channel-mixing MLP
        # Assuming Mlp has a hidden layer with size `channels_dim`
        channels_dim = int(self.mlp_ratio[1] * self.dim)
        flops += self.dim * channels_dim * 2  # Linear layers (input to hidden and hidden to output), times 2 for multiply-add
        flops += channels_dim  # Bias addition for hidden layer
        flops += self.dim  # Bias addition for output layer

        return flops


def last_zero_init(m: Union[nn.Module, nn.Sequential]) -> None:
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)

class AF(nn.Module):
    def __init__(self):
        super(AF, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 2)) #torch.Size([480, 32, 128, 32])
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64), #torch.Size([480, 64, 128, 32])
            nn.MaxPool2d(kernel_size=(4, 1)) #torch.Size([480, 64, 32, 32])
        )

        self.p5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=(5, 3), padding=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(4, 2))
        )

        self.p7 = nn.Sequential(
            nn.Linear(15360, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
        )

    def forward(self, input):
        input = self.p1(input)
        input = self.p2(input)
        input = self.p5(input)
        input = torch.flatten(input, start_dim=-3)
        input = self.p7(input)
        return input

class Bi_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=2, rnn_type='LSTM'):
        super(Bi_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the initial linear hidden layer
        self.init_linear = nn.Linear(self.input_dim, self.input_dim)

        # Define the LSTM layer
        self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True,
                                           bidirectional=True)

    def forward(self, input):
        # Forward pass through initial hidden layer
        linear_input = self.init_linear(input)

        lstm_out, self.hidden = self.lstm(linear_input)

        return lstm_out

class SwinTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        #self.mlp_ratio = mlp_ratio
        self.swin_mlp_ratio = 4.0
        self.mixer_mlp_ratio = (0.5, 4.0)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.swin_mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        def create_mixer_block(embed_dim, num_patches, mlp_ratio, mlp_layer, norm_layer, act_layer, drop, drop_path):
            return MixerBlock(
                dim=embed_dim,
                seq_len=num_patches,
                mlp_ratio=mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=drop,
                drop_path=drop_path
            )

        self.blocks = nn.ModuleList()
        for i_layer in range(8):
            block = create_mixer_block(
                embed_dim=embed_dim,  # 假设所有层使用相同的嵌入维度
                num_patches=num_patches,  # 从PatchEmbed获得的补丁数量
                mlp_ratio=self.mixer_mlp_ratio,
                mlp_layer=Mlp,  # 假设Mlp已经被定义
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                drop=drop_rate,
                drop_path=drop_path_rate * i_layer / (sum(depths) - 1)  # 按层分配drop_path_rate
            )
            self.blocks.append(block)
        # self.blocks = nn.Sequential(*[
        #     create_mixer_block(
        #         embed_dim=embed_dim,  # 假设所有层使用相同的嵌入维度
        #         num_patches=num_patches,  # 从PatchEmbed获得的补丁数量
        #         mlp_ratio=self.mixer_mlp_ratio,
        #         mlp_layer=Mlp,  # 假设Mlp已经被定义
        #         norm_layer=norm_layer,
        #         act_layer=nn.GELU,
        #         drop=drop_rate,
        #         drop_path=drop_path_rate * i_layer / (sum(depths) - 1)  # 按层分配drop_path_rate
        #     ) for i_layer in range(8)
        # ])

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.quality = nn.Linear(3200, 1)
        self.apply(self._init_weights)
        self.AF = AF()
        self.SE = SE_Block(3)
        self.ECA = ECANet(3)
        self.ATP_BiL1 = Bi_RNN(input_dim=512, hidden_dim=128)
        self.ATP_BiL2 = Bi_RNN(input_dim=256, hidden_dim=64)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, x_3D_features,x_audio):
    #def forward_features(self, x, x_3D_features):
        #(1,T,3,224,224)
        x_size = x.shape  #(1,10,3,224,224)
        x_3D_features_size = x_3D_features.shape  #(1,10,2048+256)
        x_audio_size = x_audio.shape  #(1,10,30,4,128,62)
        x = x.view(-1, x_size[2], x_size[3], x_size[4])    #(1*10,3,224,224)
        x_3D_features = x_3D_features.view(-1, x_3D_features_size[2])  #(1*10,2304)
        x_audio_reshape = x_audio.view(-1, x_audio_size[-3],x_audio_size[-2],x_audio_size[-1])  #(300,4,128,62)

        x_audio = self.AF(x_audio_reshape)  #(300,1024)
        x_audio = x_audio.view(-1,x_audio_size[2],x_audio.shape[-1])  #(10,30,1024)/(8,30,512)
        x_audio = self.ATP_BiL1(x_audio)  #(10,30,256)
        x_audio = self.ATP_BiL2(x_audio)  #(10,30,128)
        x_audio = x_audio.reshape(-1, x_audio.shape[-1], x_audio_size[2])  #(10,128,30)
        x = self.ECA(x)
        x_ = self.patch_embed(x)   #(1,3136,96)
        if self.ape:
            x = x + self.pos_embed
        y = self.pos_drop(x_)   #(1,3136,96)
        for layer in self.layers:
            y = layer(y)  #[0]>>(1,784,192),[1]>>(1,196,384),[2]>>(1,49,768),[3]=[2]
        y = self.norm(y)  # B L C    #(1,49,768)

        y = y.reshape(6,768,7,7)

        y = self.avg_pool(y)
        y = torch.flatten(y, 1)
        x_audio = torch.mean(x_audio,dim=2)
        y = torch.cat((y, x_3D_features,x_audio), dim=1)  #(10,768+2048+256+1024)=3584/1024>>128=3200
        #y = torch.cat((y, x_3D_features), dim=1)
        y = self.quality(y)
        y = y.view(x_size[0], x_size[1])
        y = torch.mean(y, dim=1)

        return y

    def forward(self, x ,x_3D_features, x_audio):
        x = self.forward_features(x ,x_3D_features, x_audio)
        return x

    # def forward(self, x ,x_3D_features):
    #     x = self.forward_features(x ,x_3D_features)
    #     return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        for i,block in enumerate(self.blocks):
            flops += block.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
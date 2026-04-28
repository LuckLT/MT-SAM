from functools import partial

from torch import nn
from torch.nn import functional as F
import torch
import os
import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg

try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
# from .registry import register_pip_model
# from .registry import register_pip_model  # debug use only
from pathlib import Path
# from nnunetv2.networks.SAM_lora_model import SAM_lora_model_new
from nnunetv2.networks.Mamba_vision.HMT_Unet import HMTUNet
from nnunetv2.networks.ma_sam.model.segmentation_sub_network.image_encoder import ImageEncoderViT
from nnunetv2.networks.ma_sam.model.segmentation_sub_network.mask_decoder import MaskDecoder
from nnunetv2.networks.ma_sam.model.segmentation_sub_network.positional_embedding import PositionalEmbedding
from nnunetv2.networks.ma_sam.model.segmentation_sub_network.prompt_encoder import PromptEncoder
from nnunetv2.networks.ma_sam.model.segmentation_sub_network.transformer import TwoWayTransformer
import os
import selective_scan_cuda
import torch._dynamo as dynamo
from torch._dynamo import OptimizedModule

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


class ConvBlock(nn.Module):
    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        # 注意测试
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # 这里的x 就是 不用先silu
        x = F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2)
        x = x*z # Hadamard product simzhang added

        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class Upsample(nn.Module):
    """
    Up-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = dim // 2
            self.expansion = nn.Sequential(
                nn.ConvTranspose2d(dim, dim_out, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False),
                # nn.BatchNorm2d(dim_out)  # Optionally add batch normalization after upsampling
            )

    def forward(self, x):
        x = self.expansion(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x



class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(d_model=dim,
                                          d_state=8,
                                          d_conv=3,
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                         for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i,
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                         for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)


class PatchExpand2D_sim(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # b h w c
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # b c h w

        return x


class MambaVisionLayer_up(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 upsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            upsample: bool argument for up-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                         for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i,
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                         for i in range(depth)])
            self.transformer_block = True

        self.upsample = None if not upsample else PatchExpand2D_sim(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)

        _, _, H, W = x.shape
        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # b h w c
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # b c h w

        return x


class MambaVision_sim(nn.Module):
    """
    MambaVision sim for unet,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        self.pos_drop = nn.Dropout(p=drop_rate)  # train drop rate
        self.encoder_depths = depths
        self.decoder_depths = depths[::-1]

        for i in range(len(depths)):  # i --> 0,1,2,3
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i),  # up layer 和 这个layer 的dim 是反过来的
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[
                                                                                                          i] % 2 != 0 else list(
                                         range(depths[i] // 2, depths[i])),
                                     )
            self.levels.append(level)

        # dim=int( dim * 2 ** (3-i) )
        # depth = depths[3-i]
        # num_heads=num_heads[3-i]
        # window_size=window_size[3-i]
        # drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])]
        self.layers_up = nn.ModuleList()
        for i in range(len(depths)):
            up_layer = MambaVisionLayer_up(dim=int(dim * 2 ** (3 - i)),
                                           depth=depths[3 - i],
                                           num_heads=num_heads[3 - i],
                                           window_size=window_size[3 - i],
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           conv=conv,
                                           drop=drop_rate,
                                           attn_drop=attn_drop_rate,
                                           drop_path=dpr[sum(self.decoder_depths[:i]):sum(self.decoder_depths[:i + 1])],
                                           upsample=(i != 0),
                                           layer_scale=layer_scale,
                                           layer_scale_conv=layer_scale_conv,
                                           transformer_blocks=list(range(depths[3 - i] // 2 + 1, depths[3 - i])) if
                                           depths[3 - i] % 2 != 0 else list(range(depths[3 - i] // 2, depths[3 - i])),
                                           )
            self.layers_up.append(up_layer)

        # classification
        # self.norm = nn.BatchNorm2d(num_features)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.norm_layer = nn.LayerNorm
        self.final_up = Final_PatchExpand2D(dim=dim, dim_scale=4, norm_layer=self.norm_layer)
        self.final_conv = nn.Conv2d(dim // 4, num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)  # [2,3,224,224] -> [2,80,56,56]
        x = self.pos_drop(x)
        # each layer output shape print:
        # level 0: [2,80,56,56]          level 1: [2,160,28,28]
        # level 2: [2,320,14,14]         level 3: [2,640,7,7]
        for level in self.levels:
            skip_list.append(x)
            x = level(x)
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])

        return x

    def forward_final(self, x):
        x = self.final_up(x)
        # x = x.permute(0,3,1,2)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)  # encoder
        x = self.forward_features_up(x, skip_list)  # decoder
        x = self.forward_final(x)    # segmentation head, no nonlinear layer

        # if self.num_classes == 1:
        #     return torch.sigmoid(x)
        return x, skip_list



class HMTUNet(nn.Module):

    def __init__(self,
                 input_channels=3,
                 num_classes=2,
                 depths=[1, 3, 8, 4],
                 num_heads=[2, 4, 8, 16],
                 window_size=[8, 8, 14, 7],
                 dim=80,
                 in_dim=32,
                 mlp_ratio=4,
                 resolution=224,
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 **kwargs):

        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.hmtunet = MambaVision_sim(
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            dim=dim,
            in_dim=in_dim,
            mlp_ratio=mlp_ratio,
            resolution=resolution,
            drop_path_rate=drop_path_rate,
        )

        # load pretrained model
        # modelpath = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_results/Dataset601_Loc_pro/nnUNetTrainer_MambaVision__nnUNetPlans__2d/fold_all/checkpoint_best_1.pth'
        # state_dict1 = torch.load(modelpath, map_location='cuda')['network_weights']
        # self.load_state_dict(state_dict1)

        # required_grad = False
        # for param in self.hmtunet.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        return self.hmtunet(x)  # no nonlinear layer

    def forward_encoder(self, x):
        x, skip_list = self.hmtunet.forward_features(x)
        return x, skip_list

    def forward_decoder(self, x, skip_list):
        return self.hmtunet.forward_features_up(x, skip_list)

    def forward_final(self, x):
        return self.hmtunet.forward_final(x)

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.hmtunet.state_dict()
            model_checkpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = model_checkpoint['state_dict']
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.hmtunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.hmtunet.state_dict()
            model_checkpoint = torch.load(self.load_ckpt_path)
            pretrained_order_dict = model_checkpoint['state_dict']
            pretrained_dict = {}
            for k, v in pretrained_order_dict.items():
                if 'levels.0' in k:
                    new_k = k.replace('levels.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'levels.1' in k:
                    new_k = k.replace('levels.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'levels.2' in k:
                    new_k = k.replace('levels.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'levels.3' in k:
                    new_k = k.replace('levels.3', 'layers_up.0')
                    pretrained_dict[new_k] = v

            # decoder
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.hmtunet.load_state_dict(model_dict)

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")

class Mamba_SAM(nn.Module):
    def __init__(self, num_classes, num_atlas, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 encoder_global_attn_indexes=(2, 5, 8, 11), image_size=512, prompt_embed_dim=256, vit_patch_size=16,
                 adapterTrain=True, input_channels=3,
                 # num_classes=2,
                 depths=[1, 3, 8, 4],
                 num_heads=[2, 4, 8, 16],
                 window_size=[8, 8, 14, 7],
                 dim=80,
                 in_dim=32,
                 mlp_ratio=4,
                 resolution=224,
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 **kwargs):

        super().__init__()

        # mambavision

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.hmtunet = MambaVision_sim(
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            dim=dim,
            in_dim=in_dim,
            mlp_ratio=mlp_ratio,
            resolution=resolution,
            drop_path_rate=drop_path_rate,
        )

        # load pretrained model
        modelpath = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_results/Dataset601_Loc_pro/nnUNetTrainer_MambaVision__nnUNetPlans__2d/fold_all/checkpoint_best_1.pth'
        state_dict1 = torch.load(modelpath, map_location='cuda')['network_weights']
        self.load_state_dict(state_dict1)

        image_embedding_size = image_size // vit_patch_size

        self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                in_chans=3,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
                adapter_train=adapterTrain
            )

        self.prompt_encoder = PromptEncoder(
                in_channel=num_atlas,
                out_channel=prompt_embed_dim,
            )

        self.mask_decoder = MaskDecoder(
                transformer_dim=prompt_embed_dim,
                numClasses=num_classes,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
            )

        self.pe = PositionalEmbedding(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size)
            )

    def freeze_layers(self, num_layers=100):
        # 冻结前 num_layers 层
        # layer_count = 0
        for name, param in self.hmtunet.named_parameters():
            # if layer_count < num_layers:
            param.requires_grad = False
                # layer_count += 1
            # else:
            #     break
    def forward(self, img1):
        # with torch.no_grad():
        x, skip_list = self.hmtunet(img1)
        x = x[:, 1, :, :].unsqueeze(1)   # (B, 1, 512, 512)
        # x = F.interpolate(x, size=(512, 512), mode='nearest')
        # img1 = img1[:, 0, :, :].unsqueeze(1)
        # img1 = img1.repeat(1, 3, 1, 1)
        # # img1 =  F.interpolate(img1, size=(512, 512), mode='nearest')
        # print('img.shape', img1.shape)

        imgFeatures = self.image_encoder(img1, skip_list)
        # print('imgFeatures.shape', imgFeatures.shape)    #imgFeatures.shape torch.Size([1, 256, 32, 32])
        # prompt = F.interpolate(prompt, size=(512, 512), mode='nearest')
        # print('prompt.shape', x.shape)   # prompt.shape torch.Size([1, 1, 224, 224])
        # prompt = x.float()
        promptFeatures, skip = self.prompt_encoder(x)
        # print('promptFeatures.shape', promptFeatures.shape)   #  torch.Size([1, 256, 16, 16])
        mask = self.mask_decoder(image_embeddings=imgFeatures,
                                pe=self.pe(),
                                prompt_embeddings=promptFeatures,
                                prompt_skip=skip)

        return mask


def froze(net):
    for n, value in net.module.image_encoder.named_parameters():
        if "Adapter" in n:
            value.requires_grad = True
        else:
            value.requires_grad = False



def load_from(net_dict, state_dicts, image_size, vit_patch_size):
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dicts.items() if
                      k in net_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:

        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in net_dict.keys() if 'rel_pos' in k]

        global_rel_pos_keys = [k for k in rel_pos_keys if
                                                        '2' in k or
                                                        '5' in k or
                                                        '7' in k or
                                                        '8' in k or
                                                        '11' in k or
                                                        '13' in k or
                                                        '15' in k or
                                                        '23' in k or
                                                        '31' in k]
        for k in global_rel_pos_keys:
            h_check, w_check = net_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(rel_pos_params, (h_check, w_check), mode='bilinear', align_corners=False)

            new_state_dict[k] = rel_pos_params[0, 0, ...]
    net_dict.update(new_state_dict)
    return net_dict

#
# img = torch.rand((1, 3, 224, 224), dtype=torch.float32).cuda()
# # taget = torch.rand((1, 1, 224, 224), dtype=torch.float32).cuda()
# net = Mamba_SAM(num_classes=2, num_atlas=1, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
#                  encoder_global_attn_indexes=(2, 5, 8, 11), image_size=224, prompt_embed_dim=256, vit_patch_size=16,
#                  adapterTrain=True).cuda()
# pred = net(img)
# print(pred.shape)
# pred_softmax = torch.nn.Softmax(dim=1)(pred)
# print(pred_softmax.shape)

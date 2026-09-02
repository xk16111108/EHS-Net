import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    try:
        from selective_scan import selective_scan_fn
    except ImportError as selective_scan_import_error:
        raise ImportError(
            "EHS-Net requires mamba_ssm or a compatible selective_scan package."
        ) from selective_scan_import_error

from natten import NeighborhoodAttention2D


import numpy as np


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def create_sobel_operators(in_channels, out_channels):
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    filter_x = torch.from_numpy(filter_x.reshape((1, 1, 3, 3))).repeat(
        out_channels, in_channels, 1, 1
    )
    filter_y = torch.from_numpy(filter_y.reshape((1, 1, 3, 3))).repeat(
        out_channels, in_channels, 1, 1
    )
    sobel_x = nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )
    sobel_x.weight = nn.Parameter(filter_x, requires_grad=False)
    sobel_y = nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )
    sobel_y.weight = nn.Parameter(filter_y, requires_grad=False)
    return sobel_x, sobel_y


def compute_sobel_magnitude(conv_x, conv_y, x):
    g_x = conv_x(x)
    g_y = conv_y(x)

    return torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + 1e-6)


class EAM(nn.Module):
    def __init__(self, in_dim1, in_dim4):
        super().__init__()
        self.reduce1 = ConvBNReLU(in_dim1, 64)
        self.reduce4 = ConvBNReLU(in_dim4, 64)
        self.block = nn.Sequential(ConvBNReLU(128, 64, 3), nn.Conv2d(64, 1, 1))

    def forward(self, x1_sobel, x4_sobel):
        x1 = self.reduce1(x1_sobel)
        x4 = F.interpolate(
            self.reduce4(x4_sobel),
            size=x1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.block(torch.cat((x1, x4), dim=1))


class PatchEmbed2D(nn.Module):
    def __init__(
        self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(
                f"Warning, x.shape {x.shape} is not match even ===========", flush=True
            )
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x1 = x1[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x2 = x2[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x3 = x3[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, input_dim, output_dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_scale = dim_scale

        self.expand = nn.Linear(
            input_dim, output_dim * (self.dim_scale**2), bias=False
        )
        self.norm = norm_layer(output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        x = self.expand(x)

        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=self.output_dim,
        )
        x = self.norm(x)
        return x


class FinalPatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // self.dim_scale,
        )
        x = self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )

        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        kernel_size=7,
        num_heads=4,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs
        )
        self.drop_path = DropPath(drop_path)
        self.local_attn = NeighborhoodAttention2D(
            dim=hidden_dim,
            kernel_size=kernel_size,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=0.0,
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        shortcut = x
        x_norm = self.ln_1(x)

        x_local = self.local_attn(x_norm)

        x_global = self.self_attention(x_norm)

        g = self.gate(x_norm)

        out = g * x_local + (1 - g) * x_global

        out = shortcut + self.drop_path(out)
        return out


class VSSLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
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


class VSSLayerUp(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

        if upsample is not None:
            self.upsample = upsample
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max")):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.adaptive_avg_pool2d(x, output_size=1)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.adaptive_max_pool2d(x, output_size=1)
                channel_att_raw = self.mlp(max_pool)
            else:
                raise ValueError(f"Unsupported channel pooling type: {pool_type}")

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max")):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class S3Bridge(nn.Module):
    def __init__(self, dim_lo, dim_hi, dilations=(1, 2, 3), heads=4):
        super().__init__()
        self.dilated = nn.ModuleList(
            [
                nn.Conv2d(dim_hi, dim_hi, 3, padding=d, dilation=d, groups=dim_hi)
                for d in dilations
            ]
        )

        self.proj_q = nn.Linear(dim_lo, dim_lo, bias=False)
        self.proj_k = nn.Linear(dim_hi, dim_lo, bias=False)
        self.proj_v = nn.Linear(dim_hi, dim_lo, bias=False)
        self.num_heads = heads
        self.scale = (dim_lo // heads) ** -0.5
        self.proj_out = nn.Linear(dim_lo, dim_lo, bias=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim_lo * 2, dim_lo, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_lo),
            nn.ReLU(inplace=True),
        )

    def forward(self, d_lo, d_hi, edge):
        B, C, H, W = d_lo.shape

        d_hi = F.interpolate(d_hi, size=(H, W), mode="bilinear", align_corners=False)
        size_feat = sum([conv(d_hi) for conv in self.dilated]) / len(self.dilated)

        q = self.proj_q(d_lo.flatten(2).transpose(1, 2))
        k = self.proj_k(size_feat.flatten(2).transpose(1, 2))
        v = self.proj_v(size_feat.flatten(2).transpose(1, 2))

        q = rearrange(q, "b n (h c) -> b h n c", h=self.num_heads)
        k = rearrange(k, "b n (h c) -> b h n c", h=self.num_heads)
        v = rearrange(v, "b n (h c) -> b h n c", h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if edge is not None:
            edge_resized = F.interpolate(
                edge, size=(H, W), mode="bilinear", align_corners=False
            )
            edge_flat = edge_resized.view(B, 1, H * W)
            edge_bias = edge_flat.unsqueeze(2)
            attn = attn + edge_bias

        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n c -> b n (h c)")
        out = self.proj_out(out).transpose(1, 2).view(B, C, H, W)

        fused = self.fuse(torch.cat([d_lo, out], dim=1))
        return fused + d_lo


class EHSNetArchitecture(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1,
        depths=(2, 2, 2, 2),
        depths_decoder=(2, 2, 2, 1),
        dims=(96, 192, 384, 768),
        d_state=16,
        drop_path_rate=0.1,
        encoder_block_types=("local", "local", "hybrid", "hybrid"),
        attn_drop_rate=0.0,
        drop_rate=0.0,
        patch_norm=True,
        use_checkpoint=False,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_layers = len(depths)
        self.dims = dims
        self.patch_embed = PatchEmbed2D(
            patch_size, in_chans, dims[0], norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        dp_rate = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i < self.num_layers - 1) else None,
                block_type=encoder_block_types[i],
                drop_path=dp_rate[sum(depths[:i]) : sum(depths[: i + 1])],
                attn_drop_rate=attn_drop_rate,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        dims_dec = dims[::-1]
        self.layers_up = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.cbams = nn.ModuleList()

        for i in range(self.num_layers):
            if i == 0:
                up = nn.Identity()
            else:
                self.cbams.append(CBAM(dims[self.num_layers - 1 - i]))
                self.skip_proj.append(
                    nn.Linear(dims[self.num_layers - 1 - i], dims_dec[i])
                )
                up = PatchExpand2D(dims_dec[i - 1], dims_dec[i], norm_layer=norm_layer)

            layer_up = VSSLayerUp(
                dim=dims_dec[i],
                depth=depths_decoder[i],
                d_state=d_state,
                upsample=up,
                block_type="hybrid",
                drop_path=dp_rate[::-1][
                    sum(depths_decoder[:i]) : sum(depths_decoder[: i + 1])
                ],
                attn_drop_rate=attn_drop_rate,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer_up)

        self.s3bridges = nn.ModuleList(
            [
                S3Bridge(dim_lo=dims_dec[1], dim_hi=dims_dec[0]),
                S3Bridge(dim_lo=dims_dec[2], dim_hi=dims_dec[1]),
            ]
        )

        self.sobel_x1, self.sobel_y1 = create_sobel_operators(dims[0], dims[0])
        self.sobel_x4, self.sobel_y4 = create_sobel_operators(dims[-1], dims[-1])
        self.eam = EAM(dims[0], dims[-1])

        self.final_up = FinalPatchExpand2D(
            dim=dims[0], dim_scale=patch_size, norm_layer=norm_layer
        )
        self.final_conv = nn.Conv2d(dims[0] // patch_size, num_classes, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        ori_size = x.shape[2:]
        skips, x_patch = [], self.patch_embed(x)
        x_patch = self.pos_drop(x_patch)

        for layer in self.layers:
            skips.append(x_patch)
            x_patch = layer(x_patch)

        s1 = compute_sobel_magnitude(
            self.sobel_x1,
            self.sobel_y1,
            skips[0].permute(0, 3, 1, 2),
        )
        s4 = compute_sobel_magnitude(
            self.sobel_x4,
            self.sobel_y4,
            x_patch.permute(0, 3, 1, 2),
        )
        edge_logits = self.eam(s1, s4)
        edge_att = torch.sigmoid(edge_logits)

        dec = x_patch
        prev_feat = None
        for i, layer_up in enumerate(self.layers_up):
            dec = layer_up.upsample(dec)

            if i > 0:
                skip = skips[self.num_layers - 1 - i]
                skip_ref = self.cbams[i - 1](skip.permute(0, 3, 1, 2)).permute(
                    0, 2, 3, 1
                )
                skip_ref = self.skip_proj[i - 1](skip_ref)
                ea = F.interpolate(
                    edge_att,
                    size=skip_ref.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                )
                dec = dec + skip_ref * ea.permute(0, 2, 3, 1)

            if i == 1:
                dec = self.s3bridges[0](
                    dec.permute(0, 3, 1, 2),
                    prev_feat.permute(0, 3, 1, 2),
                    F.interpolate(
                        edge_att,
                        size=dec.shape[1:3],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ).permute(0, 2, 3, 1)
            if i == 2:
                dec = self.s3bridges[1](
                    dec.permute(0, 3, 1, 2),
                    prev_feat.permute(0, 3, 1, 2),
                    F.interpolate(
                        edge_att,
                        size=dec.shape[1:3],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ).permute(0, 2, 3, 1)

            prev_feat = dec

            for blk in layer_up.blocks:
                dec = blk(dec)

        mask = self.final_up(dec)
        mask_logits = self.final_conv(mask.permute(0, 3, 1, 2))
        return (
            F.interpolate(
                mask_logits, size=ori_size, mode="bilinear", align_corners=False
            ),
            F.interpolate(
                edge_logits, size=ori_size, mode="bilinear", align_corners=False
            ),
        )

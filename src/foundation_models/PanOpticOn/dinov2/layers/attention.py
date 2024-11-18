# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QueryCrossAttention(nn.Module):
    """ minor modification of Attention to allow for separate query. 
        Memory efficient version only for self attention possible. 
       """
    def __init__(
            self, 
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.inproj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.inproj_kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, key_padding_mask=None):
        """ q: (B, Nq, D), kv: (B, Nkv, D), key_padding_mask: (B, Nkv)"""

        B, Nq, D = q.shape
        q = self.inproj_q(q).reshape(B, Nq, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        B, Nkv, D = kv.shape
        kv = self.inproj_kv(kv).reshape(B, Nkv, 2, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # (B, num_heads, Nkv, D//self.num_heads)

        attn = q @ k.transpose(-2, -1)  # shape: (B, num_heads, Nq, Nkv)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nkv)
            attn = attn.masked_fill(key_padding_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
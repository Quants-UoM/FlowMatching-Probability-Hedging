# architecture.py

from typing import Optional, Tuple

import math
import torch
from torch import nn, Tensor


# -------- basic blocks --------

class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    """
    Root mean square normalisation.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.scale * x_normed


class RotaryEmbedding(nn.Module):
    """
    Simple RoPE for attention heads.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Returns cos and sin, each with shape (1, 1, seq_len, dim)
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    x: (batch, heads, seq_len, head_dim)
    cos, sin: (1, 1, seq_len, head_dim)
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1)
    x_rot = x_rot.reshape_as(x)
    return x * cos + x_rot * sin


class MultiHeadAttentionSigmoid(nn.Module):
    """
    Multi head attention with QK RMSNorm, RoPE and sigmoid gating on logits.
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert dim_model % num_heads == 0
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

        self.q_norm = RMSNorm(dim_model)
        self.k_norm = RMSNorm(dim_model)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        x: (batch, seq_len_q, dim_model)
        context: (batch, seq_len_kv, dim_model) or None (self attention)
        mask: (batch, 1, seq_len_q, seq_len_kv) or None
        """
        if context is None:
            context = x

        B, Lq, D = x.shape
        Bc, Lk, Dk = context.shape
        assert B == Bc and D == Dk

        q = self.q_proj(self.q_norm(x))       # (B, Lq, D)
        k = self.k_proj(self.k_norm(context)) # (B, Lk, D)
        v = self.v_proj(context)              # (B, Lk, D)

        # reshape to heads
        q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Lq, Hd)
        k = k.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Lk, Hd)
        v = v.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Lk, Hd)

        if self.use_rope:
            device = x.device
            max_len = max(Lq, Lk)
            cos, sin = self.rope(max_len, device=device)
            cos_q, sin_q = cos[:, :, :Lq, :], sin[:, :, :Lq, :]
            cos_k, sin_k = cos[:, :, :Lk, :], sin[:, :, :Lk, :]
            q = apply_rotary(q, cos_q, sin_q)
            k = apply_rotary(k, cos_k, sin_k)

        # attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, Lq, Lk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # sigmoid attention: gate then renormalise
        weights = torch.sigmoid(scores)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        attn = torch.matmul(weights, v)  # (B, H, Lq, Hd)
        attn = attn.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.out_proj(attn)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim_model: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# -------- encoder and decoder layers --------

class FlowTSEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.attn = MultiHeadAttentionSigmoid(
            dim_model=dim_model,
            num_heads=num_heads,
            use_rope=use_rope,
        )
        self.norm1 = RMSNorm(dim_model)
        self.ff = FeedForward(dim_model, dim_ff, dropout=dropout)
        self.norm2 = RMSNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # self attention
        h = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        h = h + self.dropout(self.ff(self.norm2(h)))
        return h


class FlowTSDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttentionSigmoid(
            dim_model=dim_model,
            num_heads=num_heads,
            use_rope=use_rope,
        )
        self.cross_attn = MultiHeadAttentionSigmoid(
            dim_model=dim_model,
            num_heads=num_heads,
            use_rope=False,
        )
        self.norm1 = RMSNorm(dim_model)
        self.norm2 = RMSNorm(dim_model)
        self.ff = FeedForward(dim_model, dim_ff, dropout=dropout)
        self.norm3 = RMSNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # self attention on query tokens
        h = x + self.dropout(self.self_attn(self.norm1(x), mask=self_mask))
        # cross attention to encoder memory
        h = h + self.dropout(
            self.cross_attn(self.norm2(h), context=memory, mask=cross_mask)
        )
        # feed forward
        h = h + self.dropout(self.ff(self.norm3(h)))
        return h


# -------- time embedding and full transformer --------

class TimeEmbedding(nn.Module):
    """
    Simple MLP time or tau embedding.
    """

    def __init__(self, dim_time: int):
        super().__init__()
        self.lin1 = nn.Linear(1, dim_time)
        self.act = Swish()
        self.lin2 = nn.Linear(dim_time, dim_time)

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) or (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        h = self.lin1(t)
        h = self.act(h)
        h = self.lin2(h)
        return h


class FlowTSVelocityNet(nn.Module):
    """
    Flow TS style encoder decoder that maps a feature window to a velocity vector.

    Input:
        x_window: (batch, L, dim_x)
        t: (batch,) or (batch, 1) time or tau
    Output:
        v: (batch, dim_x)
    """

    def __init__(
        self,
        dim_x: int,
        window_len: int,
        dim_model: int = 128,
        num_heads: int = 4,
        num_layers_enc: int = 3,
        num_layers_dec: int = 2,
        num_registers: int = 2,
        dim_ff: int = 256,
        dim_time: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim_x = dim_x
        self.window_len = window_len
        self.dim_model = dim_model
        self.num_registers = num_registers

        # project features to model dimension
        self.in_proj = nn.Linear(dim_x, dim_model)

        # learned register tokens
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_registers, dim_model)
            )
        else:
            self.register_tokens = None

        # encoder stack
        self.encoder_layers = nn.ModuleList(
            [
                FlowTSEncoderLayer(
                    dim_model=dim_model,
                    num_heads=num_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                    use_rope=True,
                )
                for _ in range(num_layers_enc)
            ]
        )

        # time embedding and projection to query space
        self.time_embed = TimeEmbedding(dim_time)
        self.time_to_model = nn.Linear(dim_time, dim_model)

        # learned base query token
        self.query_token = nn.Parameter(torch.zeros(1, 1, dim_model))

        # decoder stack
        self.decoder_layers = nn.ModuleList(
            [
                FlowTSDecoderLayer(
                    dim_model=dim_model,
                    num_heads=num_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                    use_rope=False,
                )
                for _ in range(num_layers_dec)
            ]
        )

        # output projection to velocity
        self.out_proj = nn.Linear(dim_model, dim_x)

    def forward(self, x_window: Tensor, t: Tensor) -> Tensor:
        """
        x_window: (B, L, dim_x)
        t: (B,) or (B, 1)
        returns v: (B, dim_x)
        """
        B, L, Dx = x_window.shape
        assert L == self.window_len, f"Expected window_len {self.window_len}, got {L}"
        assert Dx == self.dim_x

        # encoder input
        h = self.in_proj(x_window)  # (B, L, dim_model)

        # append register tokens
        if self.num_registers > 0:
            reg = self.register_tokens.expand(B, self.num_registers, self.dim_model)
            h = torch.cat([h, reg], dim=1)  # (B, L + R, dim_model)

        # encoder pass
        for layer in self.encoder_layers:
            h = layer(h)

        memory = h  # (B, L + R, dim_model)

        # build decoder query token, conditioned on time
        t_emb = self.time_embed(t)                   # (B, dim_time)
        t_proj = self.time_to_model(t_emb)          # (B, dim_model)
        q_base = self.query_token.expand(B, 1, self.dim_model)  # (B, 1, dim_model)
        q = q_base + t_proj.unsqueeze(1)            # (B, 1, dim_model)


        for layer in self.decoder_layers:
            q = layer(q, memory)

        q_final = q.squeeze(1)          # (B, dim_model)
        v = self.out_proj(q_final)      # (B, dim_x)
        return v


# -------- wrapper for Meta Flow Matching --------

class FlowTSVelocityWrapper(nn.Module):
    """
    Wrapper around FlowTSVelocityNet so it looks like a simple model
    that takes (x_window, t) and returns a velocity vector.

    x is interpreted as a window of features with shape (batch, L, dim_x)
    and t is the scalar time index or tau per batch element.
    """

    def __init__(
        self,
        dim_x: int,
        window_len: int,
        **net_kwargs,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.window_len = window_len
        self.net = FlowTSVelocityNet(
            dim_x=dim_x,
            window_len=window_len,
            **net_kwargs,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        x: (batch, L, dim_x)
        t: (batch,) or (batch, 1)
        returns velocity: (batch, dim_x)
        """
        return self.net(x, t)

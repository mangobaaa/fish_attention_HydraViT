import torch
import torch.nn as nn
import torch.nn.functional as F
from .hydravit_layers import QKVLinear, Linear

class FiSHAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_global_heads: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        stochastic: bool = True,
        hard: bool = False,
        tau: float = 1.0,
    ):
        super().__init__()
        self.stochastic = stochastic
        self.hard = hard
        self.tau = tau

        self.num_heads_max = num_heads
        self.num_global_heads = num_global_heads
        self.head_dim = 64

        self.global_dim = self.num_global_heads * self.head_dim  # K*64

        self.attn_drop = nn.Dropout(attn_drop)

        # Global heads용 QKV
        self.qkv_global = QKVLinear(dim, self.global_dim * 3, bias=qkv_bias)

        # Mixing weights (H, K)
        self.mix_logits = nn.Parameter(0.02 * torch.randn(num_heads, num_global_heads))

        # Local value projection (폭 p 유지)
        self.v_local = Linear(dim, dim, bias=qkv_bias)

        # Output projection
        self.proj = Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, p):
        """
        x: (B, N, dim_max)
        p: HydraViT subnetwork width
        """
        B, N, _ = x.shape
        if p % self.head_dim != 0:
            raise ValueError(f"p({p}) must be divisible by head_dim({self.head_dim}).")

        H_eff = max(1, p // self.head_dim)
        K = self.num_global_heads
        K_eff = min(K, H_eff)  # 작은 subnet에서 낭비 제거(권장)

        # -------------------------
        # 1) Global Q,K (K_eff heads)
        # -------------------------
        # qkv_g: (B, N, 3, K, 64)
        qkv_g = self.qkv_global(x, p_in=p, p_out=self.global_dim).reshape(B, N, 3, K, self.head_dim)
        qg, kg, _vg = qkv_g.unbind(2)  # (B, N, K, 64) / vg는 현재 사용하지 않음

        qg = qg[:, :, :K_eff, :]
        kg = kg[:, :, :K_eff, :]

        # (B, K_eff, N, 64)
        qg = qg.transpose(1, 2)
        kg = kg.transpose(1, 2)

        # scale
        qg = qg * (self.head_dim ** -0.5)

        # global_logits: (B, K_eff, N, N)
        global_logits = torch.matmul(qg, kg.transpose(-2, -1))

        # -------------------------
        # 2) Mixing (H_eff x K_eff)
        # -------------------------
        mix_logits = self.mix_logits[:H_eff, :K_eff]  # (H_eff, K_eff)

        if self.training and self.stochastic:
            # Gumbel-Softmax(soft) : logits에 gumbel noise 추가 후 softmax
            u = torch.rand_like(mix_logits).clamp_(1e-6, 1.0 - 1e-6)
            g = -torch.log(-torch.log(u))
            mix = F.softmax((mix_logits + g) / self.tau, dim=-1)
        else:
            mix = F.softmax(mix_logits, dim=-1)

        # mix: (H_eff, K_eff) -> (B, H_eff, K_eff)로 broadcast
        mix_b = mix.unsqueeze(0).expand(B, -1, -1).to(global_logits.dtype)

        # -------------------------
        # 3) local_logits 생성: einsum -> bmm
        # -------------------------
        # global_logits: (B, K_eff, N, N) -> (B, K_eff, N*N)
        gl = global_logits.reshape(B, K_eff, N * N)

        # (B, H_eff, K_eff) @ (B, K_eff, N*N) -> (B, H_eff, N*N)
        local_logits = torch.bmm(mix_b, gl).reshape(B, H_eff, N, N)

        # -------------------------
        # 4) softmax (안정화: softmax만 fp32로 계산)
        # -------------------------
        # fp16/bf16 텐서에 대해 softmax만 fp32로 수행 후 dtype 복귀
        attn = F.softmax(local_logits, dim=-1, dtype=torch.float32).to(local_logits.dtype)
        attn = self.attn_drop(attn)

        # -------------------------
        # 5) local V 및 attn@V
        # -------------------------
        # v: (B, N, p) -> (B, N, H_eff, 64) -> (B, H_eff, N, 64)
        v = self.v_local(x, p_in=p, p_out=p).reshape(B, N, H_eff, self.head_dim).transpose(1, 2)

        # out: (B, H_eff, N, 64) -> (B, N, p)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, p)

        # -------------------------
        # 6) output projection
        # -------------------------
        out = self.proj(out, p_in=p, p_out=p)
        out = self.proj_drop(out)
        return out

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hydravit_layers import QKVLinear, Linear

class FiSHAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,              # "최대" 헤드 수
        num_global_heads: int = 4,       # K
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        stochastic: bool = True,
        hard: bool = False,              # HardFiSH 모드
        tau: float = 1.0,                # gumbel
    ):
        super().__init__()
        self.stochastic = stochastic
        self.hard = hard
        self.tau = tau
        self.num_heads_max = num_heads
        self.num_global_heads = num_global_heads
        self.head_dim = 64
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        
        #  Global heads용 별도 QKV (K개만)
        global_dim = num_global_heads * self.head_dim
        self.qkv_global = QKVLinear(dim, global_dim * 3, bias=qkv_bias)
        
        #  Mixing weights: (H, K)
        self.mix_logits = nn.Parameter(torch.zeros(num_heads, num_global_heads))
        
        #  Value projection은 H개 모두 필요 (output 조합용)
        self.v_local = Linear(dim, dim, bias=qkv_bias)
        
    def forward(self, x, p):
        B, N, _ = x.shape
        H_eff = max(1, p // 64)  # 실제 사용할 head 수
        K = self.num_global_heads
        
        # K개 global heads만 계산 (효율성!)
        qkv_g = self.qkv_global(x[:, :, :p]).reshape(B, N, 3, K, self.head_dim)
        qg, kg, vg = qkv_g.unbind(2)  # (B, N, K, 64)
        
        # K개 global attention logits
        qg = qg.transpose(1, 2) * (self.head_dim ** -0.5)  # (B, K, N, 64)
        kg = kg.transpose(1, 2)
        global_logits = qg @ kg.transpose(-2, -1)  # (B, K, N, N)
        
        # H_eff개 local heads를 mixture로 생성
        mix = self.mix_logits[:H_eff, :K].softmax(dim=-1)  # (H_eff, K)
        
        if self.training and self.stochastic:
            g = -torch.log(-torch.log(torch.rand_like(mix).clamp_min(1e-6)).clamp_min(1e-6))
            mix = ((self.mix_logits[:H_eff, :K] + g) / self.tau).softmax(dim=-1)
        
        # Local attention = weighted sum of globals
        local_logits = torch.einsum("hk,bkij->bhij", mix, global_logits)
        attn = local_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Value는 H_eff개 필요
        v = self.v_local(x[:, :, :p]).reshape(B, N, H_eff, self.head_dim)
        v = v.transpose(1, 2)  # (B, H_eff, N, 64)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, p)
        out = self.proj(out, p)
        return out
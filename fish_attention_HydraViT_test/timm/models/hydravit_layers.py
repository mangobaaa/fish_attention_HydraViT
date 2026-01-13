import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


class QKVLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, p=float) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, p_in=None, p_out=None) -> Tensor:
        if p_in is None:
          return F.linear(input, self.weight, self.bias)
        if p_out is None:
            p_out = p_in  # 기존 동작과 호환
        # out_features = 3 * (각 블록 폭)
        block = int(self.out_features / 3)  # = global_dim 또는 dim

        # 안전장치 (필수)
        if p_out > block:
            raise ValueError(f"p_out({p_out}) > block({block}). QKVLinear slicing out of range.")
        if p_in > self.in_features:
            raise ValueError(f"p_in({p_in}) > in_features({self.in_features}).")

        l1 = list(range(0, p_out))
        l2 = list(range(block, block + p_out))
        l3 = list(range(2 * block, 2 * block + p_out))

        return F.linear(
            input[:, :, 0:p_in],
            self.weight[l1 + l2 + l3, 0:p_in],
            self.bias[l1 + l2 + l3] if self.bias is not None else None
        )

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, p=float) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, p_in=None, p_out=-1) -> Tensor:
        if p_out== -1:
            p_out = p_in
        if p_in is None:
          return F.linear(input, self.weight, self.bias)
        else:
            if len(input.shape) == 2:
                return F.linear(input[:,0:p_in],
                         self.weight[0:p_out,0:p_in],
                         self.bias[0:p_out])
            if len(input.shape) == 3:
                return F.linear(input[:,:,0:p_in],
                         self.weight[0:p_out,0:p_in],
                         self.bias[0:p_out])
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
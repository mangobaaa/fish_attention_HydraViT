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

    def forward(self, input: Tensor, p=None) -> Tensor:
        if p is None:
          return F.linear(input, self.weight, self.bias)
        else:
            self.out_features
            l1 = [i for i in range(p)]
            l2 = [i for i in range(int(self.out_features/3), int(self.out_features/3) + p)]
            l3 = [i for i in range(2*int(self.out_features/3), 2*int(self.out_features/3) + p)]
            return F.linear(input[:,:,0:p],
                     self.weight[l1+l2+l3, 0:p],
                     self.bias[l1+l2+l3])
        
        return F.linear(input, self.weight, self.bias)

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
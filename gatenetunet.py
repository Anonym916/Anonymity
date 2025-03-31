import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.jit import Final
from module.activation import _gumbel_sigmoid


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class controller(nn.Module):
    def __init__(
            self,
            n_channels: int,
            thred: float = 0.5,
            pool: Literal['avg', 'max'] = 'avg',
            bias: bool = True,
            training: bool = True,
    ):
        super().__init__()
        self.thred = thred
        self.training = training

        if pool == 'avg':
            self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            self.pool_layer = nn.AdaptiveMaxPool2d((1, 1))

        # self.unit = nn.Sequential(
        #     nn.LayerNorm(n_channels),
        #     nn.Linear(n_channels, 1, bias=bias),
        #     # nn.ReLU(),
        #     nn.BatchNorm1d(1),
        # )

        # self._unit = nn.Sequential(
        #     # nn.LayerNorm(n_channels),
        #     nn.Linear(n_channels, 16, bias=bias),
        #     nn.ReLU(),
        #     nn.Linear(16, 1, bias=bias)
        # )

        self._unit = nn.Sequential(
            # nn.LayerNorm(n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, 1, bias=bias),
            # nn.BatchNorm1d(1)
        )

        self._gumbel_sigmoid = _gumbel_sigmoid

    def forward(self, x: torch.Tensor):
        # transfer spatial feature to linear
        x = self.pool_layer(x)
        x = x.view(x.size(0), -1)

        # compute the mask
        x = self._unit(x)
        mask = self._gumbel_sigmoid(
            logits=x,
            tau=5,
            hard=True,
            threshold=self.thred,
            training=self.training,
        )
        # print(mask)
        return mask

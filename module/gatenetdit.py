import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
from module.activation import _gumbel_sigmoid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class controller(nn.Module):
    def __init__(
            self,
            thred: float = 0.5,
            token_dim: int = 1024,  # 4096
            embed_dim: int = 1536,
            training: bool = True,
    ):
        super().__init__()
        self.thred = thred

        self.ln = nn.LayerNorm(embed_dim)
        # self.mlp_local = nn.Sequential(
        #     nn.Linear(token_dim, 16),
        #     nn.Linear(16, embed_dim),
        #     nn.LayerNorm(embed_dim)
        # )
        # self.mlp_mix = nn.Linear(embed_dim, 1)

        self._mlp_local = nn.Sequential(
            nn.ReLU(),
            nn.Linear(token_dim, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, embed_dim // 24, bias=True),  # 1536/24=64
        )
        self._mlp_global = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, embed_dim // 24, bias=True)  # 1536/24=64
        )
        self._mlp_mix = nn.Sequential(
            # nn.LayerNorm(embed_dim//24),
            nn.ReLU(),
            nn.Linear(embed_dim // 24, 1, bias=True),
            # nn.BatchNorm1d(1)
        )

        # self.bn = nn.BatchNorm1d(1)

        self.training = training
        self._gumbel_sigmoid = _gumbel_sigmoid

    def forward(self, x: torch.Tensor):
        x = self.ln(x)

        # compute local features
        x_local = x.mean(dim=2)
        x_local = self._mlp_local(x_local)

        # compute global features
        x_global = x.mean(dim=1)
        x_global = self._mlp_global(x_global)

        # fuse the local and global features
        x_mix = x_local + x_global
        # x_mix = 0.5*x_local + 0.5*x_global

        # compute the mask
        logits = self._mlp_mix(x_mix)
        # logits = self.bn(logits)
        mask = _gumbel_sigmoid(
            logits=logits,
            tau=5,
            hard=True,
            threshold=self.thred,
            training=self.training
        )

        return mask

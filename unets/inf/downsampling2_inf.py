# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from src.utils import mul
from diffusers.utils import deprecate, is_torch_version, logging
from diffusers.utils.torch_utils import apply_freeu
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor, AttnAddedKVProcessor2_0
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.resnet import (
    Downsample2D,
    FirDownsample2D,
    FirUpsample2D,
    KDownsample2D,
    KUpsample2D,
    ResnetBlock2D,
    ResnetBlockCondNorm2D,
    Upsample2D,
)
from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )

    raise ValueError(f"{down_block_type} does not exist.")


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            ) # use_conv=False
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditional_ctrl: torch.Tensor = None,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        additional_residuals: Optional[torch.Tensor] = None,
        gate_down_block: nn.ModuleList = None,
        skiplayer_down_block: nn.ModuleList = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        output_states = ()
        blocks = list(zip(self.resnets, self.attentions))
        if gate_down_block is not None:
            mask_states = ()

        for i, (resnet, attn) in enumerate(blocks):
            if gate_down_block is not None:
                mask = gate_down_block[i](hidden_states)
                if mask[0] == 0:
                    hidden_states = skiplayer_down_block[i](hidden_states)
                else:
                    if self.training and self.gradient_checkpointing:
                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)
                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet),
                            hidden_states,
                            temb,
                            **ckpt_kwargs,
                        )
                        hidden_states = attn(
                            hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    else:
                        hidden_states = resnet(hidden_states, temb)
                        hidden_states = attn(
                            hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]

                    # apply additional residuals to the output of the last pair of resnet and attention blocks
                    if i == len(blocks) - 1 and additional_residuals is not None:
                        hidden_states = hidden_states + additional_residuals

                    # inject conditional control
                    if conditional_ctrl is not None:
                        hidden_states = hidden_states + conditional_ctrl.pop(0)

            output_states = output_states + (hidden_states,)
            if gate_down_block is not None:
                mask_states = mask_states + (mask,)
                # hidden_states = mul(hidden_states, mask) + mul(identity, (1 - mask))

        if self.downsamplers is not None:
            if gate_down_block is not None:
                mask = gate_down_block[-1](hidden_states)
                if mask[0] == 0:
                    hidden_states = skiplayer_down_block[-1](hidden_states)
                else:
                    for downsampler in self.downsamplers:
                        hidden_states = downsampler(hidden_states)

                    # inject conditional control
                    if conditional_ctrl is not None:
                        hidden_states = hidden_states + conditional_ctrl.pop(0)

            output_states = output_states + (hidden_states,)
            if gate_down_block is not None:
                mask_states = mask_states + (mask,)
                # hidden_states = mul(hidden_states, mask) + mul(identity, (1 - mask))

        if gate_down_block is not None:
            return hidden_states, output_states, mask_states
        else:
            return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            ) # use_conv=False
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditional_ctrl: torch.Tensor = None,
        temb: Optional[torch.Tensor] = None,
        gate_down_block: nn.ModuleList = None,
        skiplayer_down_block: nn.ModuleList = None,
        *args, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        output_states = ()
        if gate_down_block is not None:
            mask_states = ()
        for i, resnet in enumerate(self.resnets):
            if gate_down_block is not None:
                mask = gate_down_block[i](hidden_states)
                if mask[0] == 0:
                    hidden_states = skiplayer_down_block[i](hidden_states)
                else:
                    if self.training and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                        if is_torch_version(">=", "1.11.0"):
                            hidden_states = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                            )
                        else:
                            hidden_states = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(resnet), hidden_states, temb
                            )
                    else:
                        hidden_states = resnet(hidden_states, temb)

                    if conditional_ctrl is not None:
                        hidden_states = hidden_states + conditional_ctrl.pop(0)

            output_states = output_states + (hidden_states,)
            if gate_down_block is not None:
                mask_states = mask_states + (mask,)
                # hidden_states = mul(hidden_states, mask) + mul(identity, (1 - mask))

        if self.downsamplers is not None:
            if gate_down_block is not None:
                mask = gate_down_block[-1](hidden_states)
                if mask[0] == 0:
                    hidden_states = skiplayer_down_block[-1](hidden_states)
                else:
                    for downsampler in self.downsamplers:
                        hidden_states = downsampler(hidden_states)

                    if conditional_ctrl is not None:
                        hidden_states = hidden_states + conditional_ctrl.pop(0)

            output_states = output_states + (hidden_states,)
            if gate_down_block is not None:
                mask_states = mask_states + (mask,)
                # hidden_states = mul(hidden_states, mask) + mul(identity, (1 - mask))

        if gate_down_block is not None:
            return hidden_states, output_states, mask_states
        else:
            return hidden_states, output_states

# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
# transformers.models.phi.modeling_phi
import torch
from importlib import import_module

import torch.nn as nn
from torch import Tensor
from transformers.models.phi.configuration_phi import PhiConfig
import math
import einops
from modules.phi_mlp import PhiMLP


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs
    

class Block(nn.Module):
    def __init__(self, d_model, config, factory_kwargs, layer_idx, **kwargs):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.layer_idx = layer_idx

        # Mixer
        MixerClass = import_module(config.CoreType).Mixer
        self.mixer = MixerClass(
            d_model=self.d_model,
            layer_idx=layer_idx,
            **kwargs,
            **config.core_input,
            **factory_kwargs,
        )

        # MLP + LayerNorm + Dropout
        self.mlp = PhiMLP(
            PhiConfig(
                hidden_size=self.d_model,
                intermediate_size=self.d_model * 4,
                hidden_act="relu",
            )
        )
        self.input_layernorm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.output_layernorm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.resid_dropout = nn.Dropout(config.block_input.resid_dropout)

        return

    def forward(
        self,
        hidden_states: Tensor,
        inference_params=None,
        run_mlp_component=True,
        return_mixer_matrix=False,
        return_mamba_outputs=False,
        position_ids=None,
        **kwargs,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        outputs = {}

        residual = hidden_states
        # B, L, D  = hidden_states.shape
        # print(hidden_states.shape)
        B, L, D  = hidden_states.shape
        # print(L)
        # print(math.sqrt(L))
        assert (math.sqrt(L) * math.sqrt(L) == L)   # make sure it's a square
        hidden_states = einops.rearrange(hidden_states, "b (h w) d -> b d h w", h=int(math.sqrt(L)), w=int(math.sqrt(L))) # change the shape to align with CV tasks
        # apply cross scan to the sequence
        hidden_states = CrossScan.apply(hidden_states)
        hidden_states = einops.rearrange(hidden_states, "b c d l -> b (c l) d") 
        hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer
        mamba_outputs = self.mixer(
            hidden_states,
            return_mixer_matrix=return_mixer_matrix,
            inference_params=inference_params,
            position_ids=position_ids,
        )
        # apply cross merge to the output of mamba module
        mamba_outputs["hidden_states"] = einops.rearrange(mamba_outputs["hidden_states"], "b (c l) d -> b c d l", l=L)
        mamba_outputs["hidden_states"] = einops.rearrange(mamba_outputs["hidden_states"], "b c d (h w) -> b c d h w", h=int(math.sqrt(L)), w=int(math.sqrt(L)))
        mamba_outputs["hidden_states"] = CrossMerge.apply(mamba_outputs["hidden_states"])
        # print(mamba_outputs["hidden_states"].shape)
        mamba_outputs["hidden_states"] = einops.rearrange(mamba_outputs["hidden_states"], "b d l -> b l d")
        # add the class token
        mamba_outputs["hidden_states"] = mamba_outputs["hidden_states"].to(
            residual.dtype
        )

        if not run_mlp_component:
            return mamba_outputs

            # store outputs
        if return_mamba_outputs:
            outputs["mamba_hidden_states"] = mamba_outputs["hidden_states"]
        if return_mixer_matrix:
            outputs["transfer_matrix"] = mamba_outputs["transfer_matrix"]

        # Feed Forward
        feed_forward_hidden_states = self.resid_dropout(self.mlp(residual)).to(
            residual.dtype
        )

        # Mixer output
        mixer_output = self.resid_dropout(mamba_outputs["hidden_states"])
        residual = mixer_output
        # outputs["hidden_states"] = self.mlp(mixer_output) + residual
        # sum all up (this is not sequential)
        # print(mixer_output.shape, feed_forward_hidden_states.shape, residual.shape)
        outputs["hidden_states"] = self.output_layernorm(mixer_output + feed_forward_hidden_states + residual)
        # outputs["hidden_states"] = self.resid_dropout(self.mlp(mixer_output)).to(
        #     residual.dtype
        # ) + residual

        return outputs

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if getattr(self.mixer, "allocate_inference_cache", None) is None:
            return
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

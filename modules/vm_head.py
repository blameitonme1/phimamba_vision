# Copyright (c) 2024, Kevin Li, Aviv Bick.

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, OrderedDict
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from mamba_ssm.utils.generation import GenerationMixin
from transformers.utils import ModelOutput
import einops
import torch.nn.init as init
from modules.backbone import MixerModel
from utils.config import Config


class Permute(nn.Module):
    def __init__(self, pattern, **axes_lengths):
        super(Permute, self).__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x):
        return einops.rearrange(x, self.pattern, **self.axes_lengths)
    
@dataclass
class CustomMambaVMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_transfer_matrices: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_mamba_outputs: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class VMHeadModel(nn.Module, GenerationMixin, PyTorchModelHubMixin):
    def __init__(
        self, config: dict, initializer_cfg=None, device=None, dtype=None, **kwargs
    ) -> None:

        super().__init__()
        
        # Load config
        if not isinstance(config, Config):
            config = Config.from_dict(config)
        self.config = config

        # Factory kwargs
        factory_kwargs = {"device": device, "dtype": dtype}

        # Mixer model
        self.backbone = MixerModel(
            input_size=self.config.VisionModel.input.res_size, # resolution of the input image
            config=self.config,
            initializer_cfg=initializer_cfg,
            **factory_kwargs,
            **kwargs
        )

        # VM head for classification
        d_model = self.config.MixerModel.input.d_model
        print(f"class size : {self.config.VisionModel.output.class_size}")
        # self.vm_head = nn.Linear(
        #     d_model, self.config.VisionModel.output.class_size, bias=True, **factory_kwargs
        # )  # changed for Phi
        # print(f"d_model: {d_model}")
        self.vm_head = nn.Sequential(OrderedDict(
            # norm=nn.LayerNorm(d_model),  # B,L,D
            permute=Permute("b (h w) d -> b d h w", h=int(self.config.VisionModel.input.res_size // self.config.VisionModel.patch_size)),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(d_model, self.config.VisionModel.output.class_size),
        ))

        for name, param in self.vm_head.named_parameters():
            if 'weight' in name:
                if isinstance(param, nn.Linear):
                    init.xavier_uniform_(param)
                elif isinstance(param, nn.Conv2d):
                    init.kaiming_uniform_(param, mode='fan_in', nonlinearity='silu')
            elif 'bias' in name:
                init.constant_(param, 0)
        # nn.init.zeros_(self.vm_head.bias)
        return

    def allocate_inference_cache(self, *args, **kwargs):
        return self.backbone.allocate_inference_cache(*args, **kwargs)

    def forward(
        self,
        input_ids,
        position_ids=None,
        return_mixer_matrix=False,
        return_mamba_outputs=False,
        return_hidden_states=False,
        return_logits=True,
        inference_params=None,
        num_last_tokens=0,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        outputs = self.backbone(
            input_ids,
            return_mixer_matrix=return_mixer_matrix,
            return_mamba_outputs=return_mamba_outputs,
            return_hidden_states=return_hidden_states,
            inference_params=inference_params,
            position_ids=position_ids,
        )

        if outputs["last_hidden_state"] is not None and return_logits:
            logits = self.vm_head(outputs["last_hidden_state"]) # only use the class token
            outputs["logits"] = (
                logits if num_last_tokens == 0 else logits[:, -num_last_tokens:]
            )
        else:
            outputs["logits"] = None

        return CustomMambaVMOutput(
            loss=None,
            logits=outputs["logits"],
            all_hidden_states=outputs["all_hidden_states"],
            all_transfer_matrices=outputs["all_transfer_matrices"],
            all_mamba_outputs=outputs["all_mamba_outputs"],
            last_hidden_state=outputs["last_hidden_state"],
        )

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaVMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f)

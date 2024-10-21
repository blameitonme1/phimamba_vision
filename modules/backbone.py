from importlib import import_module
import torch
import torch.nn as nn
import collections
import math
from transformers.models.vit.configuration_vit import ViTConfig
from typing import Dict, List, Optional

class Embeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.VisionModel.hidden_size)) if use_mask_token else None
        self.patch_embeddings = PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, config.VisionModel.hidden_size))
        self.dropout = nn.Dropout(config.VisionModel.hidden_dropout_prob)
        self.patch_size = config.VisionModel.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask


        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.VisionModel.image_size, config.VisionModel.patch_size
        num_channels, hidden_size = config.VisionModel.num_channels, config.VisionModel.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

class MixerModel(nn.Module):
    def __init__(
        self, input_size, config=None, device=None, dtype=None, **kwargs
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        n_layer = self.config.MixerModel.input.n_layer
        d_model = self.config.MixerModel.input.d_model

        # self.embedding = nn.Embedding(input_size, d_model, **factory_kwargs)
        self.embedding = Embeddings(config)

        blocks = [
            self.config.__dict__[name]
            for name in self.config.__dict__.keys()
            if name.startswith("Block")
        ]
        self.layers = nn.ModuleList()
        for block_cfg in blocks:
            n_layers = block_cfg.n_layers
            Block = import_module(block_cfg.BlockType).Block
            layers = nn.ModuleList(
                [
                    Block(
                        d_model=d_model,
                        config=block_cfg,
                        factory_kwargs=factory_kwargs,
                        layer_idx=i,
                        **kwargs,
                    ).to(device)
                    for i in range(len(self.layers), len(self.layers) + n_layers)
                ]
            )
            self.layers += layers
        assert len(self.layers) == n_layer

        # Initialize norm:
        norm_epsilon: float = 1e-5
        norm_cls = self.config.MixerModel.input.lm_head_prenorm
        if norm_cls == "layer":
            self.final_layernorm = nn.LayerNorm(d_model, eps=norm_epsilon).to(device)
        else:
            raise Exception(f"Norm class {norm_cls} is not valid.")

        return

    def allocate_inference_cache(self, *args, **kwargs):
        return {
            i: layer.allocate_inference_cache(*args, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids,
        return_mixer_matrix=False,
        return_mamba_outputs=False,
        return_hidden_states=False,
        inference_params=None,
        position_ids=None,
    ):

        # Start running the layers
        hidden_states = self.embedding(input_ids)

        # Initialize outputs
        outputs = {
            "last_hidden_state": None,
            "all_hidden_states": (hidden_states,) if return_hidden_states else (),
            "all_transfer_matrices": tuple(),
            "all_mamba_outputs": tuple(),
        }

        # Run the layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                return_mixer_matrix=return_mixer_matrix,
                return_mamba_outputs=return_mamba_outputs,
                inference_params=inference_params,
                position_ids=position_ids,
            )
            # Record outputs
            hidden_states = layer_outputs["hidden_states"]
            if return_hidden_states:
                outputs["all_hidden_states"] += (hidden_states,)
            if return_mamba_outputs:
                outputs["all_mamba_outputs"] += (layer_outputs["mamba_hidden_states"],)
            if return_mixer_matrix:
                outputs["all_transfer_matrices"] += (layer_outputs["transfer_matrix"],)

        # Last layer, apply layer norm
        outputs["last_hidden_state"] = self.final_layernorm(hidden_states)
        return outputs

from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from .adduct_embedding import AdductEmbeddingLayer
from .collision_energy_feature import CollisionEnergyFeatureLayer

from ....common.torch_utils.nn_utils import build_fc_layers
from .....libs.mmkit.mmkit import Adduct


class MS2ConditionEncoder(nn.Module):
    """
    Measurement-condition encoder for MS2 spectrum generation.

    This module encodes MS/MS experimental conditions such as:

    - precursor adduct type
    - collision energy

    Components
    ----------
    - AdductEmbeddingLayer
    - CollisionEnergyFeatureLayer

    Inputs
    ------
    adduct_idx:
        LongTensor with shape [batch_size].

    collision_energy:
        Tensor with shape [batch_size] or [batch_size, 1].

    Output
    ------
    Tensor with shape [batch_size, feature_dim].
    """

    def __init__(
        self,
        *,
        # --- AdductEmbeddingLayer args ---
        adduct_type_strs: Tuple[str, ...],
        adduct_embedding_dim: int,

        # --- CollisionEnergyFeatureLayer args ---
        ce_feature_dim: int,
        ce_fc_dims: Tuple[int, ...] = (),

        # --- condition encoder args ---
        feature_dim: int,
        fc_dims: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()

        self.adduct_layer = AdductEmbeddingLayer(
            adduct_type_strs=adduct_type_strs,
            embedding_dim=adduct_embedding_dim,
        )

        self.ce_layer = CollisionEnergyFeatureLayer(
            feature_dim=ce_feature_dim,
            fc_dims=ce_fc_dims,
        )

        in_dim = self.adduct_layer.feature_dim + self.ce_layer.feature_dim
        self._feature_dim = int(feature_dim)

        self.fnet = build_fc_layers(
            input_dim=in_dim,
            fc_dims=fc_dims,
            output_dim=self._feature_dim,
            dropout=0.0,
        )

        self.norm = nn.LayerNorm(self._feature_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize only the fusion network defined in this module."""
        for module in self.fnet.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def adducts_to_idx(
        self,
        adducts: Union[Tuple[Adduct, ...], List[Adduct], List[str], Tuple[str, ...]],
        device: torch.device | None = None,
    ) -> torch.LongTensor:
        """
        Convert adduct objects or adduct strings to integer indices.

        Parameters
        ----------
        adducts:
            Adduct objects or adduct strings.

        device:
            Device for the returned tensor.

        Returns
        -------
        torch.LongTensor
            Tensor with shape [batch_size].
        """
        return self.adduct_layer.adducts_to_idx(
            adducts=adducts,
            device=device,
        )

    def ce_to_tensor(
        self,
        ce: torch.Tensor | list[float],
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Convert collision energy values to a tensor.

        Parameters
        ----------
        ce:
            Collision energy values.

        device:
            Device for the returned tensor.

        dtype:
            Data type for the returned tensor.

        Returns
        -------
        torch.Tensor
            Tensor with shape [batch_size].
        """
        return self.ce_layer.ce_to_tensor(
            ce=ce,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        adduct_idx: torch.LongTensor,
        collision_energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode MS2 measurement conditions.

        Parameters
        ----------
        adduct_idx:
            LongTensor with shape [batch_size].

        collision_energy:
            Tensor with shape [batch_size] or [batch_size, 1].

        Returns
        -------
        torch.Tensor
            Condition embedding with shape [batch_size, feature_dim].
        """
        z_adduct = self.adduct_layer(adduct_idx)
        z_ce = self.ce_layer(collision_energy)

        z = torch.cat([z_adduct, z_ce], dim=-1)

        out = self.fnet(z)
        out = self.norm(out)

        return out

    @property
    def in_feature_dim(self) -> int:
        return self.adduct_layer.feature_dim + self.ce_layer.feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def adduct_type_strs(self) -> Tuple[str, ...]:
        return self.adduct_layer.adduct_type_strs

    @property
    def adduct_types(self) -> Tuple[Adduct, ...]:
        return self.adduct_layer.adduct_types
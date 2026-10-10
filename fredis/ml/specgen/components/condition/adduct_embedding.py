from typing import Tuple, Dict, List, Union
import torch
import torch.nn as nn

from .....libs.mmkit.mmkit import Adduct

class AdductEmbeddingLayer(nn.Module):
    def __init__(
        self,
        adduct_type_strs: Tuple[str, ...],
        embedding_dim: int,
    ):
        super().__init__()

        # --- normalize & register adduct types ---
        self._adduct_type_strs = tuple(adduct_type_strs)
        self._adduct_types: Tuple[Adduct] = tuple(
            Adduct.parse(ad) for ad in self._adduct_type_strs
        )

        self._adduct2idx: Dict[str, int] = {
            adduct_str: i for i, adduct_str in enumerate(self._adduct_type_strs)
        }

        # --- embedding ---
        self.embedding = nn.Embedding(
            num_embeddings=len(self._adduct_type_strs),
            embedding_dim=embedding_dim,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    # ---------- ① adducts -> idx ----------
    def adducts_to_idx(
        self,
        adducts: Union[Tuple["Adduct"], List["Adduct"], List[str]],
        device: torch.device | None = None,
    ) -> torch.LongTensor:
        """
        Convert adducts (Adduct or str) to index tensor.

        Returns
        -------
        LongTensor
            shape: [B]
        """
        if isinstance(adducts[0], str):
            adduct_strs = adducts
        else:
            adduct_strs = [str(ad) for ad in adducts]

        idx = torch.tensor(
            [self._adduct2idx.get(a, -1) for a in adduct_strs],
            dtype=torch.long,
            device=device,
        )
        return idx

    # ---------- ② idx -> embedding ----------
    def forward(self, adduct_idx: torch.LongTensor):
        """
        Parameters
        ----------
        adduct_idx : LongTensor
            shape: [B]

        Returns
        -------
        Tensor
            shape: [B, embedding_dim]
        """
        return self.embedding(adduct_idx)
    
    @property
    def feature_dim(self) -> int:
        return self.embedding.embedding_dim
    
    @property
    def adduct_types(self) -> Tuple["Adduct"]:
        return self._adduct_types
    
    @property
    def adduct_type_strs(self) -> Tuple[str, ...]:
        return self._adduct_type_strs
    
    @property
    def num_adduct_types(self) -> int:
        return len(self._adduct_type_strs)
    
    @property
    def adduct2idx(self) -> Dict[str, int]:
        return self._adduct2idx
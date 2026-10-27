import math
from typing import Tuple
import torch
import torch.nn as nn

class CollisionEnergyFeatureLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        fc_dims: Tuple[int, ...] = (32,),
        *,
        center: float = 0.0,
        scale: float = 100.0,
    ):
        super().__init__()
        self._feature_dim = int(feature_dim)
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                "CE center must be finite and scale must be finite and positive"
            )
        self.register_buffer("ce_center", torch.tensor(float(center)))
        self.register_buffer("ce_scale", torch.tensor(float(scale)))
        dims = (1, *fc_dims, self._feature_dim)
        if any(type(width) is not int or width < 1 for width in dims):
            raise ValueError("CE MLP widths must be positive integers")
        layers = []
        for index, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.proj = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def ce_to_tensor(
        ce: torch.Tensor | list[float],
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Convert CE to float tensor.

        Returns
        -------
        Tensor
            shape: [B]
        """
        if isinstance(ce, torch.Tensor):
            x = ce.to(device=device, dtype=dtype)
        else:
            x = torch.tensor(ce, device=device, dtype=dtype)

        if x.dim() == 0:
            x = x.view(1)
        return x

    def forward(self, ce: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ce : Tensor
            shape: [B] or [B, 1]

        Returns
        -------
        Tensor
            shape: [B, feature_dim]
        """
        if ce.dim() == 1:
            ce = ce.unsqueeze(-1)  # [B, 1]
        elif ce.dim() != 2 or ce.size(-1) != 1:
            raise ValueError("ce must be [B] or [B,1]")

        normalized = (ce - self.ce_center) / self.ce_scale.clamp_min(1e-6)
        encoded = normalized
        out = self.proj(encoded)
        return out

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

from typing import Tuple
import torch
import torch.nn as nn

from ....common.torch_utils.nn_utils import build_fc_layers

class CollisionEnergyFeatureLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        fc_dims: Tuple[int, ...] = (),
    ):
        super().__init__()
        self._feature_dim = int(feature_dim)
        if self._feature_dim != 16:
            raise ValueError(
                "CollisionEnergyFeatureLayer uses eight sin/cos pairs and "
                "therefore requires feature_dim=16."
            )
        self.register_buffer(
            "frequencies",
            torch.exp(
                -torch.log(torch.tensor(10000.0))
                * torch.arange(8, dtype=torch.float32)
                / 8.0
            ),
        )
        self.register_buffer("ce_center", torch.tensor(0.0))
        self.register_buffer("ce_scale", torch.tensor(1.0))
        self.proj = build_fc_layers(
            input_dim=16,
            fc_dims=fc_dims,
            output_dim=self._feature_dim,
            dropout=0.0,
        )

        self.norm = nn.LayerNorm(self._feature_dim)
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
        angles = normalized * self.frequencies.to(ce.dtype).view(1, -1)
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        out = self.proj(encoded)
        out = self.norm(out)
        return out
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ...libs.mmkit.mmkit import Adduct, Formula


@dataclass(frozen=True)
class FormulaTensorizer:
    """Convert mmkit Formula/Adduct objects to fixed-order tensors.

    The element order is supplied by the caller. In the specgen path this is
    derived from MolEncoder.symbols plus elements introduced by adduct rules.
    """

    element_order: tuple[str, ...]
    include_charge: bool = True

    def __post_init__(self) -> None:
        if len(self.element_order) == 0:
            raise ValueError("element_order must not be empty.")

        deduped = tuple(dict.fromkeys(str(element) for element in self.element_order))
        if deduped != self.element_order:
            object.__setattr__(self, "element_order", deduped)

    @property
    def dim(self) -> int:
        return len(self.element_order) + int(self.include_charge)

    def formula_to_tensor(
        self,
        formula: Formula,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        missing = [
            element
            for element, count in formula.elements.items()
            if count != 0 and element not in self.element_order
        ]
        if missing:
            raise ValueError(
                "Formula contains elements that are not in element_order: "
                f"{missing}. element_order={self.element_order}."
            )

        values = [
            int(formula.elements.get(element, 0))
            for element in self.element_order
        ]

        if self.include_charge:
            values.append(int(formula.charge))

        return torch.tensor(values, dtype=dtype, device=device)

    def formulas_to_tensor(
        self,
        formulas: Sequence[Formula],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if len(formulas) == 0:
            return torch.empty((0, self.dim), dtype=dtype, device=device)

        return torch.stack(
            [
                self.formula_to_tensor(
                    formula,
                    dtype=dtype,
                    device=device,
                )
                for formula in formulas
            ],
            dim=0,
        )

    def tensor_to_formula(self, tensor: Tensor) -> Formula:
        if tensor.dim() != 1:
            raise ValueError(
                "tensor_to_formula expects a 1D tensor, "
                f"got shape {tuple(tensor.shape)}."
            )

        if tensor.numel() != self.dim:
            raise ValueError(
                f"Formula tensor has width {tensor.numel()}, "
                f"expected {self.dim}."
            )

        values = [
            int(round(float(value)))
            for value in tensor.detach().cpu().tolist()
        ]
        elements = {
            element: values[index]
            for index, element in enumerate(self.element_order)
            if values[index] != 0
        }
        charge = values[-1] if self.include_charge else 0

        return Formula.from_dict(elements, charge=charge).normalized

    def tensors_to_formulas(self, tensor: Tensor) -> list[Formula]:
        if tensor.dim() != 2:
            raise ValueError(
                "tensors_to_formulas expects a 2D tensor, "
                f"got shape {tuple(tensor.shape)}."
            )

        return [self.tensor_to_formula(row) for row in tensor]

    def adduct_to_delta_tensor(
        self,
        adduct: Adduct,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        missing = [
            element
            for element, count in adduct.element_diff.items()
            if count != 0 and element not in self.element_order
        ]
        if missing:
            raise ValueError(
                "Adduct contains elements that are not in element_order: "
                f"{missing}. element_order={self.element_order}."
            )

        values = [
            int(adduct.element_diff.get(element, 0))
            for element in self.element_order
        ]

        if self.include_charge:
            values.append(int(adduct.charge))

        return torch.tensor(values, dtype=dtype, device=device)

    def adducts_to_delta_tensor(
        self,
        adducts: Sequence[Adduct],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if len(adducts) == 0:
            return torch.empty((0, self.dim), dtype=dtype, device=device)

        return torch.stack(
            [
                self.adduct_to_delta_tensor(
                    adduct,
                    dtype=dtype,
                    device=device,
                )
                for adduct in adducts
            ],
            dim=0,
        )

    @classmethod
    def from_symbols_and_adducts(
        cls,
        *,
        symbols: Sequence[str],
        adducts: Sequence[Adduct] = (),
        include_charge: bool = True,
    ) -> "FormulaTensorizer":
        element_order = list(dict.fromkeys(str(symbol) for symbol in symbols))

        for adduct in adducts:
            for element, count in adduct.element_diff.items():
                if count != 0 and element not in element_order:
                    element_order.append(element)

        return cls(
            element_order=tuple(element_order),
            include_charge=include_charge,
        )

    @classmethod
    def from_element_order(
        cls,
        element_order: Sequence[str],
        *,
        include_charge: bool = True,
    ) -> "FormulaTensorizer":
        return cls(
            element_order=tuple(str(element) for element in element_order),
            include_charge=include_charge,
        )

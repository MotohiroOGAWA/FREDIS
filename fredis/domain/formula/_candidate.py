from __future__ import annotations

from fredis.libs.mmkit.mmkit import Formula


def estimate_subformula_candidate_count(
    formula: Formula,
    *,
    hydrogen_delta: int = 1,
) -> int:
    """Estimate number of possible subformula candidates.

    This estimate includes zero-count candidates for each element.

    Examples
    --------
    C6H6O with hydrogen_delta=0:
        (6 + 1) * (6 + 1) * (1 + 1) = 98

    C6H6O with hydrogen_delta=1:
        (6 + 1) * (6 + 1 + 1) * (1 + 1) = 112
    """

    elements = formula.elements.copy()

    elements["H"] = elements.get("H", 0) + int(hydrogen_delta)
    elements["H"] = max(elements["H"], 0)

    if len(elements) == 0:
        return 0

    count = 1

    for element_count in elements.values():
        count *= int(element_count) + 1

    return count
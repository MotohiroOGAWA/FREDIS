import math
import re

charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75}
nce_instruments = ["Orbitrap", "LC-ESI-QFT", "LC-APCI-ITFT", "Linear Ion Trap", "LC-ESI-ITFT"] # "Flow-injection QqQ/MS",

def NCE_to_eV(nce, precursor_mz, charge=1):
    return float(nce) * float(precursor_mz) / 500 * charge_factor[charge]

def parse_ce_to_ev(ce, precursor_mz, instrument=None) -> float | None:
    """Convert supported CE metadata to eV, returning None on invalid input."""
    try:
        value = _parse_ce_to_ev(ce, precursor_mz, instrument)
        return value if value is not None and math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        # Metadata may contain strings, missing values, or invalid precursor
        # masses. Callers can skip these records or group them as non-finite.
        return None


def _parse_ce_to_ev(ce, precursor_mz, instrument=None) -> float | None:
    try:
        ce = float(ce)
    except (TypeError, ValueError, OverflowError):
        pass
    if type(ce) == float:
        if instrument in nce_instruments:
            return NCE_to_eV(ce, precursor_mz)
        return ce
    elif type(ce) == str:
        if "kev" in ce.lower():
            match = re.search(r"(\d+(?:\.\d+)?)\s*kev", ce.lower())
            if match:
                ce = float(match.group(1)) * 1000
                return ce
        if "ev" in ce.lower():
            match = re.search(r"(\d+(?:\.\d+)?)\s*ev", ce.lower())
            if match:
                ce = float(match.group(1))
                return ce
        if "v" in ce.lower():
            match = re.search(r"(\d+(?:\.\d+)?)\s*v", ce.lower())
            if match:
                ce = float(match.group(1))
                return ce
        if "%" in ce:
            match = re.search(r"(\d+(?:\.\d+)?)\s*%", ce)
            if match:
                nce = float(match.group(1))
                return NCE_to_eV(nce, precursor_mz)
        return None
    else:
        return None

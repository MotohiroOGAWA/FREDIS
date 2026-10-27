"""Write a small illustrative MSDS fixture with disjoint training/validation molecules.

The peaks are synthetic. Use measured experimental spectra for scientific training.
Run from the repository root: python docs/examples/create_demo_dataset.py --output runs/demo-input
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fredis.libs.mmkit.mmkit import Adduct, Compound
from fredis.data.core.MSDataset import MSDataset
from fredis.data.core.PeakSeries import PeakSeries


def write_split(directory, name, smiles, energies, fragment_masses):
    precursor = (
        Adduct.parse("[M+H]+")
        .apply_to_formula(Compound.from_smiles(smiles).formula)
        .normalized
    )
    precursor_mz = precursor.exact_mass
    metadata = pd.DataFrame(
        {
            "SpecID": [f"demo-{name}-{index}" for index in range(len(energies))],
            "SMILES": [smiles] * len(energies),
            "AdductType": ["[M+H]+"] * len(energies),
            "CollisionEnergy": energies,
            "PrecursorMZ": [precursor_mz] * len(energies),
        }
    )
    peaks = []
    for index, _ in enumerate(energies):
        peaks.extend(
            [
                (precursor_mz, 1.0),
                (fragment_masses[0], 0.4 + index * 0.2),
                (fragment_masses[1], 0.2 + index * 0.1),
            ]
        )
    values = np.asarray(peaks, dtype=np.float64)
    series = PeakSeries(values, np.arange(0, len(values) + 1, 3), sort_by_mz=True)
    dataset = MSDataset(metadata, series)
    path = directory / (name + ".msds")
    dataset.save(str(path))
    print(f"{path.name}: {len(dataset)} spectra, {dataset.n_peaks_total} peaks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_split(args.output, "train", "CCO", [20.0, 40.0], [31.01839, 29.03858])
    write_split(
        args.output, "validation", "CC(N)O", [10.0, 30.0], [18.033826, 44.049476]
    )


if __name__ == "__main__":
    main()

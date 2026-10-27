"""Tests of the Python binding. Run from bindings/python:

    python -m unittest discover -s tests -v

The C libraries are taken from MSSIM_C_LIBRARY / MSENTITY_C_LIBRARY or the
build directories; the msentity package must be importable (PYTHONPATH).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import msentity as ms
import numpy as np

import msentity_similarity as mss

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "tests" / "data"
SAMPLE = Path(os.environ.get("MSENTITY_TEST_DATA_DIR", REPO.parent / "msentity" / "tests" / "data")) \
    / "sample.msp"

# Reference values for sample.msp, cosine, bin width 0.01.
ROW0 = [1.0, 0.7569831609725952, 0.85833740234375, 0.6929314732551575, 0.6837769150733948]


class TestCalculation(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = ms.load(SAMPLE)

    def test_all_pairs_and_pair_scores(self) -> None:
        matrix = mss.all_pairs(self.ds, self.ds)
        self.assertEqual(matrix.shape, (5, 5))
        self.assertEqual(matrix.dtype, np.float32)
        np.testing.assert_allclose(matrix[0], ROW0, atol=1e-6)
        scores = mss.pair_scores(self.ds, [0, 2], self.ds, [1, 4], method="reverse_cosine")
        np.testing.assert_allclose(scores, [0.7586880922317505, 0.31010058522224426], atol=1e-6)
        # Views and plain bytes are accepted too.
        np.testing.assert_allclose(mss.all_pairs(self.ds[[0]], self.ds.to_bytes())[0], ROW0,
                                   atol=1e-6)

    def test_errors(self) -> None:
        with self.assertRaises(mss.InvalidArgumentError):
            mss.all_pairs(self.ds, self.ds, method="dot")
        with self.assertRaises(mss.InvalidArgumentError):
            mss.all_pairs(self.ds, self.ds, bin_width=0)
        with self.assertRaises(mss.PositionError):
            mss.pair_scores(self.ds, [9], self.ds, [0])
        with self.assertRaises(mss.FormatError):
            mss.all_pairs(b"junk", self.ds)
        with self.assertRaises(TypeError):
            mss.all_pairs("sample.msp", self.ds)

    def test_library_search(self) -> None:
        calls = []
        result = mss.library_search(self.ds, self.ds, 0.9, query_source="sample.msp",
                                    progress=lambda done, total: calls.append((done, total)))
        self.assertEqual(len(result), 9)
        self.assertEqual(calls[-1], (25, 25))
        self.assertEqual(result.columns, ["index1", "index2", "cosine_similarity",
                                          "data_index1", "data_index2"])
        self.assertEqual(result["index2"], [0, 1, 4, 2, 3, 2, 3, 1, 4])
        metadata = result.metadata
        self.assertEqual(metadata["mode"], "library_search")
        self.assertEqual(metadata["sources"][0]["path"], "sample.msp")
        self.assertEqual(metadata["parameters"]["threshold"], 0.9)

        self.assertTrue(result.has_matched_data)
        query, reference = result.match_records(2)  # pair (1, 4)
        self.assertEqual(query["Name"], self.ds[1]["Name"])
        self.assertEqual(reference["Name"], self.ds[4]["Name"])
        np.testing.assert_array_equal(reference.mz, self.ds[4].mz)
        self.assertEqual(list(result.source_indices(2)), [0, 1, 4, 2, 3])

    def test_by_key(self) -> None:
        self.ds.set_spec_id("SpecID", "SP")
        reordered = self.ds[[4, 3, 2]]
        result = mss.similarity_by_key(self.ds, reordered, method="reverse_cosine")
        # The score column keeps its name for every method.
        self.assertEqual(result.columns, ["SpecID", "index1", "index2", "cosine_similarity"])
        self.assertEqual(result.metadata["parameters"]["method"], "reverse_cosine")
        self.assertEqual(result["index1"], [2, 3, 4])
        self.assertEqual(result["index2"], [2, 1, 0])
        self.assertFalse(result.has_matched_data)
        with self.assertRaises(mss.InvalidArgumentError):
            result.match_records(0)
        with self.assertRaises(mss.ColumnError):
            mss.similarity_by_key(self.ds, self.ds, "Missing", "SpecID")


class TestSimilarityDataset(unittest.TestCase):
    def test_reference_file(self) -> None:
        result = mss.load(DATA / "similarity.mssim")
        self.assertEqual(len(result), 16)
        self.assertTrue(result.has_matched_data)
        self.assertEqual(len(result.matched_datasets[0]), 6)
        self.assertIsInstance(result.metadata, dict)

    def test_filter_sort_save_load(self) -> None:
        ds = ms.load(SAMPLE)
        result = mss.library_search(ds, ds, 0.9)
        filtered = result.filter([i == 1 for i in result["index1"]])
        self.assertEqual(filtered.metadata["source_row_count"], 9)
        ranked = filtered.sort_by("cosine_similarity", ascending=False)
        self.assertEqual(ranked["index2"], [1, 4])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hits.mssim"
            ranked.save(path)
            loaded = mss.load(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual([len(d) for d in loaded.matched_datasets], [1, 2])
            self.assertEqual(list(loaded.source_indices(2)), [1, 4])
            loaded.export(Path(tmp) / "hits.tsv")
            text = (Path(tmp) / "hits.tsv").read_text()
            self.assertTrue(text.startswith("index1\tindex2\tcosine_similarity"))
            with self.assertRaises(mss.InvalidArgumentError):
                loaded.save(Path(tmp) / "hits.txt")
            with self.assertRaises(mss.FileError):
                mss.load(Path(tmp) / "missing.mssim")

    def test_statistics_and_metadata(self) -> None:
        ds = ms.load(SAMPLE)
        result = mss.library_search(ds, ds, 0.0, include_matched_data=False)
        summary = result.describe_scores()
        self.assertEqual(summary["count"], 25)
        self.assertEqual(summary["max"], 1.0)
        counts, edges = result.histogram(4)
        self.assertEqual(counts.sum(), 25)
        self.assertEqual(len(edges), 5)
        result.metadata = {"note": "名前"}
        self.assertEqual(result.metadata, {"note": "名前"})
        with self.assertRaises(mss.InvalidArgumentError):
            result.histogram(0)

    def test_arrow(self) -> None:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        ds = ms.load(SAMPLE)
        table = mss.library_search(ds, ds, 0.9).to_arrow()
        self.assertEqual(table.num_rows, 9)
        self.assertEqual(str(table.schema.field("cosine_similarity").type), "float")


if __name__ == "__main__":
    unittest.main()

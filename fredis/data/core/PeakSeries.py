from __future__ import annotations

from typing import Any, Iterator, Optional, Sequence, Union, overload

import numpy as np
import pandas as pd

from .Peak import Peak


class PeakSeries:
    """
    Collection of spectra stored in a flattened peak array.

    This class stores all peaks in a single two-column array with shape
    ``(n_peaks, 2)``, where the columns correspond to:

    - column 0: m/z
    - column 1: intensity

    Individual spectra are defined by ``offsets``, where the peaks of the
    ``i``-th spectrum are stored in the interval
    ``data[offsets[i]:offsets[i + 1]]``.

    A :class:`PeakSeries` instance can also represent a view over a subset
    of spectra by using ``index``.

    Parameters
    ----------
    data : numpy.ndarray
        Peak array with shape ``(n_peaks, 2)``.
    offsets : numpy.ndarray
        One-dimensional integer array of length ``n_spectra + 1``.
    metadata : pandas.DataFrame, optional
        Peak-level metadata aligned with ``data``.
    metadata_columns : sequence of str, optional
        Metadata columns exposed by this view.
    index : numpy.ndarray, optional
        Spectrum indices included in the current view.
    sort_by_mz : bool, default=False
        If ``True``, sort peaks by m/z within each spectrum during initialization.
    """

    def __init__(
        self,
        data: np.ndarray,
        offsets: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
        metadata_columns: Optional[Sequence[str]] = None,
        index: Optional[np.ndarray] = None,
        sort_by_mz: bool = False,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("data must have shape (n_peaks, 2)")

        if not isinstance(offsets, np.ndarray):
            raise TypeError("offsets must be a numpy.ndarray")
        if offsets.ndim != 1:
            raise ValueError("offsets must be one-dimensional")
        if not np.issubdtype(offsets.dtype, np.integer):
            raise TypeError("offsets must have an integer dtype")

        if metadata is not None:
            if not isinstance(metadata, pd.DataFrame):
                raise TypeError("metadata must be a pandas.DataFrame")
            if len(metadata) != len(data):
                raise ValueError("metadata must have the same number of rows as data")

        if metadata is not None and metadata_columns is None:
            metadata_columns = metadata.columns.tolist()

        if metadata_columns is None:
            metadata_columns = []

        metadata_columns = [col for col in metadata_columns if col not in {"mz", "intensity"}]

        self._data_ref = data
        self._offsets_ref = offsets.astype(np.int64, copy=False)
        self._metadata_ref = metadata
        self._metadata_columns = list(metadata_columns)

        if index is None:
            self._index = np.arange(len(offsets) - 1, dtype=np.int64)
        else:
            index = np.asarray(index, dtype=np.int64)
            self._index = index.copy()

        if sort_by_mz:
            self.sort_by_mz(in_place=True)

    def __len__(self) -> int:
        """Number of spectra in the current view."""
        return int(self._index.size)

    def __repr__(self) -> str:
        """Return a concise representation of the series."""
        return f"PeakSeries(n_spectra={len(self)}, n_peaks={self.n_peaks_total})"

    def __iter__(self) -> Iterator[Spectrum]:
        """
        Iterate over spectra in the current view.

        Yields
        ------
        Spectrum
            Spectrum view for each selected spectrum.
        """
        for i in range(len(self)):
            yield self[i]

    @property
    def offsets(self) -> np.ndarray:
        """
        Offsets of the current view.

        Returns
        -------
        numpy.ndarray
            Offsets relative to the visible spectra in this view.
        """
        lengths = (self._offsets_ref[1:] - self._offsets_ref[:-1])[self._index]
        offsets = np.empty(len(self) + 1, dtype=np.int64)
        offsets[0] = 0
        offsets[1:] = np.cumsum(lengths)
        return offsets

    @property
    def data(self) -> np.ndarray:
        """
        Peak array of the current view.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_visible_peaks, 2)``.
        """
        parts = [
            self._data_ref[self._offsets_ref[i]:self._offsets_ref[i + 1]]
            for i in self._index
        ]
        if not parts:
            return self._data_ref[:0]
        return np.concatenate(parts, axis=0)

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """
        Replace the peak array of the current view.

        Parameters
        ----------
        value : numpy.ndarray
            Replacement array with the same shape as :attr:`data`.
        """
        expected_shape = self.data.shape
        if value.shape != expected_shape:
            raise ValueError(f"value must have shape {expected_shape}, got {value.shape}")

        cursor = 0
        for i in self._index:
            start = int(self._offsets_ref[i])
            end = int(self._offsets_ref[i + 1])
            length = end - start
            self._data_ref[start:end] = value[cursor:cursor + length]
            cursor += length

    @property
    def mz(self) -> np.ndarray:
        """
        m/z values of the current view.

        Returns
        -------
        numpy.ndarray
        """
        return self.data[:, 0]

    @mz.setter
    def mz(self, value: np.ndarray) -> None:
        if value.shape != self.mz.shape:
            raise ValueError(f"value must have shape {self.mz.shape}, got {value.shape}")
        data = self.data
        data[:, 0] = value
        self.data = data

    @property
    def intensity(self) -> np.ndarray:
        """
        Intensity values of the current view.

        Returns
        -------
        numpy.ndarray
        """
        return self.data[:, 1]

    @intensity.setter
    def intensity(self, value: np.ndarray) -> None:
        if value.shape != self.intensity.shape:
            raise ValueError(
                f"value must have shape {self.intensity.shape}, got {value.shape}"
            )
        data = self.data
        data[:, 1] = value
        self.data = data

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """
        Peak-level metadata of the current view.

        Returns
        -------
        pandas.DataFrame or None
        """
        if self._metadata_ref is None:
            return None

        parts = [
            self._metadata_ref.iloc[self._offsets_ref[i]:self._offsets_ref[i + 1]]
            for i in self._index
        ]
        if not parts:
            meta = self._metadata_ref.iloc[0:0]
        else:
            meta = pd.concat(parts, ignore_index=True)

        return meta[self._metadata_columns].reset_index(drop=True)

    @metadata.setter
    def metadata(self, value: pd.DataFrame) -> None:
        if self._metadata_ref is None:
            raise AttributeError("metadata is not available for this PeakSeries")

        expected_rows = len(self.metadata)
        if len(value) != expected_rows:
            raise ValueError(f"metadata must have {expected_rows} rows, got {len(value)}")

        cursor = 0
        for i in self._index:
            start = int(self._offsets_ref[i])
            end = int(self._offsets_ref[i + 1])
            length = end - start
            self._metadata_ref.iloc[start:end] = value.iloc[cursor:cursor + length].values
            cursor += length

    @property
    def metadata_columns(self) -> list[str]:
        """
        Metadata columns exposed by the current view.

        Returns
        -------
        list of str
        """
        return list(self._metadata_columns)

    @metadata_columns.setter
    def metadata_columns(self, columns: Sequence[str]) -> None:
        if self._metadata_ref is None and columns is not None:
            raise AttributeError("metadata is not available for this PeakSeries")

        columns = [col for col in columns if col not in {"mz", "intensity"}]

        if not all(col in self._metadata_ref.columns for col in columns):
            raise ValueError("all metadata columns must exist in metadata")

        self._metadata_columns = list(columns)

    @property
    def count(self) -> int:
        """
        Number of spectra in the current view.

        Returns
        -------
        int
        """
        return len(self)

    @property
    def n_peaks_total(self) -> int:
        """
        Total number of peaks in the current view.

        Returns
        -------
        int
        """
        lengths = (self._offsets_ref[1:] - self._offsets_ref[:-1])[self._index]
        return int(lengths.sum())

    @property
    def peak_indices(self) -> np.ndarray:
        """
        Global peak indices corresponding to the current view.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of row indices into the underlying peak array.
        """
        ranges = [
            np.arange(self._offsets_ref[i], self._offsets_ref[i + 1], dtype=np.int64)
            for i in self._index
        ]
        if not ranges:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(ranges)

    @property
    def lengths(self) -> np.ndarray:
        """
        Number of peaks in each visible spectrum.

        Returns
        -------
        numpy.ndarray
        """
        all_lengths = self._offsets_ref[1:] - self._offsets_ref[:-1]
        return all_lengths[self._index]

    def n_peaks(self, spectrum_index: int) -> int:
        """
        Number of peaks in a specific visible spectrum.

        Parameters
        ----------
        spectrum_index : int
            Spectrum index within the current view.

        Returns
        -------
        int
        """
        i = int(self._index[spectrum_index])
        return int(self._offsets_ref[i + 1] - self._offsets_ref[i])

    @overload
    def __getitem__(self, key: int) -> Spectrum: ...
    @overload
    def __getitem__(self, key: slice) -> PeakSeries: ...
    @overload
    def __getitem__(self, key: Sequence[int]) -> PeakSeries: ...
    @overload
    def __getitem__(self, key: np.ndarray) -> PeakSeries: ...

    def __getitem__(self, key: Union[int, slice, Sequence[int], np.ndarray]) -> Union[Spectrum, PeakSeries]:
        """
        Access one spectrum or a subset of spectra.

        Parameters
        ----------
        key : int, slice, sequence of int, or numpy.ndarray
            Indexer for spectra.

        Returns
        -------
        Spectrum or PeakSeries
        """
        if isinstance(key, int):
            if not (0 <= key < len(self)):
                raise IndexError(f"spectrum index out of range: {key}")
            return Spectrum(self, key)

        if isinstance(key, slice):
            new_index = self._index[key]
            return PeakSeries(
                self._data_ref,
                self._offsets_ref,
                self._metadata_ref,
                self.metadata_columns,
                index=new_index,
            )

        indexer = np.asarray(key, dtype=np.int64)
        new_index = self._index[indexer]
        return PeakSeries(
            self._data_ref,
            self._offsets_ref,
            self._metadata_ref,
            self.metadata_columns,
            index=new_index,
        )

    def __setitem__(self, key: str, value: Union[Sequence[Any], pd.Series, Any]) -> None:
        """
        Add or update a metadata column for the current view.

        Parameters
        ----------
        key : str
            Metadata column name.
        value : sequence, pandas.Series, or scalar
            Values assigned to the peaks in the current view.
        """
        peak_indices = self.peak_indices
        n_total_peaks = int(self._offsets_ref[-1] - self._offsets_ref[0])
        n_view_peaks = self.n_peaks_total

        if self._metadata_ref is None:
            self._metadata_ref = pd.DataFrame(index=range(n_total_peaks))

        if key not in self._metadata_ref.columns:
            self._metadata_ref[key] = pd.Series(dtype=object)

        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if len(value) != n_view_peaks:
                raise ValueError(
                    f"value must have length {n_view_peaks}, got {len(value)}"
                )
            self._metadata_ref.loc[peak_indices, key] = value
        else:
            self._metadata_ref.loc[peak_indices, key] = value

        if key not in self._metadata_columns:
            self._metadata_columns.append(key)

    def copy(self) -> PeakSeries:
        """
        Materialize the current view as an independent object.

        Returns
        -------
        PeakSeries
        """
        return PeakSeries(
            data=self.data.copy(),
            offsets=self.offsets.copy(),
            metadata=None if self.metadata is None else self.metadata.copy(),
            metadata_columns=self.metadata_columns,
            index=None,
        )

    def reset_view(
        self,
        in_place: bool = True,
    ) -> PeakSeries:
        """
        Reset the current view to all spectra.

        Parameters
        ----------
        in_place : bool, default=True
            If True, modify the current object.
            If False, return a new PeakSeries with a full view.

        Returns
        -------
        PeakSeries
            PeakSeries with a full view.
        """
        if in_place:
            self._index = np.arange(
                len(self._offsets_ref) - 1,
                dtype=np.int64,
            )
            return self

        return PeakSeries(
            data=self._data_ref,
            offsets=self._offsets_ref,
            metadata=self._metadata_ref,
            metadata_columns=self.metadata_columns,
            index=None,
        )

    def normalize(self, scale: float = 1.0, in_place: bool = False) -> PeakSeries:
        """
        Normalize intensities so that the maximum intensity of each spectrum becomes ``scale``.

        Parameters
        ----------
        scale : float, default=1.0
            Target maximum intensity in each spectrum.
        in_place : bool, default=False
            If ``True``, modify the current object.

        Returns
        -------
        PeakSeries
        """
        data = self.data if in_place else self.data.copy()
        offsets = self.offsets

        for start, end in zip(offsets[:-1], offsets[1:]):
            if end <= start:
                continue
            max_value = data[start:end, 1].max()
            if max_value > 0:
                data[start:end, 1] = data[start:end, 1] / max_value * scale

        if in_place:
            self.data = data
            return self

        return PeakSeries(
            data=data,
            offsets=offsets.copy(),
            metadata=None if self.metadata is None else self.metadata.copy(),
            metadata_columns=self.metadata_columns,
            index=None,
        )

    def _sort_by(
        self,
        column: str,
        ascending: bool = True,
        in_place: bool = False,
        return_index: bool = False,
    ) -> Union[PeakSeries, tuple[PeakSeries, np.ndarray], tuple[PeakSeries, None], PeakSeries]:
        """
        Sort peaks within each spectrum by a given column.

        Parameters
        ----------
        column : {"mz", "intensity"}
            Column used for sorting.
        ascending : bool, default=True
            Sort order.
        in_place : bool, default=False
            If ``True``, modify the current object.
        return_index : bool, default=False
            If ``True``, also return the permutation indices.

        Returns
        -------
        PeakSeries or tuple
        """
        column_index = {"mz": 0, "intensity": 1}.get(column)
        if column_index is None:
            raise ValueError("column must be 'mz' or 'intensity'")

        data = self.data.copy()
        metadata = None if self.metadata is None else self.metadata.copy()
        offsets = self.offsets
        permutation = np.empty(len(data), dtype=np.int64)

        for start, end in zip(offsets[:-1], offsets[1:]):
            order = np.argsort(data[start:end, column_index], kind="stable")
            if not ascending:
                order = order[::-1]
            permutation[start:end] = order + start

        data = data[permutation]
        if metadata is not None:
            metadata = metadata.iloc[permutation].reset_index(drop=True)

        if in_place:
            self.data = data
            if metadata is not None:
                self.metadata = metadata
            return (self, permutation) if return_index else self

        result = PeakSeries(
            data=data,
            offsets=offsets.copy(),
            metadata=metadata,
            metadata_columns=self.metadata_columns,
            index=None,
        )
        return (result, permutation) if return_index else result

    def sort_by_mz(
        self,
        ascending: bool = True,
        in_place: bool = False,
        return_index: bool = False,
    ) -> Union[PeakSeries, tuple[PeakSeries, np.ndarray]]:
        """
        Sort peaks by m/z within each spectrum.

        Parameters
        ----------
        ascending : bool, default=True
            Sort order.
        in_place : bool, default=False
            If ``True``, modify the current object.
        return_index : bool, default=False
            If ``True``, also return the permutation indices.

        Returns
        -------
        PeakSeries or tuple
        """
        return self._sort_by(
            column="mz",
            ascending=ascending,
            in_place=in_place,
            return_index=return_index,
        )

    def sort_by_intensity(
        self,
        ascending: bool = False,
        in_place: bool = False,
        return_index: bool = False,
    ) -> Union[PeakSeries, tuple[PeakSeries, np.ndarray]]:
        """
        Sort peaks by intensity within each spectrum.

        Parameters
        ----------
        ascending : bool, default=False
            Sort order.
        in_place : bool, default=False
            If ``True``, modify the current object.
        return_index : bool, default=False
            If ``True``, also return the permutation indices.

        Returns
        -------
        PeakSeries or tuple
        """
        return self._sort_by(
            column="intensity",
            ascending=ascending,
            in_place=in_place,
            return_index=return_index,
        )

    def replace_data(
        self,
        data: np.ndarray,
        offsets: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Replace peak data for the current visible spectra.

        ``offsets`` must describe the currently visible spectra, i.e.
        ``len(offsets) - 1 == len(self)``.

        Invisible spectra keep their spectrum slots, but their peak count becomes 0.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("data must have shape (n_peaks, 2)")

        if not isinstance(offsets, np.ndarray):
            raise TypeError("offsets must be a numpy.ndarray")
        if offsets.ndim != 1:
            raise ValueError("offsets must be one-dimensional")
        if not np.issubdtype(offsets.dtype, np.integer):
            raise TypeError("offsets must have an integer dtype")
        offsets = offsets.astype(np.int64, copy=False)

        if len(offsets) != len(self) + 1:
            raise ValueError(
                "len(offsets) - 1 must match the number of visible spectra: "
                f"{len(offsets) - 1} != {len(self)}"
            )
        if len(offsets) == 0 or offsets[0] != 0:
            raise ValueError("offsets must start with 0")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("offsets must be non-decreasing")
        if int(offsets[-1]) != len(data):
            raise ValueError(
                f"offsets[-1] must equal len(data): {offsets[-1]} != {len(data)}"
            )

        if metadata is not None:
            if not isinstance(metadata, pd.DataFrame):
                raise TypeError("metadata must be a pandas.DataFrame")
            if len(metadata) != len(data):
                raise ValueError("metadata must have the same number of rows as data")

        n_all_spectra = len(self._offsets_ref) - 1
        visible_to_new_pos = {
            int(source_index): view_pos
            for view_pos, source_index in enumerate(self._index)
        }

        new_offsets_ref = np.empty(n_all_spectra + 1, dtype=np.int64)
        new_offsets_ref[0] = 0

        cursor = 0
        data_parts = []
        metadata_parts = [] if metadata is not None else None

        for source_index in range(n_all_spectra):
            view_pos = visible_to_new_pos.get(source_index)

            if view_pos is None:
                # Not visible in the current view.
                # Keep the spectrum slot, but make it empty.
                length = 0
            else:
                start = int(offsets[view_pos])
                end = int(offsets[view_pos + 1])
                length = end - start

                if length > 0:
                    data_parts.append(data[start:end])
                    if metadata_parts is not None:
                        metadata_parts.append(metadata.iloc[start:end])

            cursor += length
            new_offsets_ref[source_index + 1] = cursor

        if data_parts:
            self._data_ref = np.concatenate(data_parts, axis=0)
        else:
            self._data_ref = data[:0].copy()

        self._offsets_ref = new_offsets_ref

        if metadata is not None:
            if metadata_parts:
                self._metadata_ref = pd.concat(metadata_parts, ignore_index=True)
            else:
                self._metadata_ref = metadata.iloc[:0].copy()
            self._metadata_columns = [
                col for col in metadata.columns if col not in {"mz", "intensity"}
            ]
        elif self._metadata_ref is not None:
            self._metadata_ref = None
            self._metadata_columns = []

    def reorder(self, order: Sequence[int]) -> PeakSeries:
        """
        Reorder spectra in the current view.

        Parameters
        ----------
        order : sequence of int
            Permutation of ``range(len(self))``.

        Returns
        -------
        PeakSeries
        """
        order = np.asarray(order, dtype=np.int64)

        if set(order.tolist()) != set(range(len(self))) or len(order) != len(self):
            raise ValueError("order must be a permutation of range(len(self))")

        new_index = self._index[order]
        return PeakSeries(
            self._data_ref,
            self._offsets_ref,
            self._metadata_ref,
            self.metadata_columns,
            index=new_index,
        )


class Spectrum:
    """
    View of a single spectrum in a :class:`PeakSeries`.

    Parameters
    ----------
    peak_series : PeakSeries
        Parent peak series.
    index : int
        Spectrum index within the current view of ``peak_series``.
    """

    def __init__(self, peak_series: PeakSeries, index: int) -> None:
        self._peak_series = peak_series
        self._view_index = index

        source_index = int(self._peak_series._index[index])
        self._start = int(self._peak_series._offsets_ref[source_index])
        self._end = int(self._peak_series._offsets_ref[source_index + 1])

    def __len__(self) -> int:
        """Number of peaks in the spectrum."""
        return self._end - self._start

    def __repr__(self) -> str:
        """Return a concise representation of the spectrum."""
        return f"Spectrum(n_peaks={len(self)})"

    def __iter__(self) -> Iterator[Peak]:
        """
        Iterate over peaks in the spectrum.

        Yields
        ------
        Peak
        """
        for i in range(len(self)):
            yield self[i]

    @overload
    def __getitem__(self, key: int) -> Peak: ...
    @overload
    def __getitem__(self, key: str) -> pd.Series: ...

    def __getitem__(self, key: Union[int, str]) -> Union[Peak, pd.Series]:
        """
        Access a peak or a metadata column.

        Parameters
        ----------
        key : int or str
            Peak index or metadata column name.

        Returns
        -------
        Peak or pandas.Series
        """
        if isinstance(key, int):
            if not (0 <= key < len(self)):
                raise IndexError(f"peak index out of range: {key}")

            row_index = self._start + key
            mz, intensity = self._peak_series._data_ref[row_index]

            metadata = None
            if self._peak_series._metadata_ref is not None:
                row = self._peak_series._metadata_ref.iloc[row_index]
                metadata = {
                    column: row[column]
                    for column in self._peak_series.metadata_columns
                }

            return Peak(
                mz=float(mz),
                intensity=float(intensity),
                metadata=metadata,
            )

        if isinstance(key, str):
            if self._peak_series._metadata_ref is None:
                raise KeyError("metadata is not available")
            if key not in self._peak_series.metadata_columns:
                raise KeyError(f"metadata column not found: {key}")

            return (
                self._peak_series._metadata_ref.iloc[self._start:self._end][key]
                .reset_index(drop=True)
            )

        raise TypeError("key must be an int or str")

    def __setitem__(self, key: str, value: Union[Sequence[Any], pd.Series, Any]) -> None:
        """
        Add or update a metadata column for this spectrum.

        Parameters
        ----------
        key : str
            Metadata column name.
        value : sequence, pandas.Series, or scalar
            Values assigned to the peaks in this spectrum.
        """
        n_peaks = len(self)

        if self._peak_series._metadata_ref is None:
            self._peak_series._metadata_ref = pd.DataFrame(
                index=range(len(self._peak_series._data_ref))
            )

        if key not in self._peak_series.metadata_columns:
            if key not in self._peak_series._metadata_ref.columns:
                self._peak_series._metadata_ref[key] = pd.NA
                columns = self._peak_series.metadata_columns
                columns.append(key)
                self._peak_series.metadata_columns = columns
            else:
                raise ValueError(
                    f"metadata column '{key}' exists in metadata but is not exposed"
                )

        row_index = range(self._start, self._end)

        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if len(value) != n_peaks:
                raise ValueError(f"value must have length {n_peaks}, got {len(value)}")
            self._peak_series._metadata_ref.loc[row_index, key] = value
        else:
            self._peak_series._metadata_ref.loc[row_index, key] = value

    def __str__(self) -> str:
        """
        Return a simple tabular string representation.

        Returns
        -------
        str
        """
        rows: list[dict[str, str]] = []
        for peak in self:
            row = {
                "mz": f"{peak.mz:.4f}",
                "intensity": f"{peak.intensity:.4f}",
            }
            for name, value in peak.metadata.items():
                row[name] = str(value)
            rows.append(row)

        columns = ["mz", "intensity"]
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)

        if not rows:
            return "mz  intensity"

        widths = {
            column: max(len(column), max(len(row.get(column, "")) for row in rows))
            for column in columns
        }

        header = "  ".join(f"{column:<{widths[column]}}" for column in columns)
        body = [
            "  ".join(f"{row.get(column, ''):<{widths[column]}}" for column in columns)
            for row in rows
        ]
        return "\n".join([header] + body)

    def __eq__(self, other: object) -> bool:
        """
        Compare two spectra.

        Two spectra are considered equal when both peak values and metadata
        are equal.

        Parameters
        ----------
        other : object

        Returns
        -------
        bool
        """
        if not isinstance(other, Spectrum):
            return False

        if self.data.shape != other.data.shape:
            return False
        if not np.allclose(self.data, other.data, atol=1e-8):
            return False

        meta1 = self.metadata
        meta2 = other.metadata

        if meta1 is None and meta2 is None:
            return True
        if (meta1 is None) != (meta2 is None):
            return False
        if set(meta1.columns) != set(meta2.columns):
            return False

        for column in meta1.columns:
            if not meta1[column].equals(meta2[column]):
                return False

        return True

    @property
    def data(self) -> np.ndarray:
        """
        Peak array of the spectrum.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_peaks, 2)``.
        """
        return self._peak_series._data_ref[self._start:self._end]

    @data.setter
    def data(self, value: np.ndarray) -> None:
        expected_shape = (len(self), 2)
        if value.shape != expected_shape:
            raise ValueError(f"value must have shape {expected_shape}, got {value.shape}")
        self._peak_series._data_ref[self._start:self._end] = value

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """
        Peak-level metadata of the spectrum.

        Returns
        -------
        pandas.DataFrame or None
        """
        if self._peak_series._metadata_ref is None:
            return None
        meta = self._peak_series._metadata_ref.iloc[self._start:self._end]
        return meta[self._peak_series.metadata_columns].reset_index(drop=True)

    @property
    def mz(self) -> np.ndarray:
        """
        m/z values of the spectrum.

        Returns
        -------
        numpy.ndarray
        """
        return self._peak_series._data_ref[self._start:self._end, 0]

    @mz.setter
    def mz(self, value: np.ndarray) -> None:
        if value.shape != self.mz.shape:
            raise ValueError(f"value must have shape {self.mz.shape}, got {value.shape}")
        self._peak_series._data_ref[self._start:self._end, 0] = value

    @property
    def intensity(self) -> np.ndarray:
        """
        Intensity values of the spectrum.

        Returns
        -------
        numpy.ndarray
        """
        return self._peak_series._data_ref[self._start:self._end, 1]

    @intensity.setter
    def intensity(self, value: np.ndarray) -> None:
        if value.shape != self.intensity.shape:
            raise ValueError(
                f"value must have shape {self.intensity.shape}, got {value.shape}"
            )
        self._peak_series._data_ref[self._start:self._end, 1] = value

    def normalize(self, scale: float = 1.0, in_place: bool = False) -> Spectrum:
        """
        Normalize intensities so that the maximum intensity becomes ``scale``.

        Parameters
        ----------
        scale : float, default=1.0
            Target maximum intensity.
        in_place : bool, default=False
            If ``True``, modify the current object.

        Returns
        -------
        Spectrum
        """
        data = self.data.copy()
        max_value = data[:, 1].max() if len(data) > 0 else 0.0
        if max_value > 0:
            data[:, 1] = data[:, 1] / max_value * scale

        if in_place:
            self.data = data
            return self

        new_series = PeakSeries(
            data=data,
            offsets=np.array([0, len(data)], dtype=np.int64),
            metadata=None if self.metadata is None else self.metadata.copy(),
            metadata_columns=self._peak_series.metadata_columns,
        )
        return Spectrum(new_series, 0)

    def sort_by_mz(self, ascending: bool = True, in_place: bool = False) -> Spectrum:
        """
        Sort peaks by m/z.

        Parameters
        ----------
        ascending : bool, default=True
            Sort order.
        in_place : bool, default=False
            If ``True``, modify the current object.

        Returns
        -------
        Spectrum
        """
        data = self.data.copy()
        order = np.argsort(data[:, 0], kind="stable")
        if not ascending:
            order = order[::-1]

        data = data[order]
        metadata = None if self.metadata is None else self.metadata.iloc[order].reset_index(drop=True)

        if in_place:
            self.data = data
            if metadata is not None:
                self._peak_series._metadata_ref.iloc[self._start:self._end] = metadata.values
            return self

        new_series = PeakSeries(
            data=data,
            offsets=np.array([0, len(data)], dtype=np.int64),
            metadata=metadata,
            metadata_columns=self._peak_series.metadata_columns,
        )
        return Spectrum(new_series, 0)

    def sort_by_intensity(self, ascending: bool = False, in_place: bool = False) -> Spectrum:
        """
        Sort peaks by intensity.

        Parameters
        ----------
        ascending : bool, default=False
            Sort order.
        in_place : bool, default=False
            If ``True``, modify the current object.

        Returns
        -------
        Spectrum
        """
        data = self.data.copy()
        order = np.argsort(data[:, 1], kind="stable")
        if not ascending:
            order = order[::-1]

        data = data[order]
        metadata = None if self.metadata is None else self.metadata.iloc[order].reset_index(drop=True)

        if in_place:
            self.data = data
            if metadata is not None:
                self._peak_series._metadata_ref.iloc[self._start:self._end] = metadata.values
            return self

        new_series = PeakSeries(
            data=data,
            offsets=np.array([0, len(data)], dtype=np.int64),
            metadata=metadata,
            metadata_columns=self._peak_series.metadata_columns,
        )
        return Spectrum(new_series, 0)
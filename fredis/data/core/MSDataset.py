from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Literal, Optional, Sequence, Union, overload

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .PeakSeries import PeakSeries, Spectrum


@dataclass(frozen=True)
class MSDatasetMeta:
    """
    Summary metadata stored at the dataset level.

    Parameters
    ----------
    description : str
        Human-readable description of the dataset.
    attributes : dict of str to str
        Arbitrary key-value attributes associated with the dataset.
    tags : list of str
        List of dataset tags.
    """

    description: str
    attributes: Dict[str, str]
    tags: list[str]


class MSDataset:
    """
    Dataset of mass spectra and associated metadata.

    An :class:`MSDataset` combines:

    - spectrum-level metadata stored in a pandas DataFrame
    - peak-level data stored in a :class:`PeakSeries`

    Each row of the spectrum metadata corresponds to one spectrum in the
    peak series.

    Parameters
    ----------
    spectrum_metadata : pandas.DataFrame
        Spectrum-level metadata. Each row corresponds to one spectrum.
    peak_series : PeakSeries
        Peak series containing the spectra.
    columns : sequence of str, optional
        Spectrum metadata columns exposed by this view. If omitted, all columns
        are included.
    description : str, default=""
        Dataset description.
    attributes : dict of str to str, optional
        Dataset-level attributes.
    tags : sequence of str, optional
        Dataset-level tags.
    """

    _ARROW_BYTES_LIMIT = 2_147_483_646
    _MAX_PART_BYTES = 1_000_000_000

    def __init__(
        self,
        spectrum_metadata: pd.DataFrame,
        peak_series: PeakSeries,
        columns: Optional[Sequence[str]] = None,
        description: str = "",
        attributes: Optional[Dict[str, str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        if not isinstance(spectrum_metadata, pd.DataFrame):
            raise TypeError("spectrum_metadata must be a pandas.DataFrame")
        if not isinstance(peak_series, PeakSeries):
            raise TypeError("peak_series must be a PeakSeries")

        if len(spectrum_metadata) != len(peak_series._offsets_ref) - 1:
            raise ValueError(
                "The number of rows in spectrum_metadata must match "
                "the number of spectra in peak_series"
            )

        if columns is None:
            columns = spectrum_metadata.columns.tolist()
        else:
            missing = [col for col in columns if col not in spectrum_metadata.columns]
            if missing:
                raise ValueError(
                    f"All columns must exist in spectrum_metadata. Missing: {missing}"
                )

        self._spectrum_metadata_ref = spectrum_metadata
        self._peak_series = peak_series
        self._columns = list(columns)

        self._description = ""
        self.description = description

        self._attributes: Dict[str, str] = {}
        self.attributes = {} if attributes is None else attributes

        self._tags: list[str] = []
        self.tags = [] if tags is None else list(tags)

    def __repr__(self) -> str:
        """Return a concise representation of the dataset."""
        return (
            f"MSDataset(n_spectra={len(self)}, "
            f"n_peaks={self.n_peaks_total}, "
            f"columns={self.columns})"
        )

    def __len__(self) -> int:
        """Number of spectra in the current dataset view."""
        return len(self._peak_series)

    def __iter__(self) -> Iterator[SpectrumRecord]:
        """
        Iterate over spectra in the dataset.

        Yields
        ------
        SpectrumRecord
            Spectrum records in the current view.
        """
        for i in range(len(self)):
            yield self[i]

    @property
    def columns(self) -> list[str]:
        """
        Spectrum metadata columns exposed by the current view.

        Returns
        -------
        list of str
        """
        return list(self._columns)

    @columns.setter
    def columns(self, columns: Sequence[str]) -> None:
        missing = [col for col in columns if col not in self._spectrum_metadata_ref.columns]
        if missing:
            raise ValueError(
                f"All columns must exist in spectrum_metadata. Missing: {missing}"
            )
        self._columns = list(columns)

    @property
    def n_rows(self) -> int:
        """
        Number of spectra in the current view.

        Returns
        -------
        int
        """
        return len(self)

    @property
    def n_columns(self) -> int:
        """
        Number of visible metadata columns.

        Returns
        -------
        int
        """
        return len(self._columns)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of the dataset view.

        Returns
        -------
        tuple of int
            ``(n_spectra, n_columns)``
        """
        return (self.n_rows, self.n_columns)

    @property
    def n_peaks_total(self) -> int:
        """
        Total number of peaks across all visible spectra.

        Returns
        -------
        int
        """
        return self._peak_series.n_peaks_total

    @property
    def description(self) -> str:
        """
        Dataset description.

        Returns
        -------
        str
        """
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("description must be a string")
        self._description = value

    @property
    def attributes(self) -> Dict[str, str]:
        """
        Dataset-level attributes.

        Returns
        -------
        dict of str to str
        """
        return dict(self._attributes)

    @attributes.setter
    def attributes(self, value: Dict[str, str]) -> None:
        if not isinstance(value, dict):
            raise TypeError("attributes must be a dictionary")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
            raise TypeError("all attribute keys and values must be strings")
        self._attributes = dict(value)

    def set_attribute(self, key: str, value: str) -> None:
        """
        Add or update a dataset attribute.

        Parameters
        ----------
        key : str
            Attribute name.
        value : str
            Attribute value.
        """
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("key and value must be strings")
        self._attributes[key] = value

    def remove_attribute(self, key: str) -> bool:
        """
        Remove a dataset attribute.

        Parameters
        ----------
        key : str
            Attribute name.

        Returns
        -------
        bool
            ``True`` if the attribute existed and was removed.
        """
        if key in self._attributes:
            del self._attributes[key]
            return True
        return False

    def has_attribute(self, key: str) -> bool:
        """
        Check whether an attribute exists.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
        """
        return key in self._attributes

    def clear_attributes(self) -> None:
        """Remove all dataset attributes."""
        self._attributes.clear()

    @property
    def tags(self) -> list[str]:
        """
        Dataset tags.

        Returns
        -------
        list of str
        """
        return list(self._tags)

    @tags.setter
    def tags(self, value: Sequence[str]) -> None:
        if any(not isinstance(tag, str) for tag in value):
            raise TypeError("all tags must be strings")
        self._tags = list(value)

    def add_tag(self, tag: str) -> bool:
        """
        Add a tag if it does not already exist.

        Parameters
        ----------
        tag : str

        Returns
        -------
        bool
            ``True`` if the tag was added.
        """
        if not isinstance(tag, str):
            raise TypeError("tag must be a string")
        if tag not in self._tags:
            self._tags.append(tag)
            return True
        return False

    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag.

        Parameters
        ----------
        tag : str

        Returns
        -------
        bool
            ``True`` if the tag existed and was removed.
        """
        if tag in self._tags:
            self._tags.remove(tag)
            return True
        return False

    def has_tag(self, tag: str) -> bool:
        """
        Check whether a tag exists.

        Parameters
        ----------
        tag : str

        Returns
        -------
        bool
        """
        return tag in self._tags

    def clear_tags(self) -> None:
        """Remove all dataset tags."""
        self._tags.clear()

    @property
    def metadata(self) -> pd.DataFrame:
        """
        Spectrum-level metadata for the current view.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the visible spectra and selected columns.

        Notes
        -----
        The returned DataFrame is a view-derived table for convenience and
        should be treated as read-only unless explicitly reassigned through
        :meth:`__setitem__` or other dataset methods.
        """
        view = self._spectrum_metadata_ref.iloc[self._peak_series._index.tolist()]
        return view[self._columns].reset_index(drop=True)

    @property
    def peaks(self) -> PeakSeries:
        """
        Peak series associated with the current view.

        Returns
        -------
        PeakSeries
        """
        return self._peak_series

    @overload
    def __getitem__(self, key: int) -> SpectrumRecord: ...
    @overload
    def __getitem__(self, key: slice) -> MSDataset: ...
    @overload
    def __getitem__(self, key: Sequence[int]) -> MSDataset: ...
    @overload
    def __getitem__(self, key: np.ndarray) -> MSDataset: ...
    @overload
    def __getitem__(self, key: str) -> pd.Series: ...
    @overload
    def __getitem__(self, key: pd.Series) -> MSDataset: ...

    def __getitem__(
        self,
        key: Union[int, slice, Sequence[int], np.ndarray, str, pd.Series],
    ) -> Union[SpectrumRecord, MSDataset, pd.Series]:
        """
        Access a single spectrum, a subset of spectra, or a metadata column.

        Parameters
        ----------
        key : int, slice, sequence of int, numpy.ndarray, pandas.Series, or str
            Indexer for spectra, boolean mask, or metadata column name.

        Returns
        -------
        SpectrumRecord, MSDataset, or pandas.Series
        """
        if isinstance(key, int):
            if not (0 <= key < len(self)):
                raise IndexError(f"spectrum index out of range: {key}")
            return SpectrumRecord(self, key)

        if isinstance(key, str):
            if key not in self._columns:
                raise KeyError(f"column not available in current view: {key}")
            return self.metadata[key]

        if isinstance(key, pd.Series):
            if key.dtype != bool:
                raise TypeError("pandas.Series indexer must be boolean")
            if len(key) != len(self):
                raise ValueError(
                    f"boolean index length {len(key)} does not match dataset length {len(self)}"
                )
            indices = np.flatnonzero(key.to_numpy())
            return MSDataset(
                self._spectrum_metadata_ref,
                self._peak_series[indices],
                columns=self._columns,
                description=self.description,
                attributes=self.attributes,
                tags=self.tags,
            )

        return MSDataset(
            self._spectrum_metadata_ref,
            self._peak_series[key],
            columns=self._columns,
            description=self.description,
            attributes=self.attributes,
            tags=self.tags,
        )

    def __setitem__(self, key: str, value: Union[Sequence[Any], pd.Series, Any]) -> None:
        """
        Add or update a spectrum metadata column for the current view.

        Parameters
        ----------
        key : str
            Metadata column name.
        value : sequence, pandas.Series, or scalar
            Values assigned to the visible spectra.
        """
        row_indices = self._peak_series._index
        n_visible = len(self)
        n_total = len(self._spectrum_metadata_ref)

        if key not in self._spectrum_metadata_ref.columns:
            self._spectrum_metadata_ref[key] = np.full(n_total, np.nan, dtype=object)

        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if len(value) != n_visible:
                raise ValueError(
                    f"value must have length {n_visible}, got {len(value)}"
                )
            self._spectrum_metadata_ref.loc[row_indices, key] = value
        else:
            self._spectrum_metadata_ref.loc[row_indices, key] = value

        if key not in self._columns:
            self._columns.append(key)

    def copy(self) -> MSDataset:
        """
        Materialize the current view as an independent dataset.

        Returns
        -------
        MSDataset
        """
        return MSDataset(
            spectrum_metadata=self.metadata.copy(),
            peak_series=self._peak_series.copy(),
            columns=self.columns,
            description=self.description,
            attributes=self.attributes,
            tags=self.tags,
        )

    def reset_view(
        self,
        *,
        reset_columns: bool = False,
        in_place: bool = True,
    ) -> MSDataset:
        """
        Reset the current dataset view to the full dataset.

        Parameters
        ----------
        reset_columns : bool, default=False
            If True, expose all spectrum metadata columns.
            If False, keep the current visible columns.
        in_place : bool, default=True
            If True, modify the current object.
            If False, return a new MSDataset with a full view.

        Returns
        -------
        MSDataset
            MSDataset with a full view.
        """
        columns = (
            self._spectrum_metadata_ref.columns.tolist()
            if reset_columns
            else self.columns
        )

        if in_place:
            self._peak_series.reset_view(
                in_place=True,
            )
            self._columns = columns
            return self

        return MSDataset(
            spectrum_metadata=self._spectrum_metadata_ref,
            peak_series=self._peak_series.reset_view(
                in_place=False,
            ),
            columns=columns,
            description=self.description,
            attributes=self.attributes,
            tags=self.tags,
        )

    def sort_by(self, column: str, ascending: bool = True) -> MSDataset:
        """
        Sort spectra by a spectrum metadata column.

        Parameters
        ----------
        column : str
            Metadata column used for sorting.
        ascending : bool, default=True
            Sort order.

        Returns
        -------
        MSDataset
        """
        if column not in self._columns:
            raise KeyError(f"column not available in current view: {column}")

        order = self[column].sort_values(ascending=ascending).index.to_numpy()

        return MSDataset(
            self._spectrum_metadata_ref,
            self._peak_series.reorder(order),
            columns=self._columns,
            description=self.description,
            attributes=self.attributes,
            tags=self.tags,
        )

    def filter_by_peak(
        self, mz: float, tolerance: float | str, intensity: float = 0.0
    ) -> MSDataset:
        """Return spectra containing a peak within ``tolerance`` of ``mz``.

        A numeric tolerance is interpreted as Da; a string may use the
        ``"10ppm"`` form. ``intensity`` is relative to each spectrum's base
        peak (maximum intensity = 1.0) and defaults to 0.0, which disables the
        intensity condition.
        """
        lower, upper, intensity = self._peak_filter_bounds(
            mz, tolerance, intensity, "m/z"
        )
        mask = self._peak_filter_mask(lower, upper, intensity)
        return self[np.flatnonzero(mask)]

    def filter_by_neutral_loss(
        self,
        neutral_loss: float,
        tolerance: float | str,
        intensity: float = 0.0,
        *,
        precursor_mz_column: str = "PrecursorMZ",
    ) -> MSDataset:
        """Return spectra containing a neutral loss within ``tolerance``.

        Neutral loss is ``precursor m/z - peak m/z``. A numeric tolerance is
        interpreted as Da; a string may use the ``"10ppm"`` form. ``intensity``
        is normalized to each spectrum's base peak and defaults to 0.0.
        """
        lower, upper, intensity = self._peak_filter_bounds(
            neutral_loss, tolerance, intensity, "neutral loss"
        )
        if precursor_mz_column not in self._columns:
            raise KeyError(f"column not available in current view: {precursor_mz_column}")
        precursor_mz = pd.to_numeric(
            self[precursor_mz_column], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.all(np.isfinite(precursor_mz)):
            raise ValueError(
                f"column {precursor_mz_column!r} must contain finite numeric values"
            )
        mask = self._peak_filter_mask(
            lower, upper, intensity, precursor_mz=precursor_mz
        )
        return self[np.flatnonzero(mask)]

    @staticmethod
    def _peak_filter_bounds(
        value: float, tolerance: float | str, intensity: float, value_name: str
    ) -> tuple[float, float, float]:
        try:
            center = float(value)
            threshold = float(intensity)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{value_name} and intensity must be numeric") from exc
        if not np.all(np.isfinite((center, threshold))):
            raise ValueError(f"{value_name} and intensity must be finite")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("intensity must be between 0.0 and 1.0")

        tolerance_text = str(tolerance).strip().lower()
        is_ppm = tolerance_text.endswith("ppm")
        if is_ppm:
            tolerance_text = tolerance_text[:-3].strip()
        try:
            tolerance_value = float(tolerance_text)
        except (TypeError, ValueError) as exc:
            raise TypeError("tolerance must be a number in Da or use the '10ppm' form") from exc
        if not np.isfinite(tolerance_value) or tolerance_value < 0:
            raise ValueError("tolerance must be finite and greater than or equal to zero")
        delta = abs(center) * tolerance_value / 1_000_000 if is_ppm else tolerance_value
        return center - delta, center + delta, threshold

    def _peak_filter_mask(
        self,
        lower: float,
        upper: float,
        intensity: float,
        *,
        precursor_mz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mask = np.zeros(len(self), dtype=bool)
        for position, spectrum in enumerate(self._peak_series):
            values = spectrum.mz
            if precursor_mz is not None:
                values = precursor_mz[position] - values
            in_range = (values >= lower) & (values <= upper)
            if intensity > 0.0:
                maximum = np.max(spectrum.intensity, initial=0.0)
                if maximum <= 0.0:
                    mask[position] = False
                    continue
                in_range &= spectrum.intensity / maximum >= intensity
            mask[position] = bool(np.any(in_range))
        return mask

    @classmethod
    def concat(
        cls,
        datasets: Sequence[MSDataset],
        description: str = "",
        attributes: Optional[Dict[str, str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> MSDataset:
        """
        Concatenate multiple datasets.

        Parameters
        ----------
        datasets : sequence of MSDataset
            Datasets to concatenate.
        description : str, default=""
            Description of the concatenated dataset.
        attributes : dict of str to str, optional
            Dataset-level attributes.
        tags : sequence of str, optional
            Dataset-level tags.

        Returns
        -------
        MSDataset
        """
        if not datasets:
            raise ValueError("datasets must not be empty")

        datasets = [ds.copy() for ds in datasets]

        spectrum_metadata = pd.concat(
            [ds._spectrum_metadata_ref for ds in datasets],
            ignore_index=True,
        )

        all_columns: list[str] = []
        seen_columns: set[str] = set()
        for ds in datasets:
            for column in ds.columns:
                if column not in seen_columns:
                    seen_columns.add(column)
                    all_columns.append(column)

        data = np.concatenate([ds.peaks._data_ref for ds in datasets], axis=0)

        offsets = [0]
        peak_offset = 0
        for ds in datasets:
            segment_offsets = ds.peaks._offsets_ref[1:] + peak_offset
            offsets.extend(segment_offsets.tolist())
            peak_offset += int(ds.peaks._offsets_ref[-1])
        offsets_array = np.asarray(offsets, dtype=np.int64)

        peak_metadata_frames = [
            ds.peaks._metadata_ref for ds in datasets if ds.peaks._metadata_ref is not None
        ]
        peak_metadata = (
            pd.concat(peak_metadata_frames, ignore_index=True)
            if peak_metadata_frames
            else None
        )

        peak_metadata_columns: list[str] = []
        seen_peak_columns: set[str] = set()
        for ds in datasets:
            for column in ds.peaks.metadata_columns:
                if column not in seen_peak_columns:
                    seen_peak_columns.add(column)
                    peak_metadata_columns.append(column)

        peak_series = PeakSeries(
            data=data,
            offsets=offsets_array,
            metadata=peak_metadata,
            metadata_columns=peak_metadata_columns,
        )

        return cls(
            spectrum_metadata=spectrum_metadata,
            peak_series=peak_series,
            columns=all_columns,
            description=description,
            attributes={} if attributes is None else attributes,
            tags=[] if tags is None else tags,
        )

    def merge_metadata(
        self,
        right: pd.DataFrame,
        *,
        on: str = "SMILES",
        add_columns: Optional[Sequence[str]] = None,
        right_prefix: str = "",
        overwrite: bool = False,
        drop_right_duplicates: bool = True,
        keep: Literal["first", "last"] = "first",
    ) -> MSDataset:
        """
        Merge an external DataFrame into the spectrum metadata of the current view.

        Parameters
        ----------
        right : pandas.DataFrame
            External metadata table.
        on : str, default="SMILES"
            Join key.
        add_columns : sequence of str, optional
            Columns to import from ``right``. If omitted, all columns except ``on``
            are imported.
        right_prefix : str, default=""
            Prefix added to imported column names.
        overwrite : bool, default=False
            If ``False``, fill only missing values in the existing metadata.
        drop_right_duplicates : bool, default=True
            Whether to drop duplicate keys in ``right`` before merging.
        keep : {"first", "last"}, default="first"
            Which duplicate to keep when ``drop_right_duplicates`` is enabled.

        Returns
        -------
        MSDataset
            The current dataset instance.
        """
        if on not in self._spectrum_metadata_ref.columns:
            raise KeyError(f"left metadata does not contain key column: {on}")
        if on not in right.columns:
            raise KeyError(f"right metadata does not contain key column: {on}")

        right_df = right.copy()

        if add_columns is None:
            add_columns = [column for column in right_df.columns if column != on]
        else:
            missing = [column for column in add_columns if column not in right_df.columns]
            if missing:
                raise KeyError(f"right metadata is missing columns: {missing}")
            add_columns = list(add_columns)

        if drop_right_duplicates:
            right_df = right_df.drop_duplicates(subset=[on], keep=keep)

        right_df = right_df[[on] + list(add_columns)]

        rename_map = {
            column: f"{right_prefix}{column}" for column in add_columns if right_prefix
        }
        if rename_map:
            right_df = right_df.rename(columns=rename_map)

        imported_columns = [rename_map.get(column, column) for column in add_columns]

        row_indices = np.asarray(self._peak_series._index)
        left_view = self._spectrum_metadata_ref.loc[row_indices, [on]].copy()
        left_view["_row_index__"] = row_indices

        merged = left_view.merge(right_df, on=on, how="left")

        for column in imported_columns:
            if column not in self._spectrum_metadata_ref.columns:
                self._spectrum_metadata_ref[column] = pd.Series(dtype=object)

            if overwrite:
                self._spectrum_metadata_ref.loc[
                    merged["_row_index__"], column
                ] = merged[column].to_numpy()
            else:
                current = self._spectrum_metadata_ref.loc[merged["_row_index__"], column]
                fill_mask = current.isna().to_numpy()
                row_ids = merged["_row_index__"].to_numpy()
                values = merged[column].to_numpy()
                self._spectrum_metadata_ref.loc[row_ids[fill_mask], column] = values[fill_mask]

            if column not in self._columns:
                self._columns.append(column)

        return self

    def save(
        self,
        path: str,
        save_view: bool = True,
        mode: Literal["w", "a"] = "w",
    ):
        """
        Save the dataset to an .msds file.

        Parameters
        ----------
        path : str
            Output file path.
        save_view : bool, default=True
            If ``True``, save the current view. If ``False``, save the underlying
            references as-is.
        mode : {"w", "a"}, default="w"
            .msds write mode.
        """
        from ..native import to_native
        if mode not in {"w", "a"}:
            raise ValueError("mode must be 'w' or 'a'")
        dataset = self if save_view else self.reset_view(in_place=False)
        with to_native(dataset) as native:
            native.save(path, format="msds", append=mode == "a")

    @classmethod
    def load(
        cls,
        path: str,
        load_peak_metadata: bool = True,
    ) -> "MSDataset":
        """
        Load a dataset from an .msds file.

        Parameters
        ----------
        path : str
            Input file path.
        load_peak_metadata : bool, default=True
            If ``True``, load peak metadata. If ``False``, skip loading peak metadata.
        """
        import msentity
        from ..native import from_native
        with msentity.load(path, file_type="msds") as native:
            return from_native(native, load_peak_metadata=load_peak_metadata)

    @classmethod
    def read_dataset_meta(cls, path: str) -> MSDatasetMeta:
        """
        Read only dataset-level metadata from an HDF5 file.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        MSDatasetMeta
        """
        with h5py.File(path, "r") as file:
            if "metadata" not in file:
                raise KeyError("HDF5 file does not contain a '/metadata' group")

            metadata_group = file["metadata"]

            def _as_str(value: Any) -> str:
                if value is None:
                    return ""
                if isinstance(value, (bytes, bytearray)):
                    return value.decode("utf-8")
                return str(value)

            description = _as_str(metadata_group.attrs.get("description", ""))
            attributes_json = _as_str(metadata_group.attrs.get("attributes_json", "{}"))
            tags_json = _as_str(metadata_group.attrs.get("tags_json", "[]"))

            try:
                attributes = json.loads(attributes_json) if attributes_json else {}
            except json.JSONDecodeError:
                attributes = {}

            try:
                tags = json.loads(tags_json) if tags_json else []
            except json.JSONDecodeError:
                tags = []

            if not isinstance(attributes, dict):
                attributes = {}
            else:
                attributes = {str(k): str(v) for k, v in attributes.items()}

            if not isinstance(tags, list):
                tags = []
            else:
                tags = [str(tag) for tag in tags]

            return MSDatasetMeta(
                description=description,
                attributes=attributes,
                tags=tags,
            )

    @staticmethod
    def _dump_parquet_to_bytes(dataframe: pd.DataFrame) -> bytes:
        """Serialize a DataFrame to Parquet bytes."""
        buffer = io.BytesIO()
        dataframe.to_parquet(buffer, engine="pyarrow")
        return buffer.getvalue()

    @staticmethod
    def _read_parquet_from_bytes(blob: bytes) -> pd.DataFrame:
        """Deserialize Parquet bytes into a DataFrame."""
        return pd.read_parquet(io.BytesIO(blob), engine="pyarrow")

    @staticmethod
    def _parquet_uncompressed_bytes(blob: bytes) -> int:
        """Estimate uncompressed size from Parquet metadata."""
        parquet_file = pq.ParquetFile(io.BytesIO(blob))
        metadata = parquet_file.metadata
        total = 0
        for i in range(metadata.num_row_groups):
            total += metadata.row_group(i).total_byte_size
        return int(total)

    @staticmethod
    def _save_bytes_h5(
        group: h5py.Group,
        name: str,
        blob: bytes,
        *,
        compression: Optional[str] = "gzip",
        chunks: bool = True,
    ) -> None:
        """Save raw bytes as a uint8 HDF5 dataset."""
        if name in group:
            del group[name]

        array = np.frombuffer(memoryview(blob), dtype=np.uint8)
        kwargs: dict[str, Any] = {}
        if compression is not None:
            kwargs["compression"] = compression
        if chunks:
            kwargs["chunks"] = True

        dataset = group.create_dataset(name, data=array, **kwargs)
        dataset.attrs["bytes_format"] = "uint8_1d"
        dataset.attrs["nbytes"] = int(len(blob))

    @staticmethod
    def _load_bytes_h5(group: h5py.Group, name: str) -> bytes:
        """Load raw bytes from an HDF5 dataset."""
        if name not in group:
            raise KeyError(f"dataset '{name}' not found in group '{group.name}'")

        dataset = group[name]
        if dataset.shape == () or dataset.dtype.kind == "V":
            return dataset[()].tobytes()
        return dataset[...].tobytes()

    @classmethod
    def _save_parquet_h5(
        cls,
        group: h5py.Group,
        name: str,
        dataframe: pd.DataFrame,
        *,
        max_part_bytes: Optional[int] = None,
        initial_rows: int = 2_000_000,
    ) -> None:
        """Save a DataFrame to HDF5 as Parquet bytes, splitting if needed."""
        if max_part_bytes is None:
            max_part_bytes = cls._MAX_PART_BYTES

        if name in group:
            del group[name]

        i = 0
        while f"{name}__part_{i:03d}" in group:
            del group[f"{name}__part_{i:03d}"]
            i += 1

        for attr_name in (f"{name}__chunked", f"{name}__num_parts"):
            if attr_name in group.attrs:
                del group.attrs[attr_name]

        blob = cls._dump_parquet_to_bytes(dataframe)
        size = max(len(blob), cls._parquet_uncompressed_bytes(blob))

        if size <= max_part_bytes:
            cls._save_bytes_h5(group, name, blob)
            group.attrs[f"{name}__chunked"] = False
            group.attrs[f"{name}__num_parts"] = 1
            return

        n_rows = len(dataframe)
        rows_per_part = min(initial_rows, max(1, n_rows))
        parts: list[bytes] = []

        start = 0
        while start < n_rows:
            end = min(n_rows, start + rows_per_part)
            chunk = dataframe.iloc[start:end].reset_index(drop=True)

            chunk_blob = cls._dump_parquet_to_bytes(chunk)
            chunk_size = max(len(chunk_blob), cls._parquet_uncompressed_bytes(chunk_blob))

            if chunk_size > max_part_bytes:
                if rows_per_part == 1:
                    raise ValueError(
                        f"even one-row parquet chunk exceeds max_part_bytes={max_part_bytes}"
                    )
                rows_per_part = max(1, rows_per_part // 2)
                continue

            parts.append(chunk_blob)
            start = end

        for i, part_blob in enumerate(parts):
            cls._save_bytes_h5(group, f"{name}__part_{i:03d}", part_blob)

        group.attrs[f"{name}__chunked"] = True
        group.attrs[f"{name}__num_parts"] = len(parts)

    @classmethod
    def _load_parquet_h5(cls, group: h5py.Group, name: str) -> pd.DataFrame:
        """Load a DataFrame from HDF5 Parquet storage."""
        if bool(group.attrs.get(f"{name}__chunked", False)):
            num_parts = int(group.attrs.get(f"{name}__num_parts", 0))
            if num_parts <= 0:
                raise ValueError(f"invalid num_parts for '{name}' in group '{group.name}'")

            frames = []
            for i in range(num_parts):
                blob = cls._load_bytes_h5(group, f"{name}__part_{i:03d}")
                frames.append(cls._read_parquet_from_bytes(blob))
            return pd.concat(frames, ignore_index=True)

        blob = cls._load_bytes_h5(group, name)
        if len(blob) > cls._ARROW_BYTES_LIMIT:
            raise ValueError(
                f"'{name}' is stored as a single parquet blob larger than the Arrow limit"
            )
        return cls._read_parquet_from_bytes(blob)

    def _to_hdf5(
        self,
        path: str,
        save_view: bool = True,
        mode: Literal["w", "a"] = "w",
    ) -> None:
        """
        Save the dataset to an HDF5 file.

        Parameters
        ----------
        path : str
            Output file path.
        save_view : bool, default=True
            If ``True``, save the current view. If ``False``, save the underlying
            references as-is.
        mode : {"w", "a"}, default="w"
            HDF5 write mode.
        """
        if mode not in {"w", "a"}:
            raise ValueError("mode must be 'w' or 'a'")

        dataset = self.copy() if save_view else self

        with h5py.File(path, mode) as file:
            metadata_group = file["metadata"] if "metadata" in file else file.create_group("metadata")
            metadata_group.attrs["description"] = dataset.description
            metadata_group.attrs["attributes_json"] = json.dumps(dataset.attributes)
            metadata_group.attrs["tags_json"] = json.dumps(dataset.tags)

            existing_groups = [name for name in file.keys() if name.startswith("dataset_")]

            if mode == "a" and existing_groups:
                next_index = max(int(name.split("_")[1]) for name in existing_groups) + 1
                group = file.create_group(f"dataset_{next_index}")
            else:
                if "dataset_0" in file:
                    del file["dataset_0"]
                group = file.create_group("dataset_0")

            peaks_group = group.create_group("peaks")
            peaks_group.create_dataset("data", data=dataset._peak_series._data_ref)
            peaks_group.create_dataset("offsets", data=dataset._peak_series._offsets_ref)
            peaks_group.create_dataset("index", data=dataset._peak_series._index)

            if dataset._peak_series._metadata_ref is not None:
                self._save_parquet_h5(
                    peaks_group,
                    "metadata_parquet",
                    dataset._peak_series._metadata_ref,
                )

                string_dtype = h5py.string_dtype(encoding="utf-8")
                peaks_group.create_dataset(
                    "metadata_columns",
                    data=np.asarray(dataset._peak_series.metadata_columns, dtype=string_dtype),
                )

            group.attrs["columns_json"] = json.dumps(dataset._columns)
            self._save_parquet_h5(
                group,
                "spectrum_metadata_parquet",
                dataset._spectrum_metadata_ref,
            )

    @staticmethod
    def _from_hdf5(path: str, load_peak_metadata: bool = True) -> MSDataset:
        """
        Load a dataset from an HDF5 file.

        Parameters
        ----------
        path : str
            Input file path.
        load_peak_metadata : bool, default=True
            Whether to load peak-level metadata.

        Returns
        -------
        MSDataset
        """
        with h5py.File(path, "r") as file:
            description = ""
            attributes: Dict[str, str] = {}
            tags: list[str] = []

            if "metadata" in file:
                metadata_group = file["metadata"]
                description = str(metadata_group.attrs.get("description", ""))
                attributes = json.loads(metadata_group.attrs.get("attributes_json", "{}"))
                tags = json.loads(metadata_group.attrs.get("tags_json", "[]"))

            dataset_group_names = sorted(
                [name for name in file.keys() if name.startswith("dataset_")],
                key=lambda name: int(name.split("_")[1]),
            )
            if not dataset_group_names:
                raise ValueError(f"no dataset groups found in {path}")

            datasets: list[MSDataset] = []

            for group_name in dataset_group_names:
                group = file[group_name]
                peaks_group = group["peaks"]

                data = np.asarray(peaks_group["data"][:], dtype=np.float64)
                offsets = np.asarray(peaks_group["offsets"][:], dtype=np.int64)
                index = np.asarray(peaks_group["index"][:], dtype=np.int64)

                peak_metadata = None
                if load_peak_metadata:
                    if (
                        "metadata_parquet" in peaks_group
                        or bool(peaks_group.attrs.get("metadata_parquet__chunked", False))
                    ):
                        peak_metadata = MSDataset._load_parquet_h5(peaks_group, "metadata_parquet")

                metadata_columns = None
                if "metadata_columns" in peaks_group:
                    metadata_columns = [
                        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
                        for value in peaks_group["metadata_columns"][:]
                    ]

                if (
                    "spectrum_metadata_parquet" not in group
                    and not bool(group.attrs.get("spectrum_metadata_parquet__chunked", False))
                ):
                    raise KeyError(
                        f"missing 'spectrum_metadata_parquet' in group '{group.name}'"
                    )

                columns_json = group.attrs.get("columns_json")
                columns = None
                if columns_json:
                    loaded_columns = json.loads(columns_json)

                    if isinstance(loaded_columns, list):
                        columns = [str(column) for column in loaded_columns]

                spectrum_metadata = MSDataset._load_parquet_h5(
                    group,
                    "spectrum_metadata_parquet",
                )

                peak_series = PeakSeries(
                    data=data,
                    offsets=offsets,
                    metadata=peak_metadata,
                    metadata_columns=metadata_columns,
                    index=index,
                )

                datasets.append(
                    MSDataset(
                        spectrum_metadata=spectrum_metadata,
                        peak_series=peak_series,
                        columns=columns,
                        description=description,
                        attributes=attributes,
                        tags=tags,
                    )
                )

        return datasets[0] if len(datasets) == 1 else MSDataset.concat(datasets)



class SpectrumRecord:
    """
    Spectrum-level record in an :class:`MSDataset`.

    A :class:`SpectrumRecord` provides access to:

    - spectrum-level metadata
    - the corresponding :class:`Spectrum`

    Parameters
    ----------
    dataset : MSDataset
        Parent dataset.
    index : int
        Spectrum index within the current dataset view.
    """

    def __init__(self, dataset: MSDataset, index: int) -> None:
        self._dataset = dataset
        self._view_index = index
        self._source_index = int(dataset._peak_series._index[index])

    def __repr__(self) -> str:
        """Return a concise representation of the record."""
        content = ", ".join(f"{column}={self[column]!r}" for column in self.columns)
        return f"SpectrumRecord(n_peaks={self.n_peaks}, {content})"

    def __str__(self) -> str:
        """
        Return a human-readable representation of the record.

        Returns
        -------
        str
        """
        lines = []
        width = max((len(column) for column in self.columns), default=0)
        for column in self.columns:
            lines.append(f"{column:<{width}} : {self[column]}")
        lines.append("")
        lines.append(str(self.spectrum))
        return "\n".join(lines)

    def __contains__(self, key: str) -> bool:
        """
        Check whether a metadata column is available.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
        """
        return key in self.columns

    def _metadata_value(self, key: str) -> Any:
        if key not in self._dataset.columns:
            raise KeyError(f"column not available in current view: {key}")
        return self._dataset._spectrum_metadata_ref.iloc[self._source_index][key]

    def __getitem__(self, key: str) -> Any:
        """
        Access spectrum metadata by column name.

        Parameters
        ----------
        key : str

        Returns
        -------
        Any
        """
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        return self._metadata_value(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Add or update a spectrum metadata value for this record.

        Parameters
        ----------
        key : str
            Metadata column name.
        value : Any
            Metadata value.
        """
        if key not in self._dataset._spectrum_metadata_ref.columns:
            self._dataset._spectrum_metadata_ref[key] = np.nan

        self._dataset._spectrum_metadata_ref.iat[
            self._source_index,
            self._dataset._spectrum_metadata_ref.columns.get_loc(key),
        ] = value

        if key not in self._dataset._columns:
            self._dataset._columns.append(key)

    def __eq__(self, other: object) -> bool:
        """
        Compare two spectrum records.

        Parameters
        ----------
        other : object

        Returns
        -------
        bool
        """
        if not isinstance(other, SpectrumRecord):
            return False

        if set(self.columns) != set(other.columns):
            return False

        for column in self.columns:
            value1 = self[column]
            value2 = other[column]

            if pd.isna(value1) and pd.isna(value2):
                continue
            if value1 != value2:
                return False

        return self.spectrum == other.spectrum

    @property
    def columns(self) -> list[str]:
        """
        Metadata columns available in the current view.

        Returns
        -------
        list of str
        """
        return self._dataset.columns

    @property
    def n_peaks(self) -> int:
        """
        Number of peaks in the spectrum.

        Returns
        -------
        int
        """
        return len(self.spectrum)

    @property
    def spectrum(self) -> Spectrum:
        """
        Peak data of the record.

        Returns
        -------
        Spectrum
        """
        return self._dataset._peak_series[self._view_index]

    @property
    def peaks(self) -> Spectrum:
        """
        Alias for :attr:`spectrum`.

        Returns
        -------
        Spectrum
        """
        return self.spectrum

    @property
    def is_integer_mz(self) -> bool:
        """
        Check whether all m/z values are integers.

        Returns
        -------
        bool
        """
        mz = self.spectrum.mz
        return bool(np.all(np.mod(mz, 1) == 0))

    def normalize(self, scale: float = 1.0, in_place: bool = False) -> SpectrumRecord:
        """
        Normalize the intensities of the spectrum.

        Parameters
        ----------
        scale : float, default=1.0
            Target maximum intensity.
        in_place : bool, default=False
            If ``True``, modify the current record.

        Returns
        -------
        SpectrumRecord
        """
        if in_place:
            self.spectrum.normalize(scale=scale, in_place=True)
            return self

        copied = self.copy()
        copied.spectrum.normalize(scale=scale, in_place=True)
        return copied

    def sort_by_mz(self, ascending: bool = True, in_place: bool = False) -> SpectrumRecord:
        """
        Sort peaks by m/z.

        Parameters
        ----------
        ascending : bool, default=True
            Sort order.
        in_place : bool, default=False
            If ``True``, modify the current record.

        Returns
        -------
        SpectrumRecord
        """
        if in_place:
            self.spectrum.sort_by_mz(ascending=ascending, in_place=True)
            return self

        copied = self.copy()
        copied.spectrum.sort_by_mz(ascending=ascending, in_place=True)
        return copied

    def sort_by_intensity(
        self,
        ascending: bool = False,
        in_place: bool = False,
    ) -> SpectrumRecord:
        """
        Sort peaks by intensity.

        Parameters
        ----------
        ascending : bool, default=False
            Sort order.
        in_place : bool, default=False
            If ``True``, modify the current record.

        Returns
        -------
        SpectrumRecord
        """
        if in_place:
            self.spectrum.sort_by_intensity(ascending=ascending, in_place=True)
            return self

        copied = self.copy()
        copied.spectrum.sort_by_intensity(ascending=ascending, in_place=True)
        return copied

    def copy(self) -> SpectrumRecord:
        """
        Materialize this record as an independent object.

        Returns
        -------
        SpectrumRecord
        """
        spectrum_metadata = (
            self._dataset._spectrum_metadata_ref
            .iloc[[self._source_index]]
            .loc[:, self.columns]
            .reset_index(drop=True)
            .copy()
        )
        peak_series = self.spectrum.normalize(in_place=False)._peak_series

        dataset = MSDataset(
            spectrum_metadata=spectrum_metadata,
            peak_series=peak_series,
            columns=self.columns,
            description=self._dataset.description,
            attributes=self._dataset.attributes,
            tags=self._dataset.tags,
        )
        return SpectrumRecord(dataset, 0)

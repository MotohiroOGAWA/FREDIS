from typing import Any, Dict, Iterator


class Peak:
    """
    A single peak in a mass spectrum.

    A peak is defined by its mass-to-charge ratio (m/z), intensity,
    and optional metadata (e.g., precursor information or annotations).

    Parameters
    ----------
    mz : float
        Mass-to-charge ratio of the peak.
    intensity : float
        Intensity of the peak.
    metadata : dict of str to Any, optional
        Additional information associated with the peak.

    Notes
    -----
    - The object can be unpacked as ``(mz, intensity)``.
    - Metadata keys follow snake_case convention.
    """

    def __init__(
        self,
        mz: float,
        intensity: float,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self._mz: float = mz
        self._intensity: float = intensity
        self._metadata: Dict[str, Any] = metadata or {}

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"Peak(mz={self._mz}, intensity={self._intensity}, "
            f"metadata={self._metadata})"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        if not self._metadata:
            return f"m/z={self._mz}, intensity={self._intensity}"
        return (
            f"m/z={self._mz}, intensity={self._intensity}, "
            f"metadata={self._metadata}"
        )

    def __getitem__(self, key: str) -> Any:
        """
        Retrieve a metadata value.

        Parameters
        ----------
        key : str
            Metadata key.

        Returns
        -------
        Any
            Value associated with the given key.

        Raises
        ------
        KeyError
            If the key is not present.
        """
        return self._metadata[key]

    def __iter__(self) -> Iterator[float]:
        """
        Iterate over the peak as ``(mz, intensity)``.

        Yields
        ------
        float
            m/z and intensity values.

        Examples
        --------
        >>> peak = Peak(100.0, 200.0)
        >>> mz, intensity = peak
        """
        yield self._mz
        yield self._intensity

    @property
    def mz(self) -> float:
        """
        Mass-to-charge ratio.

        Returns
        -------
        float
        """
        return self._mz

    @property
    def intensity(self) -> float:
        """
        Peak intensity.

        Returns
        -------
        float
        """
        return self._intensity

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Metadata associated with the peak.

        Returns
        -------
        dict of str to Any
        """
        return self._metadata

    @property
    def neutral_loss(self) -> float:
        """
        Neutral loss of the peak.

        Defined as::

            PrecursorMZ - mz

        Returns
        -------
        float

        Raises
        ------
        KeyError
            If ``"PrecursorMZ"`` is not present in metadata.
        """
        if "PrecursorMZ" not in self._metadata:
            raise KeyError("metadata must contain 'PrecursorMZ'")
        return self._metadata["PrecursorMZ"] - self._mz
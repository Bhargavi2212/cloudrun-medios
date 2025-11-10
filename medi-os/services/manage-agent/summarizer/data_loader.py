"""
Utilities for loading patient-level events from the MEDS OMOP parquet export.

The data loader uses Polars lazy scans so we can efficiently filter down to a
single subject without loading the full 40M-row dataset into memory. Results
are cached with an LRU strategy to avoid redundant scans during a single run.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import polars as pl

from summarizer.errors import CodesMetadataError, DataLoadError, PatientNotFoundError

DEFAULT_DATA_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
    .joinpath("data", "meds_omop_ehrshot", "meds_omop_ehrshot")
)


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for the data loader."""

    data_glob: str = str(DEFAULT_DATA_ROOT.joinpath("data", "*.parquet"))
    codes_path: Path = DEFAULT_DATA_ROOT.joinpath("metadata", "codes.parquet")
    max_cache_entries: int = 32

    @classmethod
    def from_env(
        cls,
        *,
        data_glob: Optional[str] = None,
        codes_path: Optional[Path] = None,
        max_cache_entries: Optional[int] = None,
    ) -> "DataLoaderConfig":
        """
        Build a config instance using optional overrides. Environment variables
        will be integrated once the central config layer is introduced.
        """
        default_instance = cls()
        return cls(
            data_glob=data_glob or default_instance.data_glob,
            codes_path=codes_path or default_instance.codes_path,
            max_cache_entries=max_cache_entries or default_instance.max_cache_entries,
        )


class _LRUCache:
    """Simple LRU cache implementation for patient-level DataFrames."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._store: "OrderedDict[int, pl.DataFrame]" = OrderedDict()

    def get(self, key: int) -> Optional[pl.DataFrame]:
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value.clone()

    def put(self, key: int, value: pl.DataFrame) -> None:
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.maxsize:
            self._store.popitem(last=False)
        # store a copy to avoid future mutations leaking out
        self._store[key] = value.clone()

    def clear(self) -> None:
        self._store.clear()


class DataLoader:
    """Load patient events and metadata from the MEDS OMOP export."""

    def __init__(self, config: DataLoaderConfig | None = None) -> None:
        self.config = config or DataLoaderConfig()
        self._scan: pl.LazyFrame = pl.scan_parquet(self.config.data_glob)
        self._patient_cache = _LRUCache(self.config.max_cache_entries)
        self._codes_lookup: Optional[Dict[str, str]] = None

    def refresh_scan(self) -> None:
        """Rebuild the lazy scan (useful if new parquet shards are added)."""
        self._scan = pl.scan_parquet(self.config.data_glob)
        self._patient_cache.clear()

    def get_patient_events(self, subject_id: int | str) -> pl.DataFrame:
        """
        Return all events for a patient ordered by time.

        Raises:
            PatientNotFoundError: if the patient has no recorded events.
        """
        subject_id_int = int(subject_id)
        cached = self._patient_cache.get(subject_id_int)
        if cached is not None:
            return cached

        try:
            df = (
                self._scan.filter(pl.col("subject_id") == subject_id_int)
                .sort("time")
                .collect()
            )
        except pl.PolarsError as exc:  # pragma: no cover - polars internal error path
            raise DataLoadError(f"Failed to collect events: {exc}") from exc

        if df.height == 0:
            raise PatientNotFoundError(f"No events found for subject_id={subject_id_int}")

        self._patient_cache.put(subject_id_int, df)
        return df.clone()

    def list_patient_ids(self, limit: Optional[int] = None) -> Iterable[int]:
        """Yield patient IDs present in the dataset."""
        lazy_ids = (
            self._scan.select("subject_id")
            .unique()
            .sort("subject_id")
        )
        if limit is not None:
            lazy_ids = lazy_ids.limit(limit)
        result = lazy_ids.collect()
        return result["subject_id"].to_list()

    def _load_codes_lookup(self) -> None:
        if self._codes_lookup is not None:
            return
        codes_path = self.config.codes_path
        if not codes_path.exists():
            raise CodesMetadataError(f"Codes file not found at {codes_path}")
        table = pl.read_parquet(codes_path)
        if "code" not in table.columns:
            raise CodesMetadataError("`code` column missing from codes parquet")
        descriptions = (
            table.fill_null("")
            .select(["code", "description"])
            .to_dict(as_series=False)
        )
        self._codes_lookup = dict(zip(descriptions["code"], descriptions["description"]))

    def get_code_description(self, code: str) -> Optional[str]:
        """Map a code string to its human-readable description."""
        self._load_codes_lookup()
        assert self._codes_lookup is not None  # for type checkers
        return self._codes_lookup.get(code)


__all__ = ["DataLoader", "DataLoaderConfig"]


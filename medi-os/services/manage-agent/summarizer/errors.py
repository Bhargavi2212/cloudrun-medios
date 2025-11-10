"""
Custom exceptions for the summarizer service.
"""

from __future__ import annotations


class SummarizerError(Exception):
    """Base exception for summarizer-related failures."""


class ConfigurationError(SummarizerError):
    """Raised when configuration or environment variables are invalid."""


class DataLoadError(SummarizerError):
    """Raised when underlying data fails to load."""


class PatientNotFoundError(DataLoadError):
    """Raised when a patient is missing from the dataset."""


class CodesMetadataError(DataLoadError):
    """Raised when code description metadata is unavailable."""


class TrendComputationError(SummarizerError):
    """Raised when trend analysis cannot be completed."""


class LLMGenerationError(SummarizerError):
    """Raised when the language model fails to generate a summary."""


__all__ = [
    "SummarizerError",
    "ConfigurationError",
    "DataLoadError",
    "PatientNotFoundError",
    "CodesMetadataError",
    "TrendComputationError",
    "LLMGenerationError",
]


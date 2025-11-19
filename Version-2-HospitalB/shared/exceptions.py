"""
Custom exception hierarchy for Medi OS Version -2.
"""

from __future__ import annotations


class MediOsError(Exception):
    """
    Base exception for project-specific errors.
    """


class ConfigurationError(MediOsError):
    """
    Raised when configuration values are missing or invalid.
    """


class NotFoundError(MediOsError):
    """
    Raised when a requested resource cannot be located.
    """


class ValidationError(MediOsError):
    """
    Raised when input data fails validation rules.
    """

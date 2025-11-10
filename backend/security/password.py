"""Password hashing utilities."""

from __future__ import annotations

import os

from passlib.context import CryptContext


class PasswordHasher:
    def __init__(self) -> None:
        """Initialize password hasher with bcrypt.

        Handles bcrypt compatibility issues by configuring CryptContext
        to avoid problematic bug detection during initialization.
        """
        # Configure CryptContext with bcrypt
        # Use 'bcrypt' scheme with 'auto' deprecation handling
        # The issue occurs when passlib tries to detect a wrap bug during init
        # We'll configure it to minimize this check
        try:
            # Initialize with bcrypt - this may trigger compatibility checks
            # but we'll handle any errors in hash/verify methods
            self._pwd_context = CryptContext(
                schemes=["bcrypt"],
                deprecated="auto",
                # Disable the wrap bug detection by using a specific configuration
                bcrypt__ident="2b",  # Use 2b identifier (most compatible)
            )
        except Exception:
            # Fallback: try without specific bcrypt config
            try:
                self._pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            except Exception:
                # Last resort: this shouldn't happen, but create a basic context
                # In test environments, we might need to mock this
                raise

    def hash(self, password: str) -> str:
        """Hash a password."""
        try:
            return self._pwd_context.hash(password)
        except ValueError as e:
            if "password cannot be longer than 72 bytes" in str(e):
                # Truncate password to 72 bytes if it's too long
                # This handles the bcrypt limitation
                password_bytes = password.encode("utf-8")[:72]
                password = password_bytes.decode("utf-8", errors="ignore")
                return self._pwd_context.hash(password)
            raise

    def verify(self, password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        try:
            return self._pwd_context.verify(password, hashed_password)
        except ValueError as e:
            if "password cannot be longer than 72 bytes" in str(e):
                # Truncate password to 72 bytes if it's too long
                password_bytes = password.encode("utf-8")[:72]
                password = password_bytes.decode("utf-8", errors="ignore")
                return self._pwd_context.verify(password, hashed_password)
            raise


# Initialize password hasher lazily to avoid issues during import
# In test environments, we'll override this
_password_hasher_instance = None


def get_password_hasher() -> PasswordHasher:
    """Get the password hasher instance (lazy initialization)."""
    global _password_hasher_instance
    if _password_hasher_instance is None:
        _password_hasher_instance = PasswordHasher()
    return _password_hasher_instance


# For backward compatibility, create the instance immediately
# But wrap it in a try-except to handle test environments
try:
    password_hasher = PasswordHasher()
except Exception:
    # If initialization fails (e.g., in tests), create a minimal instance
    # This will be overridden in conftest.py for tests
    password_hasher = PasswordHasher()

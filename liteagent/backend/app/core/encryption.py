"""
Encryption service for securing sensitive data at rest.
Uses Fernet symmetric encryption for API keys and tokens.
"""
import base64
import os
import hashlib

from cryptography.fernet import Fernet, InvalidToken


class EncryptionService:
    """
    Service for encrypting and decrypting sensitive data.

    Uses Fernet symmetric encryption with a configurable key.
    If no key is provided, generates one (should be persisted in production).
    """

    # Prefix to identify encrypted values
    ENCRYPTED_PREFIX = "enc:v1:"

    def __init__(self, key: str | None = None):
        """
        Initialize the encryption service.

        Args:
            key: Encryption key (32+ bytes). If None, generates a new key.
        """
        if key:
            # Derive a proper Fernet key from the provided key
            # Use SHA256 to get consistent 32 bytes, then base64 encode
            key_bytes = hashlib.sha256(key.encode()).digest()
            self._key = base64.urlsafe_b64encode(key_bytes)
        else:
            # Generate a new key
            self._key = Fernet.generate_key()

        self._fernet = Fernet(self._key)

    def encrypt(self, plaintext: str | None) -> str | None:
        """
        Encrypt a plaintext string.

        Args:
            plaintext: The string to encrypt.

        Returns:
            Encrypted string with prefix, or None if input is None.
        """
        if plaintext is None:
            return None

        if not plaintext:
            return ""

        # Encrypt the plaintext
        encrypted_bytes = self._fernet.encrypt(plaintext.encode("utf-8"))
        encrypted_str = base64.urlsafe_b64encode(encrypted_bytes).decode("utf-8")

        # Add prefix to identify as encrypted
        return f"{self.ENCRYPTED_PREFIX}{encrypted_str}"

    def decrypt(self, ciphertext: str | None) -> str | None:
        """
        Decrypt an encrypted string.

        Args:
            ciphertext: The encrypted string (with or without prefix).

        Returns:
            Decrypted plaintext, or None if input is None.

        Raises:
            ValueError: If decryption fails (wrong key or corrupted data).
        """
        if ciphertext is None:
            return None

        if not ciphertext:
            return ""

        # Remove prefix if present
        if ciphertext.startswith(self.ENCRYPTED_PREFIX):
            ciphertext = ciphertext[len(self.ENCRYPTED_PREFIX) :]

        try:
            # Decode from base64 and decrypt
            encrypted_bytes = base64.urlsafe_b64decode(ciphertext.encode("utf-8"))
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode("utf-8")
        except (InvalidToken, Exception) as e:
            raise ValueError(f"Decryption failed: {e}") from e

    def is_encrypted(self, value: str | None) -> bool:
        """
        Check if a value appears to be encrypted.

        Args:
            value: The value to check.

        Returns:
            True if the value has the encrypted prefix.
        """
        if value is None:
            return False
        return value.startswith(self.ENCRYPTED_PREFIX)

    def get_key(self) -> bytes:
        """Get the encryption key (for persistence)."""
        return self._key


# Global instance (initialized with env var in production)
_encryption_service: EncryptionService | None = None


def get_encryption_service() -> EncryptionService:
    """Get the global encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        # In production, this should come from environment variable
        key = os.environ.get("LITEAGENT_ENCRYPTION_KEY")
        _encryption_service = EncryptionService(key=key)
    return _encryption_service

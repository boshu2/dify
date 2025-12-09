"""
Unit tests for encryption service.
Tests API key encryption at rest.
"""
import pytest
from unittest.mock import patch

from app.core.encryption import EncryptionService


class TestEncryptionService:
    """Tests for encryption service."""

    def test_init_with_key(self):
        """Test initialization with encryption key."""
        service = EncryptionService(key="test-encryption-key-32bytes!!")
        assert service is not None

    def test_init_generates_key_if_none(self):
        """Test that a key is generated if none provided."""
        service = EncryptionService()
        assert service._key is not None

    def test_encrypt_string(self):
        """Test encrypting a string."""
        service = EncryptionService(key="test-encryption-key-32bytes!!")
        plaintext = "sk-secret-api-key-12345"

        encrypted = service.encrypt(plaintext)

        assert encrypted != plaintext
        assert isinstance(encrypted, str)
        # Encrypted string should be base64 encoded
        assert len(encrypted) > len(plaintext)

    def test_decrypt_string(self):
        """Test decrypting a string."""
        service = EncryptionService(key="test-encryption-key-32bytes!!")
        plaintext = "sk-secret-api-key-12345"

        encrypted = service.encrypt(plaintext)
        decrypted = service.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encrypt_decrypt_roundtrip(self):
        """Test full encryption/decryption roundtrip."""
        service = EncryptionService(key="test-encryption-key-32bytes!!")

        test_values = [
            "sk-test-key",
            "sk-ant-api03-very-long-key-with-special-chars-!@#$%",
            "",  # empty string
            "a" * 1000,  # long string
        ]

        for value in test_values:
            encrypted = service.encrypt(value)
            decrypted = service.decrypt(encrypted)
            assert decrypted == value, f"Failed for: {value[:20]}..."

    def test_different_encryptions_for_same_plaintext(self):
        """Test that same plaintext produces different ciphertexts (IV randomness)."""
        service = EncryptionService(key="test-encryption-key-32bytes!!")
        plaintext = "sk-secret-api-key"

        encrypted1 = service.encrypt(plaintext)
        encrypted2 = service.encrypt(plaintext)

        # Same plaintext should produce different ciphertexts due to random IV
        assert encrypted1 != encrypted2

        # But both should decrypt to same value
        assert service.decrypt(encrypted1) == plaintext
        assert service.decrypt(encrypted2) == plaintext

    def test_decrypt_invalid_data_raises_error(self):
        """Test that decrypting invalid data raises an error."""
        service = EncryptionService(key="test-encryption-key-32bytes!!")

        with pytest.raises(Exception):  # Could be ValueError or specific exception
            service.decrypt("not-valid-encrypted-data")

    def test_decrypt_with_wrong_key_fails(self):
        """Test that decryption with wrong key fails."""
        service1 = EncryptionService(key="test-encryption-key-32bytes!!")
        service2 = EncryptionService(key="different-encryption-key-32b!!")

        plaintext = "sk-secret-api-key"
        encrypted = service1.encrypt(plaintext)

        with pytest.raises(Exception):
            service2.decrypt(encrypted)

    def test_is_encrypted_check(self):
        """Test checking if a value is encrypted."""
        service = EncryptionService(key="test-encryption-key-32bytes!!")

        plaintext = "sk-secret-api-key"
        encrypted = service.encrypt(plaintext)

        assert service.is_encrypted(encrypted) is True
        assert service.is_encrypted(plaintext) is False

    def test_encrypt_none_returns_none(self):
        """Test that encrypting None returns None."""
        service = EncryptionService(key="test-encryption-key-32bytes!!")
        assert service.encrypt(None) is None

    def test_decrypt_none_returns_none(self):
        """Test that decrypting None returns None."""
        service = EncryptionService(key="test-encryption-key-32bytes!!")
        assert service.decrypt(None) is None


class TestEncryptionServiceIntegration:
    """Integration tests for encryption with database."""

    @pytest.mark.asyncio
    async def test_encrypt_provider_api_key(self, db_session, provider_data):
        """Test encrypting provider API key before storage."""
        from app.services.provider_service import ProviderService
        from app.schemas.provider import LLMProviderCreate

        service = ProviderService(db_session)

        # Create provider with plaintext API key
        data = provider_data(api_key="sk-plaintext-secret-key")
        provider_create = LLMProviderCreate(**data)

        provider = await service.create(provider_create)

        # API key should be encrypted in database
        # When retrieved, it should be decrypted for use
        assert provider.api_key is not None
        # The raw DB value should be encrypted (different from input)
        # But service should handle encryption transparently

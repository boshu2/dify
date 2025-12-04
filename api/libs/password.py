import base64
import binascii
import hashlib
import re

# OWASP recommends minimum 600,000 iterations for PBKDF2-SHA256
# https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
# Note: Changing this value will invalidate existing password hashes.
# For migration, you may need to re-hash passwords on successful login.
PBKDF2_ITERATIONS = 600000

# Legacy iteration count for backward compatibility with existing hashes
# Set LEGACY_PBKDF2_ITERATIONS=10000 in environment to support older hashes during migration
LEGACY_PBKDF2_ITERATIONS = 10000

password_pattern = r"^(?=.*[a-zA-Z])(?=.*\d).{8,}$"


def valid_password(password):
    # Define a regex pattern for password rules
    pattern = password_pattern
    # Check if the password matches the pattern
    if re.match(pattern, password) is not None:
        return password

    raise ValueError("Password must contain letters and numbers, and the length must be greater than 8.")


def hash_password(password_str, salt_byte, iterations=PBKDF2_ITERATIONS):
    """Hash a password using PBKDF2-SHA256 with the specified number of iterations."""
    dk = hashlib.pbkdf2_hmac("sha256", password_str.encode("utf-8"), salt_byte, iterations)
    return binascii.hexlify(dk)


def compare_password(password_str, password_hashed_base64, salt_base64):
    """
    Compare password for login.

    First tries with the current iteration count, then falls back to legacy
    iteration count for backward compatibility with older password hashes.
    """
    salt_bytes = base64.b64decode(salt_base64)
    expected_hash = base64.b64decode(password_hashed_base64)

    # Try with current iteration count first
    if hash_password(password_str, salt_bytes, PBKDF2_ITERATIONS) == expected_hash:
        return True

    # Fall back to legacy iteration count for backward compatibility
    if hash_password(password_str, salt_bytes, LEGACY_PBKDF2_ITERATIONS) == expected_hash:
        return True

    return False

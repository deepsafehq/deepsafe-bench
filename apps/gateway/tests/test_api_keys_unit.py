"""Unit tests for api_keys.py: generation, hashing, and verification."""

from api_keys import KEY_PREFIX, generate_api_key, hash_key, verify_key


def test_key_prefix_constant():
    """KEY_PREFIX must equal 'ds_live_'."""
    assert KEY_PREFIX == "ds_live_"


def test_generate_api_key_starts_with_prefix():
    """Generated key must start with KEY_PREFIX."""
    key = generate_api_key()
    assert key.startswith("ds_live_")


def test_generate_api_key_total_length():
    """Generated key must be exactly 40 characters (8-char prefix + 32 hex chars)."""
    key = generate_api_key()
    assert len(key) == 40


def test_generate_api_key_uniqueness():
    """Two generated keys must be different."""
    key1 = generate_api_key()
    key2 = generate_api_key()
    assert key1 != key2


def test_hash_key_returns_64_char_hex():
    """hash_key must return a 64-character hex string (SHA-256 digest)."""
    key = generate_api_key()
    digest = hash_key(key)
    assert len(digest) == 64
    # Confirm it's a valid hex string
    int(digest, 16)


def test_hash_key_deterministic():
    """Same input must always produce the same hash."""
    key = "ds_live_" + "0123456789abcdef" * 2  # synthetic fixture
    assert hash_key(key) == hash_key(key)


def test_hash_key_different_inputs_produce_different_hashes():
    """Different keys must produce different hashes."""
    key1 = "ds_live_" + "a" * 32
    key2 = "ds_live_" + "b" * 32
    assert hash_key(key1) != hash_key(key2)


def test_verify_key_returns_true_for_correct_key():
    """verify_key must return True when the plaintext matches the stored hash."""
    key = generate_api_key()
    stored = hash_key(key)
    assert verify_key(key, stored) is True


def test_verify_key_returns_false_for_wrong_key():
    """verify_key must return False when the plaintext does not match."""
    key = generate_api_key()
    stored = hash_key(key)
    wrong_key = "ds_live_" + "0" * 32
    assert verify_key(wrong_key, stored) is False

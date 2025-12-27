import pytest
from app.logic import hash_airport

def test_hash_airport_consistency():
    # Same input must always return same bucket
    assert hash_airport("IST") == hash_airport("IST")

def test_hash_airport_range():
    # Result must be between 0-99 (100 buckets)
    bucket = hash_airport("JFK", buckets=100)
    assert 0 <= bucket < 100

def test_hash_airport_default():
    # Test with default buckets (10)
    bucket = hash_airport("LHR")
    assert 0 <= bucket < 10

def test_hash_airport_empty():
    # Test empty string behavior
    assert hash_airport("") == 0

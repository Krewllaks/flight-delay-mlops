import pytest
from app.logic import hash_airport

def test_hash_airport_consistency():
    assert hash_airport("IST") == hash_airport("IST")

def test_hash_airport_range():
    bucket = hash_airport("JFK", buckets=100)
    assert 0 <= bucket < 100

def test_hash_airport_default():
    bucket = hash_airport("LHR")
    assert 0 <= bucket < 10

def test_hash_airport_empty():
    assert hash_airport("") == 0

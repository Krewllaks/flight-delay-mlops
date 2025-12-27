import hashlib

def hash_airport(airport_code: str, buckets: int = 10) -> int:
    if not airport_code:
        return 0
    
    hash_object = hashlib.md5(airport_code.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return hash_int % buckets

def bucket_delay(delay_minutes: float) -> str:
    if delay_minutes < 15:
        return "on_time"
    elif delay_minutes < 45:
        return "short_delay"
    return "long_delay"

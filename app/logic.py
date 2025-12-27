import hashlib

def hash_airport(airport_code: str, buckets: int = 10) -> int:
    """
    Hashes an airport code string to an integer bucket index.
    
    Args:
        airport_code (str): The airport code (e.g., 'IST').
        buckets (int): The number of buckets (default 10).
    
    Returns:
        int: The bucket index (0 to buckets-1).
    """
    if not airport_code:
        return 0  # Fallback for empty/None
    
    # MD5 hash of the string
    hash_object = hashlib.md5(airport_code.encode())
    # Convert hex digest to integer
    hash_int = int(hash_object.hexdigest(), 16)
    # Modulo to get bucket
    return hash_int % buckets

def bucket_delay(delay_minutes: float) -> str:
    """
    Buckets delay minutes into categories.
    
    Args:
        delay_minutes (float): The delay in minutes.
    
    Returns:
        str: Category ('on_time', 'short_delay', 'long_delay').
    """
    if delay_minutes < 15:
        return "on_time"
    elif delay_minutes < 45:
        return "short_delay"
    return "999"

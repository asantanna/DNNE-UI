"""
Time parsing utilities for DNNE framework.
Shared utilities for parsing duration strings like "10s", "5m", "2m30s", etc.
"""

import re
from typing import Union


def parse_duration(duration: Union[str, int, float]) -> float:
    """
    Parse a duration string into seconds.
    
    Supports formats:
    - Integer/float: Direct seconds (e.g., 60, 30.5)
    - Time strings: "10s", "5m", "2h", "1h30m", "2m30s", "1h5m10s"
    
    Args:
        duration: Duration as string, int, or float
        
    Returns:
        Duration in seconds as float
        
    Raises:
        ValueError: If duration format is invalid
        
    Examples:
        >>> parse_duration(60)
        60.0
        >>> parse_duration("30s")
        30.0
        >>> parse_duration("5m")
        300.0
        >>> parse_duration("2m30s")
        150.0
        >>> parse_duration("1h5m10s")
        3910.0
    """
    # Handle numeric input
    if isinstance(duration, (int, float)):
        if duration < 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        return float(duration)
    
    # Handle string input
    if not isinstance(duration, str):
        raise ValueError(f"Duration must be string, int, or float, got {type(duration)}")
    
    duration = duration.strip()
    if not duration:
        raise ValueError("Duration string cannot be empty")
    
    # Try to parse as plain number
    try:
        seconds = float(duration)
        if seconds < 0:
            raise ValueError(f"Duration must be positive, got {seconds}")
        return seconds
    except ValueError:
        pass  # Not a plain number, continue with pattern matching
    
    # Parse time format (e.g., "1h30m", "2m30s", "45s")
    pattern = r'^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?$'
    match = re.match(pattern, duration.lower())
    
    if not match:
        raise ValueError(f"Invalid duration format: '{duration}'. "
                        "Expected formats: '30', '30s', '5m', '2h', '1h30m', '2m30s', etc.")
    
    hours, minutes, seconds = match.groups()
    
    # Check that at least one component was specified
    if not any([hours, minutes, seconds]):
        raise ValueError(f"Invalid duration format: '{duration}'. No time components found.")
    
    total_seconds = 0.0
    
    if hours:
        total_seconds += int(hours) * 3600
    if minutes:
        total_seconds += int(minutes) * 60
    if seconds:
        total_seconds += float(seconds)
    
    if total_seconds <= 0:
        raise ValueError(f"Duration must be positive, got {total_seconds} seconds from '{duration}'")
    
    return total_seconds


def format_duration(seconds: float, compact: bool = False) -> str:
    """
    Format seconds into a human-readable duration string.
    
    Args:
        seconds: Duration in seconds
        compact: If True, use compact format (e.g., "2m30s"), else verbose (e.g., "2 minutes 30 seconds")
        
    Returns:
        Formatted duration string
        
    Examples:
        >>> format_duration(90, compact=True)
        "1m30s"
        >>> format_duration(3661, compact=True)
        "1h1m1s"
        >>> format_duration(30, compact=False)
        "30 seconds"
    """
    if seconds < 0:
        raise ValueError(f"Duration must be positive, got {seconds}")
    
    # Round to avoid floating point issues
    total_seconds = round(seconds)
    
    if total_seconds == 0:
        return "0s" if compact else "0 seconds"
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    parts = []
    
    if compact:
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:  # Always show seconds if no other parts
            parts.append(f"{secs}s")
        return "".join(parts)
    else:
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs > 0 or not parts:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")
        
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} {parts[1]}"
        else:
            return f"{parts[0]} {parts[1]} {parts[2]}"
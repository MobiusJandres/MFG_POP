"""
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

time.py: Provides utility functions for generating timestamps.
"""
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

def fancy_timestamp(tz: str = 'UTC', time_sep: str = '-', include_seconds: bool = False) -> str:
    """
    Generate a human-readable timestamp string in a specific timezone.
    
    Args:
        tz_str (str): The timezone for the timestamp (e.g., 'America/Los_Angeles', 'UTC').
        time_sep (str): The separator to use between time components.
        include_seconds (bool): Whether to include seconds in the timestamp.
    
    Returns:
        str: The formatted timestamp string.
    """
    try:
        tz = ZoneInfo(tz)
    except ZoneInfoNotFoundError:
        # Fallback to UTC if the timezone string is invalid
        tz = ZoneInfo("UTC")
        
    time_format = f'%Y-%m-%d %H{time_sep}%M'
    if include_seconds:
        time_format += f'{time_sep}%S'
    return datetime.now(tz).strftime(time_format)

def numeric_timestamp(tz: str = 'UTC', include_seconds: bool = False) -> str:
    """
    Generate a purely numerical timestamp string in a specific timezone.
    
    Args:
        tz_str (str): The timezone for the timestamp (e.g., 'America/Los_Angeles', 'UTC').
        include_seconds (bool): Whether to include seconds in the timestamp.
    
    Returns:
        str: The formatted numerical timestamp string.
    """
    try:
        tz = ZoneInfo(tz)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
        
    time_format = '%Y%m%d%H%M'
    if include_seconds:
        time_format += '%S'
    return datetime.now(tz).strftime(time_format)

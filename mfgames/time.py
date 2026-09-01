"""
Copyright (c) 2025 Matrix Research, Inc
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

time.py: Timestamp utilities for file naming and logging.

This module provides timezone-aware timestamp generation functions for use in
output file naming, logging, and timestamping simulation artifacts. All functions
support arbitrary timezones via IANA timezone database strings.

The module provides two timestamp formats:
    - fancy_timestamp: Human-readable format (e.g., "2026-08-26 14-30-45")
    - numeric_timestamp: Compact numerical format (e.g., "20260826143045")

Both formats are designed to be filesystem-safe (no colons in time component)
and lexicographically sortable (ISO 8601 date ordering).

Example usage:
    >>> from mfgames.time import fancy_timestamp, numeric_timestamp
    >>> fancy_timestamp(tz='America/New_York', include_seconds=True)
    '2026-08-26 10-30-45'
    >>> numeric_timestamp(tz='UTC', include_seconds=False)
    '202608261430'

Authors:
    * Johnathan Andres, Mobius Logic
    * Christina Cole, Matrix Research
    * Joel Klipfel, Matrix Research
Companies: [Mobius Logic](https://www.mobiuslogic.com), [Matrix Research, Inc](matrixresearch.com)
Contact: [Joel Klipfel](mailto:joel.klipfel@matrixresearch.com)
"""
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

def fancy_timestamp(tz: str = 'UTC', time_sep: str = '-', include_seconds: bool = False) -> str:
    """
    Generate a human-readable, filesystem-safe timestamp string.

    Creates a timestamp in ISO 8601 date format with customizable time separator
    for use in filenames. Default format uses hyphens instead of colons to ensure
    cross-platform filesystem compatibility (colons are problematic on Windows).

    The timestamp is timezone-aware and uses the IANA timezone database for
    accurate conversion. If an invalid timezone is provided, falls back silently
    to UTC to ensure function robustness.

    Format examples:
        - Without seconds: "2026-08-26 14-30"
        - With seconds:    "2026-08-26 14-30-45"

    Args:
        tz (str): IANA timezone string (e.g., 'America/Los_Angeles', 'Europe/London', 'UTC').
                  Defaults to 'UTC'. Invalid timezones fall back to UTC.
        time_sep (str): Separator character between time components (hours, minutes, seconds).
                        Defaults to '-' for filesystem safety. Use ':' for ISO 8601 compliance.
        include_seconds (bool): If True, includes seconds in the output timestamp.
                                Defaults to False for brevity in filenames.

    Returns:
        str: Formatted timestamp string in the pattern "YYYY-MM-DD HH{sep}MM[{sep}SS]"
             where {sep} is the specified time_sep character.

    Example:
        >>> fancy_timestamp(tz='America/New_York', time_sep='-', include_seconds=True)
        '2026-08-26 10-30-45'
        >>> fancy_timestamp()  # UTC, no seconds
        '2026-08-26 14-30'
    """
    try:
        # Attempt to parse the IANA timezone string
        tz = ZoneInfo(tz)
    except ZoneInfoNotFoundError:
        # Fallback to UTC if the timezone string is invalid or not found
        # This ensures the function is robust to user input errors
        tz = ZoneInfo("UTC")


    # Build the strftime format string dynamically based on options
    # ISO 8601 date (YYYY-MM-DD), then custom time separator for HH, MM, [SS]
    time_format = f'%Y-%m-%d %H{time_sep}%M'
    if include_seconds:
        time_format += f'{time_sep}%S'

    # Get current time in the specified timezone and format it
    return datetime.now(tz).strftime(time_format)

def numeric_timestamp(tz: str = 'UTC', include_seconds: bool = False) -> str:
    """
    Generate a compact, purely numerical timestamp string.

    Creates a timezone-aware timestamp with no separators or spaces, optimized
    for use in machine-readable filenames, database keys, or compact logging.
    The format is lexicographically sortable (chronological order matches
    alphabetical order).

    Unlike fancy_timestamp(), this function produces no delimiters between
    date/time components, resulting in a maximally compact representation
    suitable for automated systems.

    Format examples:
        - Without seconds: "202608261430" (12 digits)
        - With seconds:    "20260826143045" (14 digits)

    Args:
        tz (str): IANA timezone string (e.g., 'America/Los_Angeles', 'Europe/London', 'UTC').
                  Defaults to 'UTC'. Invalid timezones fall back to UTC.
        include_seconds (bool): If True, includes seconds in the output timestamp.
                                Defaults to False for maximum compactness.

    Returns:
        str: Purely numerical timestamp string in the pattern "YYYYMMDDHHMM[SS]"
             with no separators.

    Example:
        >>> numeric_timestamp(tz='America/New_York', include_seconds=True)
        '20260826103045'
        >>> numeric_timestamp()  # UTC, no seconds
        '202608261430'
    """
    try:
        # Attempt to parse the IANA timezone string
        tz = ZoneInfo(tz)
    except ZoneInfoNotFoundError:
        # Fallback to UTC if the timezone string is invalid or not found
        # This ensures the function is robust to user input errors
        tz = ZoneInfo("UTC")

    # Build the strftime format string for purely numerical output
    # No separators: YYYYMMDDHHMM[SS]
    time_format = '%Y%m%d%H%M'
    if include_seconds:
        time_format += '%S'

    # Get current time in the specified timezone and format it
    return datetime.now(tz).strftime(time_format)

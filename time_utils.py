"""
Global Time Handler for Mini Simon
Enforces a strict Indian Standard Time (IST - Asia/Kolkata) format across all modules.
Acts as the Single Source of Truth for all time-related operations.
"""

from datetime import datetime
import zoneinfo
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Constants
IST_TZ = zoneinfo.ZoneInfo('Asia/Kolkata')
STD_TIME_FORMAT = "%d-%b-%Y %I:%M %p"

class GlobalTimeHandler:
    @staticmethod
    def now_ist() -> datetime:
        """Returns the current aware datetime strictly in IST."""
        return datetime.now(IST_TZ)

    @staticmethod
    def now_ist_str() -> str:
        """Returns the current datetime as a standardized IST string."""
        return GlobalTimeHandler.now_ist().strftime(STD_TIME_FORMAT)

    @staticmethod
    def format_ist(dt: datetime) -> str:
        """Converts any datetime (naive or aware) to our standard IST string format."""
        if dt.tzinfo is None:
            # Assume naive datetimes are already in IST or local time,
            # but we explicitly attach the IST timezone to be safe.
            dt = dt.replace(tzinfo=IST_TZ)
        else:
            # Convert aware datetimes (like UTC) to IST.
            dt = dt.astimezone(IST_TZ)
            
        return dt.strftime(STD_TIME_FORMAT)

    @staticmethod
    def parse_to_ist(time_str: str, fmt: Optional[str] = None) -> datetime:
        """
        Parses a time string from API/CSV (like ISO 8601 or custom format)
        and forces it to an aware IST datetime.
        Great for unifying historical CSV/API data during Backtesting.
        """
        try:
            if fmt:
                dt = datetime.strptime(time_str, fmt)
            else:
                # Try ISO format by default
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                
            if dt.tzinfo is None:
                # Naive time strings treated as IST by default since it's the Indian market
                dt = dt.replace(tzinfo=IST_TZ)
            else:
                dt = dt.astimezone(IST_TZ)
                
            return dt
        except Exception as e:
            logger.error(f"Failed to parse time string '{time_str}' to IST: {e}")
            raise ValueError(f"Failed to parse time string '{time_str}': {e}")

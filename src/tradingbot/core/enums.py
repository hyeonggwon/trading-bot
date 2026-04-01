from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"


class SignalType(str, Enum):
    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"


class PositionSide(str, Enum):
    LONG = "long"
    FLAT = "flat"

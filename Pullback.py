import LabelPullback
from datetime import datetime


class Pullback:
    index_start: int
    index_end: int
    start_price: float
    high_price: float
    total_volume: int
    ohlc_total_size: int
    bullish_ohlc_size: int
    bullish_volume: int
    retracement_volume: int
    retracement_percentage: float
    retracement_ohlc_size: int
    retracement_low_price: float
    retracement_date_start: datetime
    label: LabelPullback

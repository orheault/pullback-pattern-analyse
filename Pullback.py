import LabelPullback


class Pullback:
    label: LabelPullback
    ohlc_total_size: int
    total_volume: int
    ohlc_bullish_size: int
    bullish_volume: int
    ohlc_retracement_size: int
    retracement_volume: int
    retracement_percentage: float
    start_price: float
    high_price: float
    retracement_low_price: float

    def set_label(self, label):
        self.label = label

    def set_ohlc_total_size(self, size):
        self.ohlc_total_size = size

    def set_total_volume(self, volume):
        self.total_volume = volume

    def set_ohlc_bullish_size(self, volume):
        self.ohlc_bullish_size = volume

    def set_bullish_volume(self, volume):
        self.bullish_volume = volume

    def set_ohlc_retracement_size(self, size):
        self.ohlc_retracement_size = size

    def set_retracement_volume(self, volume):
        self.retracement_volume = volume

    def set_retracement_percentage(self, percentage):
        self.retracement_percentage = percentage

    def set_start_price(self, price):
        self.start_price = price

    def set_high_price(self, price):
        self.high_price = price

    def set_retracement_low_price(self, price):
        self.retracement_low_price = price

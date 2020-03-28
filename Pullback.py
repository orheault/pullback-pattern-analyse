import LabelPullback


class Pullback:
    data_ohlc: object
    data_breakout: []
    label: LabelPullback

    def __init__(self, data_ohlc, data_breakout, label):
        self.data_ohlc = data_ohlc
        self.data_breakout = data_breakout
        self.label = label

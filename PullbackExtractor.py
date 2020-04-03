from LabelPullback import LabelPullback
import numpy as np
from Pullback import Pullback


class PullbackExtractor:
    def extract(self, data_bid, data_volume):
        # SEARCH FOR PULLBACK PATTERN
        # THIS ALGO SEARCH BACKWARD, A BIT WEIRD
        # Time move forward, for each candle, look backward if pattern is form.
        pullback_datas = []
        index_breakout = 0
        i = 0
        real_i = 0
        for index, row in data_bid.iterrows():
            i = real_i
            real_i += 1

            current_candle = row
            if i - 1 < 0:
                continue
            previous_candle = data_bid.iloc[i - 1]

            # 1: Current candle positive
            current_candle_positive = current_candle['open'] - current_candle['close'] <= 0
            if not current_candle_positive:
                continue

            # 2: Previous candle negative
            previous_candle_negative = previous_candle['open'] - previous_candle['close'] >= 0
            if not previous_candle_negative:
                continue

            # 3: Current candle higher than previous candle
            current_candle_higher_than_previous = current_candle['close'] >= previous_candle['open']
            if not current_candle_higher_than_previous:
                continue

            # 4; Previous candle make a local low.
            # Compare with the candle before since I just check if the next candle id higher
            previous_candle_local_low = previous_candle['open'] <= data_bid.iloc[i - 2]['close']
            if not previous_candle_local_low:
                continue

            # 5: Sequence of previous candle negative
            index_retracement_end = i - 1
            index_retracement_start = -1
            index_retracement = index_retracement_end - 1
            is_searching_retracement = True
            while is_searching_retracement:
                # Get previous candle
                retracement_candle = data_bid.iloc[index_retracement]
                # Check if candle is negative
                retracement_candle_is_negative = retracement_candle['open'] - retracement_candle['close'] > 0
                if not retracement_candle_is_negative:
                    # todo can i transform the while with a while True and delete the isSearchingRetracement variable???
                    break
                # And higher then the next candle
                next_retracement_candle = data_bid.iloc[index_retracement_end + 1]
                retracement_candle_is_higher_than_next_candle = \
                    retracement_candle['close'] >= next_retracement_candle['open']
                if not retracement_candle_is_higher_than_next_candle:
                    break

                index_retracement_start = index_retracement
                index_retracement -= 1

            if index_retracement_start == -1:
                continue

            i = index_retracement_start - 1

            # 6: Candle is positive with local high
            candle_local_high = data_bid.iloc[i]
            candle_is_local_high = candle_local_high['close'] >= data_bid.iloc[i - 1]['close']
            if not candle_is_local_high:
                continue
            candle_local_high_is_positive = candle_local_high['open'] - candle_local_high['close'] < 0
            if not candle_local_high_is_positive:
                continue

            i = i - 1

            # 7 Sequence of previous candle positive
            index_bull_sequence_end = i
            index_bull_sequence_start = -1
            index_bull_sequence = i
            is_searching_bull = True
            while is_searching_bull:
                candle_bull = data_bid.iloc[index_bull_sequence]

                # Candle is positive
                candle_is_positive = candle_bull['open'] - candle_bull['close'] < 0
                if not candle_is_positive:
                    is_searching_bull = False
                    break

                # Bullish movement
                # isBullishMovement = candleBull['close'] < data.iloc[indexBullSequence + 1]
                # if isBullishMovement == False:
                #    isSearchingBull = False
                #    break

                index_bull_sequence_start = index_bull_sequence
                index_bull_sequence -= 1

            if index_bull_sequence_start == -1:
                continue

            # Retracement does not exceed 100% of pullback
            if data_bid.iloc[index_retracement_end]['close'] < data_bid.iloc[index_bull_sequence_start]['low']:
                continue

            # 9 - Breakout define by target price higher than the pullback high
            index_breakout = real_i
            candle_breakout_is_greater_than_high_pullback = False
            label = LabelPullback.FAIL
            while True:
                candle_breakout = data_bid.iloc[index_breakout]
                if candle_breakout['open'] - candle_breakout['close'] > 0:
                    index_breakout -= 1
                    break

                candle_breakout_is_greater_than_high_pullback = candle_breakout['close'] >= \
                                                                data_bid.iloc[index_retracement_start - 1]['close']
                if candle_breakout_is_greater_than_high_pullback:
                    label = LabelPullback.SUCCESSFUL

                index_breakout += 1

            # Found a pullback pattern!
            pullback_data_bid = data_bid[index_bull_sequence_start:index_retracement_end + 3]
            pullback_data_volume = data_volume[index_bull_sequence_start:index_retracement_end + 3]

            pullback = Pullback()
            pullback.set_label(label)
            pullback.set_ohlc_total_size(pullback_data_bid.size)
            pullback.set_total_volume(pullback_data_volume["bidVolume"].sum())
            pullback.set_ohlc_bullish_size(index_bull_sequence_end - index_bull_sequence_start)

            bullish_volume_data = pullback_data_volume[0:index_bull_sequence_end - index_bull_sequence_start]
            if bullish_volume_data.size > 0:
                bullish_volume_value = bullish_volume_data["bidVolume"].sum()
            else:
                bullish_volume_value = 0
            pullback.set_bullish_volume(bullish_volume_value)

            pullback.set_ohlc_retracement_size(pullback_data_bid[index_retracement_start:index_retracement_end].size)
            pullback.set_retracement_volume(
                pullback_data_volume[index_retracement_start - index_bull_sequence_start:index_retracement_end][
                    "bidVolume"].sum())
            pullback.set_retracement_percentage(abs(
                (pullback_data_bid.iloc[0]["high"] -
                 pullback_data_bid.iloc[index_retracement_end - index_retracement_start]["high"]) / (
                        pullback_data_bid.iloc[index_bull_sequence_end - index_retracement_start]["high"] -
                        pullback_data_bid.iloc[0]["low"])))
            pullback.set_retracement_low_price(pullback_data_bid.iloc[index_retracement_end- index_retracement_start]["low"])
            pullback.set_start_price(pullback_data_bid.iloc[0]["low"])
            pullback.set_high_price(pullback_data_bid.iloc[index_bull_sequence_end- index_retracement_start]['high'])
            pullback_datas.append(pullback)

        return pullback_datas

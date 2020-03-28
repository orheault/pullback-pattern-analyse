import enum
import numpy as np


class LabelPullbackId(enum.Enum):
    SUCCESSFUL = 0
    FAIL = 1


class PullbackExtractor:
    def extract(self, data):
        # SEARCH FOR PULLBACK PATTERN
        # THIS ALGO SEARCH BACKWARD, A BIT WEIRD
        # Time move forward, for each candle, look backward if pattern is form.
        pullback_datas = []
        index_breakout = 0
        i = 0
        real_i = 0
        for index, row in data.iterrows():
            i = real_i
            real_i += 1

            current_candle = row
            if i - 1 < 0:
                continue
            previous_candle = data.iloc[i - 1]

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
            previous_candle_local_low = previous_candle['open'] <= data.iloc[i - 2]['close']
            if not previous_candle_local_low:
                continue

            # 5: Sequence of previous candle negative
            index_retracement_end = i - 1
            index_retracement_start = -1
            index_retracement = index_retracement_end - 1
            is_searching_retracement = True
            while is_searching_retracement:
                # Get previous candle
                retracement_candle = data.iloc[index_retracement]
                # Check if candle is negative
                retracement_candle_is_negative = retracement_candle['open'] - retracement_candle['close'] > 0
                if not retracement_candle_is_negative:
                    # todo can i transform the while with a while True and delete the isSearchingRetracement variable???
                    is_searching_retracement = False
                    break
                # And higher then the next candle
                next_retracement_candle = data.iloc[index_retracement_end + 1]
                retracement_candle_is_higher_than_next_candle = \
                    retracement_candle['close'] >= next_retracement_candle['open']
                if not retracement_candle_is_higher_than_next_candle:
                    is_searching_retracement = False
                    break

                index_retracement_start = index_retracement
                index_retracement -= 1

            if index_retracement_start == -1:
                continue

            i = index_retracement_start - 1

            # 6: Candle is positive with local high
            candle_local_high = data.iloc[i]
            candle_is_local_high = candle_local_high['close'] >= data.iloc[i - 1]['close']
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
                candle_bull = data.iloc[index_bull_sequence]

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
            if data.iloc[index_retracement_end]['close'] < data.iloc[index_bull_sequence_start]['low']:
                continue

            # 9 - Breakout define by target price higher than the pullback high
            index_breakout = real_i
            candle_breakout_is_greater_than_high_pullback = False
            while True:
                candle_breakout = data.iloc[index_breakout]
                if candle_breakout['open'] - candle_breakout['close'] > 0:
                    index_breakout -= 1
                    break

                candle_breakout_is_greater_than_high_pullback = candle_breakout['close'] >= \
                                                                data.iloc[index_retracement_start - 1]['close']
                # LabelPullbackId label =

                index_breakout += 1

            # Found a pullback pattern!
            pullback_data = data[index_bull_sequence_start:index_retracement_end + 3]
            pullback_datas.append(np.array(
                [
                    pullback_data,
                    data.iloc[index_breakout],
                    candle_breakout_is_greater_than_high_pullback
                ]
            ))
        return pullback_datas

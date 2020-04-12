from LabelPullback import LabelPullback
from Pullback import Pullback
import plotly.graph_objects as go


class PullbackExtractor:
    @staticmethod
    def extract(data_bid):
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
            previous_candle_local_low = previous_candle['open'] <= data_bid.iloc[i - 2]['close'] and previous_candle[
                'close'] <= data_bid.iloc[i - 2]['close']
            if not previous_candle_local_low:
                continue

            # 5: Sequence of previous candle negative
            index_retracement_end = i - 1
            index_retracement_start = -1
            index_current_retracement = index_retracement_end - 1

            while True:
                # Get previous candle
                retracement_candle = data_bid.iloc[index_current_retracement]
                # Check if candle is negative
                retracement_candle_is_negative = retracement_candle['open'] - retracement_candle['close'] > 0
                if not retracement_candle_is_negative:
                    break
                # And higher then the next candle
                next_retracement_candle = data_bid.iloc[index_retracement_end + 1]
                retracement_candle_is_higher_than_next_candle = \
                    retracement_candle['close'] >= next_retracement_candle['open']
                if not retracement_candle_is_higher_than_next_candle:
                    break

                index_retracement_start = index_current_retracement
                index_current_retracement -= 1

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

            if index_bull_sequence_start < 0:
                continue

            # Retracement does not exceed 100% of pullback
            if data_bid.iloc[index_retracement_end]['close'] < data_bid.iloc[index_bull_sequence_start]['low']:
                continue

            # 9 - Breakout define by target price higher than the pullback high
            # index_breakout = real_i
            # label = LabelPullback.FAIL
            # while True:
            #    candle_breakout = data_bid.iloc[index_breakout]
            #    if candle_breakout['open'] - candle_breakout['close'] > 0:
            #        index_breakout -= 1
            #        break

            #    if candle_breakout['high'] >= data_bid.iloc[index_retracement_start]['high']:
            #        label = LabelPullback.SUCCESSFUL

            #    index_breakout += 1

            # 9 - pullback is a success when the candle breakout is more than 2 pip
            index_breakout = real_i - 1
            label = LabelPullback.FAIL
            if data_bid.iloc[index_breakout]['high'] - data_bid.iloc[index_retracement_end]['close'] > 0.0002:
                label = LabelPullback.SUCCESSFUL

            # Found a pullback pattern!
            pullback_data_bid = data_bid[index_bull_sequence_start:index_retracement_end]

            pullback = Pullback()
            pullback.label = label
            pullback.ohlc_total_size = pullback_data_bid.size

            bid_volume_sum = 0
            if pullback_data_bid.size > 0:
                bid_volume_sum = pullback_data_bid["bidVolume"].sum()
            pullback.total_volume = bid_volume_sum

            if index_bull_sequence_end - index_bull_sequence_start == 0:
                pullback.bullish_ohlc_size = 1
            else:
                pullback.bullish_ohlc_size = index_bull_sequence_end - index_bull_sequence_start

            bullish_volume_data_index = index_bull_sequence_end - index_bull_sequence_start
            if bullish_volume_data_index == 0:
                bullish_volume_data_index = 1
            bullish_volume_data = pullback_data_bid[0:bullish_volume_data_index]
            pullback.bullish_volume = bullish_volume_data["bidVolume"].sum()

            retracement_ohlc_size = index_retracement_end - index_retracement_start
            pullback.retracement_ohlc_size = retracement_ohlc_size

            retracement_volume = 0
            if pullback_data_bid[index_retracement_start - index_bull_sequence_start:index_retracement_end].size > 0:
                retracement_volume = \
                    pullback_data_bid[index_retracement_start - index_bull_sequence_start:index_retracement_end][
                        "bidVolume"].sum()

            pullback.retracement_volume = retracement_volume

            pullback.retracement_percentage = abs(
                (pullback_data_bid.iloc[0]["high"] -
                 pullback_data_bid.iloc[index_retracement_end - index_retracement_start]["high"]) / (
                        pullback_data_bid.iloc[index_bull_sequence_end - index_retracement_start]["high"] -
                        pullback_data_bid.iloc[0]["low"]))
            pullback.retracement_low_price = pullback_data_bid.iloc[index_retracement_end - index_retracement_start][
                "low"]
            pullback.start_price = pullback_data_bid.iloc[0]["low"]
            pullback.high_price = pullback_data_bid.iloc[index_bull_sequence_end - index_retracement_start]['high']

            pullback.index_start = index_bull_sequence_start
            pullback.index_end = index_retracement_end + 1
            pullback.retracement_date_start = data_bid.index[index_retracement_start]
            pullback_datas.append(pullback)

            # data_to_shows = data_bid[index_bull_sequence_start:index_breakout + 1]
            # title = pullback.label.name + ": Retracement Date Start: " + pullback.retracement_date_start.strftime(
            #     '%Y-%m-%d %H:%M:%S') + " Candle breakout date: " + data_bid.index[index_breakout].strftime(
            #    '%Y-%m-%d %H:%M:%S')
            # go.Figure(data=[go.Candlestick(x=data_to_shows.axes[0],
            #                                open=data_to_shows['open'],
            #                              high=data_to_shows['high'],
            #                              low=data_to_shows['low'],
            #                              close=data_to_shows['close'], )]) \
            #   .update_layout(title=title) \
            #   .show()

        return pullback_datas

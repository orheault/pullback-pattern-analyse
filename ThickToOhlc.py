import pandas as pd
import plotly.graph_objs as go
import sqlite3
import numpy as np
from datetime import datetime
from collections import OrderedDict

databaseConnection = sqlite3.connect("DukascopyEurUsd.db",detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
databaseCursor = databaseConnection.cursor()
 
# THICK TO OHLC
dateFormat = '%Y%m%d %H%M%S%f'
#dateparse = lambda x: datetime.strptime(x, dateFormat)
#data_frame = pd.read_csv('DAT_ASCII_EURUSD_T_202001.csv', names=['Date_Time', 'Bid', 'Ask', 'Volume'], index_col=0, parse_dates=['Date_Time'], date_parser=dateparse)
#data_frame = pd.read_csv('EURUSD_T_202001_o.csv', names=['Date_Time', 'Bid', 'Ask', 'Volume'], index_col=0, parse_dates=['Date_Time'], date_parser=dateparse)

SQL_QUERY_GET_TICK = 'SELECT * FROM ticks WHERE time > "2019-01-01" AND time < "2019-01-31"'
data_frame = pd.read_sql_query(SQL_QUERY_GET_TICK, databaseConnection, parse_dates={'time':dateFormat}, index_col='time')

data_ask =  data_frame['ask'].resample('30Min').ohlc()
data_bid =  data_frame['bid'].resample('30Min').ohlc()

# DISPLAY CANDLESTICK CHART
chartData = go.Candlestick(
                x=data_ask.axes[0],
                open=data_bid['open'],
                high=data_bid['high'],
                low=data_bid['low'],     
                           close=data_bid['close'])

#chart = go.Figure(data=chartData)
#chart.show()

# SEARCH FOR PULLBACK PATTERN
# THIS ALGO SEARCH BACKWARD, A BIT WEIRD
# Time move forward, for each candle, look backward if pattern is form.
data = data_bid
pullbackDatas = []
indexBreakout = 0
i=0
realI=0
for index, row in data.iterrows():
    i = realI
    realI += 1

    #print(data.iloc[i])
    currentCandle = row
    if i-1 < 0:
        continue
    previousCandle = data.iloc[i-1]

    # 1: Current candle positive
    currentCandlePositive = currentCandle['open']-currentCandle['close']<=0
    if currentCandlePositive == False:
        continue

    # 2: Previous candle negative
    previousCandleNegative = previousCandle['open']-previousCandle['close']>=0
    if previousCandleNegative == False:
        continue

    # 3: Current candle higher than previous candle
    currentCandleHigherThanPrevious = currentCandle['close']>=previousCandle['open']
    if currentCandleHigherThanPrevious == False:
        continue

    # 4; Previous candle make a local low. Compare with the candle before since I just check if the next candle id higher
    previousCandleLocalLow = previousCandle['open']<= data.iloc[i-2]['close']
    if previousCandleLocalLow == False:
        continue


    # 5: Sequence of previous candle negative
    indexRetracementEnd = i - 1
    indexRetracementStart = -1
    indexRetracement = indexRetracementEnd - 1
    isSearchingRetracement = True
    while isSearchingRetracement:
        # Get previous candle
        retracementCandle = data.iloc[indexRetracement]
        # Check if candle is negative
        retracementCandleIsNegative = retracementCandle['open'] - retracementCandle['close'] > 0
        if  retracementCandleIsNegative == False:
            # todo can i transform the while with a while True and delete the isSearchingRetracement variable???
            isSearchingRetracement = False
            break
        # And higer then the next candle
        nextRetracementCandle = data.iloc[indexRetracementEnd+1]
        retracementCandleIsHigherThanNextCandle = retracementCandle['close'] >= nextRetracementCandle['open']
        if retracementCandleIsHigherThanNextCandle == False:
            isSearchingRetracement = False
            break

        indexRetracementStart =  indexRetracement
        indexRetracement -= 1

    if indexRetracementStart == -1:
        continue

    i = indexRetracementStart - 1

    # 6: Candle is positive with local high
    candleLocalHigh = data.iloc[i]
    candleIsLocalHigh = candleLocalHigh['close'] >= data.iloc[i-1]['close']
    if candleIsLocalHigh == False:
        continue
    candleLocalHighIsPositive = candleLocalHigh['open'] - candleLocalHigh['close'] < 0
    if candleLocalHighIsPositive == False:
        continue

    i = i-1

    # 7 Sequence of previous candle positive
    indexBullSequenceEnd = i
    indexBullSequenceStart = -1
    indexBullSequence = i
    isSearchingBull = True
    while isSearchingBull:
        candleBull = data.iloc[indexBullSequence]

        # Candle is positive
        candleIsPositive = candleBull['open'] - candleBull['close'] < 0
        if candleIsPositive == False:
            isSearchingBull = False
            break

        # Bullish movement
        #isBullishMovement = candleBull['close'] < data.iloc[indexBullSequence + 1]
        #if isBullishMovement == False:
        #    isSearchingBull = False
        #    break

        indexBullSequenceStart = indexBullSequence
        indexBullSequence -= 1

    if indexBullSequenceStart == -1:
        continue

    # Retracement does not exceed 100% of pullback
    if data.iloc[indexRetracementEnd]['close'] < data.iloc[indexBullSequenceStart]['low']:
        continue

    # 9 - Breakout define by target price higher than the pullback high
    indexBreakout = realI
    candleBreakoutIsGratherThanHighPullback = False
    while True:
        candleBreakout = data.iloc[indexBreakout]
        if candleBreakout['open'] - candleBreakout['close'] > 0:
            indexBreakout -= 1
            break

        candleBreakoutIsGratherThanHighPullback = candleBreakout['close'] >= data.iloc[indexRetracementStart - 1]['close']
        indexBreakout += 1



    # Found a pullback pattern!
    pullbackData = data[indexBullSequenceStart:indexRetracementEnd + 1]
    pullbackDatas.append(np.array(
        [
            pullbackData,
            data.iloc[indexBreakout],
            candleBreakoutIsGratherThanHighPullback
        ]
    ))

print(sum(1 for t in pullbackDatas if t[2] == True))
print(sum(1 for t in pullbackDatas if t[2] == False))


#print(positiveCount)
#print(negativeCount)

# Verify each pull back with a chart
for pullback in pullbackDatas:
    go.Candlestick(  
        x=pullback[0].axes[0],
        open=pullback[0]['open'],
        high=pullback[0]['high'],
        low=pullback[0]['low'],                
        close=pullback[0]['close'])
    go.Figure(data=pullback[0], ).show()
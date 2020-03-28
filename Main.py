import numpy as np
import plotly.graph_objects as go
from PullbackExtractor import *
from DataManager import *

dataManager = DataManager("./data/DukascopyEurUsd.db")
pullbackExtractor = PullbackExtractor()

# Retrieve data
dataFrame = dataManager.retrieve_tick()
data = dataFrame['ask'].resample('30Min').ohlc()

# Extract pullback pattern
extractedData = pullbackExtractor.extract(data)

# Prepare data: populate training_data
# Prepare data: even number of item for each label
total_number_of_successful_pullback = sum(1 for t in extractedData if t.label == LabelPullback.SUCCESSFUL)
total_number_of_fail_pullback = sum(1 for t in extractedData if t.label == LabelPullback.FAIL)

number_of_item_to_retrieve = min(total_number_of_successful_pullback, total_number_of_fail_pullback)
training_data = []
fail_pullback = 0
successful_pullback = 0
for pullback in extractedData:
    training_data.append([pullback.data_ohlc, np.eye(2)[pullback.label.value]])

    if pullback.label == LabelPullback.FAIL:
        fail_pullback += 1
    else:
        successful_pullback += 1

    if fail_pullback >= number_of_item_to_retrieve or successful_pullback >= number_of_item_to_retrieve:
        break

#print(training_data[0])
#chartData = go.Candlestick(
#    x=training_data[0][0].axes[0],
#    open=training_data[0][0]['open'],
#    high=training_data[0][0]['high'],
#    low=training_data[0][0]['low'],
#    close=training_data[0][0]['close'])
#chart = go.Figure(data=chartData)
#chart.show()


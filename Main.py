import numpy as np
import plotly.graph_objects as go
import torch
from torch import optim
from PullbackExtractor import *
from DataManager import *


def prepare_data(data_unprocess):
    # Prepare data: populate training_data
    # Prepare data: even number of item for each label
    total_number_of_successful_pullback = sum(1 for t in data_unprocess if t.label == LabelPullback.SUCCESSFUL)
    total_number_of_fail_pullback = sum(1 for t in data_unprocess if t.label == LabelPullback.FAIL)

    number_of_item_to_retrieve = min(total_number_of_successful_pullback, total_number_of_fail_pullback)
    ret_data = []
    fail_pullback = 0
    successful_pullback = 0
    for pullback in data_unprocess:
        # todo check for unexpected values
        # todo normalize data between 0 and 1

        # index 0: Number of candle forming the pullback
        # index 1: Volume total
        # index 2: Number of candle forming bullish movement
        # index 3: Volume for the bullish movement
        # index 4: Number of candle forming retracement movement
        # index 5: Volume total forming the retracement
        # index 6: Retracement percentage
        # index 7: Start price
        # index 8: High price
        # index 9: Lower price

        features = [pullback.ohlc_total_size,
                    pullback.total_volume,
                    pullback.ohlc_bullish_size,
                    pullback.bullish_volume,
                    pullback.ohlc_retracement_size,
                    pullback.retracement_volume,
                    pullback.retracement_percentage,
                    pullback.start_price,
                    pullback.high_price,
                    pullback.retracement_low_price]

        ret_data.append([torch.tensor(features), torch.tensor(np.eye(2)[pullback.label.value])])

        if pullback.label == LabelPullback.FAIL:
            fail_pullback += 1
        else:
            successful_pullback += 1

        if fail_pullback >= number_of_item_to_retrieve or successful_pullback >= number_of_item_to_retrieve:
            break

    return ret_data


def ohlc_volume(x):
    if len(x):
        ohlc = {"open": x["open"][0], "high": max(x["high"]), "low": min(x["low"]), "close": x["close"][-1],
                "volume": sum(x["volume"])}
        return pd.Series(ohlc)


dataManager = DataManager("./data/DukascopyEurUsd.db")
pullbackExtractor = PullbackExtractor()

# Retrieve data
data_bid = dataManager.retrieve_tick()['bid'].resample('15Min').ohlc()
data_volume = dataManager.retrieve_tick()['bidVolume'].resample('15Min').agg({'bidVolume': 'sum'})

# agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'askVolume': 'sum'})
# data = dataFrame['ask'].resample('15Min').ohlc()

# Extract pullback pattern
extractedData = pullbackExtractor.extract(data_bid, data_volume)
prepared_data = prepare_data(extractedData)
training_data = prepared_data[0:(len(prepared_data) * 2) // 3]
testing_data = prepared_data[(len(prepared_data) * 2) // 3:len(prepared_data)]

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # fc1 refer to fully connected; first layer
        # 784 come from the flatten image of 28*28
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        # There is 2 classes, or 2 neuronnes
        self.fc4 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001, )

EPOCHS = 3

for EPOCHS in range(EPOCHS):
    for data in training_data:
        # data is a batch of featuresets and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 9))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

# Calculate accuracy
with torch.no_grad():
    for data in testing_data:
        X, y = data
        output = net(X.view(-1, 9))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct / total, 3))
# print(torch.argmax(net(X[1].view(-1,784))[0]))

import datetime
import os

import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras

from DataManager import DataManager
from LabelPullback import LabelPullback
from PullbackExtractor import PullbackExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def prepare_data(data_unprocess):
    # Prepare data: populate training_data
    # Prepare data: even number of item for each label
    total_number_of_successful_pullback = 0
    total_number_of_fail_pullback = 0

    for index, row in data_unprocess.iterrows():
        if row.iloc[0].label == LabelPullback.SUCCESSFUL:
            total_number_of_successful_pullback += 1
        else:
            total_number_of_fail_pullback += 1

    print("Number of successful pullback: " + str(total_number_of_successful_pullback))
    print("Number of failed pullback: " + str(total_number_of_fail_pullback))

    number_of_item_to_retrieve = min(total_number_of_successful_pullback, total_number_of_fail_pullback)
    ret_data = []
    fail_pullback = 0
    successful_pullback = 0
    for index, pullback in data_unprocess.iterrows():
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

        add_fail_pullback = False
        add_successful_pullback = False

        pullback = pullback.iloc[0]
        if fail_pullback < number_of_item_to_retrieve and pullback.label == LabelPullback.FAIL:
            add_fail_pullback = True
            fail_pullback += 1
        elif successful_pullback < number_of_item_to_retrieve and pullback.label == LabelPullback.SUCCESSFUL:
            add_successful_pullback = True
            successful_pullback += 1

        if add_fail_pullback or add_successful_pullback:
            features = [pullback.ohlc_total_size,
                        pullback.total_volume,
                        pullback.bullish_ohlc_size,
                        pullback.bullish_volume,
                        pullback.retracement_ohlc_size,
                        pullback.retracement_volume,
                        pullback.retracement_percentage,
                        pullback.start_price,
                        pullback.high_price,
                        pullback.retracement_low_price]
            ret_data.append([features, pullback.label.value])

    return pd.DataFrame(data=ret_data)


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(data_frame, shuffle=True, batch_size=64):
    data_frame = data_frame.copy()
    ds = tf.data.Dataset.from_tensor_slices((data_frame.iloc[:, 0], data_frame.iloc[:, 1]))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_frame))
    ds = ds.batch(batch_size)
    return ds


def normalize(value, minimum, maximum):
    if maximum - minimum == 0:
        return 0
    else:
        return (value - minimum) / (maximum - minimum)


def normalize_pullback(data):
    max_bullish_volume = 0
    min_bullish_volume = float('inf')
    max_high_price = 0
    min_high_price = float('inf')
    min_bullish_size = float('inf')
    max_bullish_size = 0
    min_retracement_size = float('inf')
    max_retracement_size = 0
    min_total_size = float('inf')
    max_total_size = 0
    min_retracement_low_price = float('inf')
    max_retracement_low_price = 0
    min_retracement_volume = float('inf')
    max_retracement_volume = 0
    min_start_price = float('inf')
    max_start_price = 0
    min_total_volume = float('inf')
    max_total_volume = 0

    # Calculate values for normalizing
    for pullback in data:
        if pullback.bullish_volume > max_bullish_volume:
            max_bullish_volume = pullback.bullish_volume
        if pullback.bullish_volume < min_bullish_volume:
            min_bullish_volume = pullback.bullish_volume
        if pullback.high_price > max_high_price:
            max_high_price = pullback.high_price
        if pullback.high_price < min_high_price:
            min_high_price = pullback.high_price
        if pullback.bullish_ohlc_size > max_bullish_size:
            max_bullish_size = pullback.bullish_ohlc_size
        if pullback.bullish_ohlc_size < min_bullish_size:
            min_bullish_size = pullback.bullish_ohlc_size
        if pullback.total_volume > max_total_volume:
            max_total_volume = pullback.total_volume
        if pullback.total_volume < min_total_volume:
            min_total_volume = pullback.total_volume
        if pullback.ohlc_total_size > max_total_size:
            max_total_size = pullback.ohlc_total_size
        if pullback.ohlc_total_size < min_total_size:
            min_total_size = pullback.ohlc_total_size
        if pullback.retracement_ohlc_size > max_retracement_size:
            max_retracement_size = pullback.retracement_ohlc_size
        if pullback.retracement_ohlc_size < min_retracement_size:
            min_retracement_size = pullback.retracement_ohlc_size
        if pullback.retracement_low_price > max_retracement_low_price:
            max_retracement_low_price = pullback.retracement_low_price
        if pullback.retracement_low_price < min_retracement_low_price:
            min_retracement_low_price = pullback.retracement_low_price
        if pullback.retracement_volume > max_retracement_volume:
            max_retracement_volume = pullback.retracement_volume
        if pullback.retracement_volume < min_retracement_volume:
            min_retracement_volume = pullback.retracement_volume
        if pullback.start_price > max_start_price:
            max_start_price = pullback.start_price
        if pullback.start_price < min_start_price:
            min_start_price = pullback.start_price

    # Normalize every pullback
    for pullback_to_normalize in data:
        pullback_to_normalize.bullish_volume = normalize(pullback_to_normalize.bullish_volume, min_bullish_volume,
                                                         max_bullish_volume)
        pullback_to_normalize.high_price = normalize(pullback_to_normalize.high_price, min_high_price, max_high_price)
        pullback_to_normalize.bullish_ohlc_size = normalize(pullback_to_normalize.bullish_ohlc_size, min_bullish_size,
                                                            max_bullish_size)
        pullback_to_normalize.retracement_ohlc_size = normalize(pullback_to_normalize.retracement_ohlc_size,
                                                                min_retracement_size, max_retracement_size)
        pullback_to_normalize.ohlc_total_size = normalize(pullback_to_normalize.ohlc_total_size, min_total_size,
                                                          max_total_size)
        pullback_to_normalize.retracement_low_price = normalize(pullback_to_normalize.retracement_low_price,
                                                                min_retracement_low_price, max_retracement_low_price)
        pullback_to_normalize.retracement_volume = normalize(pullback_to_normalize.retracement_volume,
                                                             min_retracement_volume, max_retracement_volume)
        pullback_to_normalize.start_price = normalize(pullback_to_normalize.start_price, min_start_price,
                                                      max_start_price)
        pullback_to_normalize.total_volume = normalize(pullback_to_normalize.total_volume, min_total_volume,
                                                       max_total_volume)


print("Initialization")
dataManager = DataManager("resources/DukascopyEurUsd.db")
pullbackExtractor = PullbackExtractor()

# Retrieve data
print("Retrieve data")
raw_data = dataManager.retrieve_ohlcv_between("2017-01-01", "2019-12-31", 5)
print("Total number of ohlcv : " + str(len(raw_data)))

# Clean data
print("Clean data")
indexNames = raw_data[(raw_data['bidVolume'] == 0)].index
raw_data.drop(indexNames, inplace=True)

# Extract pullback pattern
print("Extract pullback pattern")
extractedData = pullbackExtractor.extract(raw_data)
print("Number of extracted pullback: " + str(len(extractedData)))

# Normalize pullback
print("Normalize pullback")
normalize_pullback(extractedData)
normalize_data = pd.DataFrame(data=extractedData)

# Shuffle pullback
print("Shuffle data")
print("Dataframe before shuffle")
print(normalize_data.head())
shuffle_data = shuffle(normalize_data)
print("Dataframe after shuffle")
print(shuffle_data.head())

# Retrieve features from pullback object
print("Retrieve features from pullback object")
prepared_data = prepare_data(shuffle_data)

# Split training and test data
print("Split training and test data")
shuffle_index_start_testing_data = (len(prepared_data) * 97) // 100
training_data = prepared_data[0:shuffle_index_start_testing_data]
testing_data = prepared_data[shuffle_index_start_testing_data:len(prepared_data)]
print("Training data size:" + str(len(training_data)))
print("Testing data size:" + str(len(testing_data)))

# Convert training/test data to datasetsmetrics=['accuracy']
print("Convert training/test data to datasets")
train_dataset = df_to_dataset(training_data, batch_size=10)
test_dataset = df_to_dataset(testing_data, shuffle=False, batch_size=10)

model = keras.Sequential([
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Fit data")
for feature_batch, label_batch in train_dataset:
    model.fit(feature_batch, label_batch, epochs=64)

import plotly.graph_objects as go

print("Evaluate data")
i = 0
for feature_batch, label_batch in test_dataset:
    test_loss, test_acc = model.evaluate(feature_batch, label_batch)
    pullback = shuffle_data.iloc[shuffle_index_start_testing_data + i][0]
    print(pullback.label.name)
    data_to_shows = raw_data[pullback.index_start:pullback.index_end + 1]

    #title = pullback.label.name + ": Retracement Date Start: " + pullback.retracement_date_start.strftime(
    #    '%Y-%m-%d %H:%M:%S:%f')
    #go.Figure(data=[go.Candlestick(x=data_to_shows.axes[0],
    #                               open=data_to_shows['open'],
    #                               high=data_to_shows['high'],
    #                               low=data_to_shows['low'],
    #                               close=data_to_shows['close'], )]) \
    #    .update_layout(title=title).show()
    i += 1
tf.saved_model.save(model, "./save_model/5_min/")

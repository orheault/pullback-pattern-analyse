import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
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
        # ret_data.append([features, np.eye(2)[pullback.label.value]])
        ret_data.append([features, pullback.label.value])

        if pullback.label == LabelPullback.FAIL:
            fail_pullback += 1
        else:
            successful_pullback += 1

        if fail_pullback >= number_of_item_to_retrieve or successful_pullback >= number_of_item_to_retrieve:
            break

    return ret_data


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(data_frame, shuffle=True, batch_size=32):
    data_frame = data_frame.copy()
    ds = tf.data.Dataset.from_tensor_slices(
        (pd.DataFrame(data=data_frame).iloc[:, 0], pd.DataFrame(data=data_frame).iloc[:, 1]))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_frame))
    ds = ds.batch(batch_size)
    return ds


dataManager = DataManager("./data/DukascopyEurUsd.db")
pullbackExtractor = PullbackExtractor()

# Retrieve data
data_bid = dataManager.retrieve_tick()['bid'].resample('15Min').ohlc()
data_volume = dataManager.retrieve_tick()['bidVolume'].resample('15Min').agg({'bidVolume': 'sum'})

# Extract pullback pattern
extractedData = pullbackExtractor.extract(data_bid, data_volume)
print("Number of extracted pullback: " + str(len(extractedData)))
prepared_data = prepare_data(extractedData)

training_data = prepared_data[0:(len(prepared_data) * 2) // 3]
testing_data = prepared_data[(len(prepared_data) * 2) // 3:len(prepared_data)]

batch_size = 2  # A small batch sized is used for demonstration purposes
train_dataset = df_to_dataset(training_data, batch_size=batch_size)
test_dataset = df_to_dataset(testing_data, shuffle=False, batch_size=batch_size)

model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(10)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
for feature_batch, label_batch in train_dataset:
    model.fit(feature_batch, label_batch, epochs=5)

for feature_batch, label_batch in train_dataset:
    test_loss, test_acc = model.evaluate(feature_batch, label_batch)
    print("Tested Acc: ", test_acc)

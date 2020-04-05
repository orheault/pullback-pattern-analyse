import os
import tensorflow as tf
from tensorflow import keras
from PullbackExtractor import *
from DataManager import *
from sklearn.utils import shuffle

# Disable all debugging log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def prepare_data(data_unprocess):
    # Prepare data: populate training_data
    # Prepare data: even number of item for each label
    total_number_of_successful_pullback = 0
    total_number_of_fail_pullback = 0

    data_unprocess = shuffle(data_unprocess)
    for index, row in data_unprocess.iterrows():
        if row.iloc[0].label == LabelPullback.SUCCESSFUL:
            total_number_of_successful_pullback += 1
        else:
            total_number_of_fail_pullback += 1

    number_of_item_to_retrieve = min(total_number_of_successful_pullback, total_number_of_fail_pullback)
    ret_data = []
    fail_pullback = 0
    successful_pullback = 0
    for index, pullback in data_unprocess.iterrows():
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

        pullback = pullback.iloc[0]
        if fail_pullback < number_of_item_to_retrieve and pullback.label == LabelPullback.FAIL:
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
            ret_data.append([features, pullback.label.value])
            fail_pullback += 1

        if successful_pullback < number_of_item_to_retrieve and pullback.label == LabelPullback.SUCCESSFUL:
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
            ret_data.append([features, pullback.label.value])
            successful_pullback += 1

    return pd.DataFrame(data=ret_data)


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(data_frame, shuffle=True, batch_size=32):
    data_frame = data_frame.copy()
    ds = tf.data.Dataset.from_tensor_slices((data_frame.iloc[:, 0], data_frame.iloc[:, 1]))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_frame))
    ds = ds.batch(batch_size)
    return ds


print("Initialization")
dataManager = DataManager("./data/DukascopyEurUsd.db")
pullbackExtractor = PullbackExtractor()

# Retrieve data
print("Retrieve data")
raw_data = dataManager.retrieve_tick_between("2019-01-01", "2019-02-15")
data_bid = raw_data.copy()['bid'].resample('5Min').ohlc()
data_volume = raw_data.copy()['bidVolume'].resample('5Min').agg({'bidVolume': 'sum'})

# Extract pullback pattern
print("Extract pullback pattern")
extractedData = pd.DataFrame(data=pullbackExtractor.extract(data_bid, data_volume))
print("Number of extracted pullback: " + str(len(extractedData)))

# Shuffle data
print("Shuffle data")
df = shuffle(extractedData)

# Retrieve features from pullback object
print("Retrieve features from pullback object")
prepared_data = prepare_data(extractedData)

# Split training and test data
print("Split training and test data")
training_data = prepared_data[0:(len(prepared_data) * 2) // 3]
testing_data = prepared_data[(len(prepared_data) * 2) // 3:len(prepared_data)]

# Convert training/test data to datasets
print("Convert training/test data to datasets")
train_dataset = df_to_dataset(training_data)
test_dataset = df_to_dataset(testing_data, shuffle=False)

model = keras.Sequential([
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Fit data")
for feature_batch, label_batch in train_dataset:
    model.fit(feature_batch, label_batch, epochs=5)

print("Evaluate data")
for feature_batch, label_batch in train_dataset:
    test_loss, test_acc = model.evaluate(feature_batch, label_batch)

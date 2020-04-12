import csv
import sqlite3
import os
import pandas
from tqdm import tqdm

connection = sqlite3.connect("resources/DukascopyEurUsd.db")
cursor = connection.cursor()


def convert_thick_to_ohlc(ticks_data_frame, time_frame):
    ohlc_bid = ticks_data_frame.copy()['bid'].resample(str(time_frame) + 'Min').ohlc()
    data_volume = ticks_data_frame.copy()['bidVolume'].resample(str(time_frame) + 'Min').agg({'bidVolume': 'sum'})
    ohlc_v = ohlc_bid.merge(data_volume, left_on=['time'], right_on=['time'])
    ohlc_v['timeFrame'] = time_frame
    ohlc_v.to_sql('ohlcv', connection, if_exists='append', index=True)


def csv_to_db(file_name):
    if os.stat(file_name).st_size > 0:
        ticks_data_frame = pandas.read_csv(file_name, index_col=0, parse_dates=True)
        # ticks_data_frame.to_sql('ticks', connection, if_exists='append', index=False, )

        # Convert thick to ohlc
        # convert_thick_to_ohlc(ticks_data_frame, 1)
        convert_thick_to_ohlc(ticks_data_frame, 5)
        convert_thick_to_ohlc(ticks_data_frame, 15)
        # convert_thick_to_ohlc(ticks_data_frame, 30)
        # convert_thick_to_ohlc(ticks_data_frame, 60)


for filename in tqdm(os.listdir("resources/ticks")):
    csv_to_db('resources/ticks/' + filename)

connection.close()

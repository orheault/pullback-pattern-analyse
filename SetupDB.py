import sqlite3
import os

DB_NAME = "resources/DukascopyEurUsd.db"

if os.path.exists(DB_NAME):
    os.remove(DB_NAME)

SQL_CREATE_TABLE_TICKS = """CREATE TABLE ticks (time timestamp PRIMARY KEY, ask REAL, bid REAL, askVolume REAL, bidVolume 
REAL) """

SQL_CREATE_TABLE_OHLCV = """CREATE TABLE ohlcv (id INTEGER PRIMARY KEY AUTOINCREMENT, timeFrame INTEGER, time timestamp, open REAL, 
high REAL, low REAL, close REAL, bidVolume LONG) """

connection = sqlite3.connect(DB_NAME)
cursor = connection.cursor()
cursor.execute(SQL_CREATE_TABLE_TICKS)
cursor.execute(SQL_CREATE_TABLE_OHLCV)
connection.commit()
connection.close()

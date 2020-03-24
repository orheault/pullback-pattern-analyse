import csv
import sqlite3
import os
import pandas


DB_NAME = "DukascopyEurUsd.db"
SQL_CREATE_TABLE = """CREATE TABLE ticks (time timestamp PRIMARY KEY, ask REAL, bid REAL, askVolume REAL, bidVolume REAL)"""

connection = sqlite3.connect(DB_NAME)
cursor = connection.cursor()
cursor.execute(SQL_CREATE_TABLE)
connection.commit()
connection.close()


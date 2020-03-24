import csv
import sqlite3
import os
import pandas

connection = sqlite3.connect("DukascopyEurUsd.db")
cursor = connection.cursor()

def csvToDb(filename):
    if os.stat(filename).st_size > 0:
        df = pandas.read_csv(filename)
        df.to_sql('ticks', connection, if_exists='append', index=False, )



for filename in os.listdir("Ticks"):
    csvToDb('Ticks/' + filename)

connection.close()


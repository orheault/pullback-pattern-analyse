import sqlite3
import numpy as np
import plotly.graph_objects as go
from PullbackExtractor import *
from DataManager import *


dataManager = DataManager("./data/DukascopyEurUsd.db")
pullbackExtractor = PullbackExtractor()

dataFrame = dataManager.retrieveTick()
data =  dataFrame['ask'].resample('15Min').ohlc()
extractedData = pullbackExtractor.extract(data)

print(sum(1 for t in extractedData if t[2] == True))
print(sum(1 for t in extractedData if t[2] == False))


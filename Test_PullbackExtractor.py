from DataManager import DataManager
from PullbackExtractor import PullbackExtractor

dataManager = DataManager("resources/DukascopyEurUsd.db")
pullbackExtractor = PullbackExtractor()

raw_data = dataManager.retrieve_ohlcv_between("2019-01-01", "2019-01-07", 5)
extractedData = pullbackExtractor.extract(raw_data)

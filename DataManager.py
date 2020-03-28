import sqlite3
import pandas as pd

class DataManager:
    DATE_FORMAT = '%Y%m%d %H%M%S%f'

    def __init__(self, databaseFilename):
        self.databaseConnection = sqlite3.connect(databaseFilename,detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        self.databaseCursor = self.databaseConnection.cursor()
    
    def retrieveTick(self):
        SQL_QUERY_GET_TICK = 'SELECT * FROM ticks WHERE time > "2019-01-01" AND time < "2019-01-31"'
        return pd.read_sql_query(SQL_QUERY_GET_TICK, self.databaseConnection, parse_dates={'time':DataManager.DATE_FORMAT}, index_col='time')
    

import sqlite3
import pandas as pd


class DataManager:
    DATE_FORMAT = '%Y%m%d %H%M%S%f'

    def __init__(self, database_filename):
        self.databaseConnection = sqlite3.connect(database_filename,
                                                  detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        self.databaseCursor = self.databaseConnection.cursor()

    def retrieve_tick(self):
        sql_query_get_tick = 'SELECT * FROM ticks WHERE time > "2019-01-01" AND time < "2019-01-15"'
        return pd.read_sql_query(sql_query_get_tick, self.databaseConnection,
                                 parse_dates={'time': DataManager.DATE_FORMAT}, index_col='time')

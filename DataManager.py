import sqlite3
import pandas as pd


class DataManager:
    DATE_FORMAT = '%Y%m%d %H%M%S%f'

    def __init__(self, database_filename):
        self.databaseConnection = sqlite3.connect(database_filename,
                                                  detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        self.databaseCursor = self.databaseConnection.cursor()

    def retrieve_ohlcv_between(self, date_start, date_end, time_frame):
        sql_query_get_ohlcv = 'SELECT * FROM ohlcv WHERE time > "' + date_start + '"  AND time < "' + date_end + '" AND timeFrame = ' + str(time_frame) + ' ORDER BY time asc'

        return pd.read_sql_query(sql_query_get_ohlcv, self.databaseConnection,
                                 parse_dates={'time': DataManager.DATE_FORMAT}, index_col='time')

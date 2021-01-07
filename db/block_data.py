import sqlite3
import shutil
import os
from datetime import datetime

import numpy

from utility import logger

COL_BLOCK_NO = "block_no"
COL_DURATION = "duration"
COL_BLOCK_VALUE = "block_value"


def set_current_db_pointer(db_name, which_db="main"):
    """
    Sets the value of the current db to the given name
    :db_name: the new name of db to set as the current db
    :which_db: default to main. Either main or pools
    :return: None
    """
    global CURRENT_DB_NAME
    global CURRENT_POOLS_DB_NAME
    if which_db == "pools":
        CURRENT_POOLS_DB_NAME = db_name
    else:
        CURRENT_DB_NAME = db_name


def set_current_db_pointer_to_main_copy(which_db="main"):
    """
    Sets the value of the current db to name of the main copy
    :param which_db: main or pools; default is main
    :return: None
    """
    global MAIN_DB_NAME
    global MAIN_POOLS_DB_NAME
    if which_db == "pools":
        set_current_db_pointer(MAIN_POOLS_DB_NAME, which_db="pools")
    else:
        set_current_db_pointer(MAIN_DB_NAME)


def get_data_file_path(db_name):
    """
    Returns the path to a data file whose name is given
    :db_name: the name of the datafile
    :return: String the path to the database file given
    """
    return os.path.join(".", DATA_DIRECTORY, db_name)


def get_current_db_file_path(which_db="main"):
    """
    Returns the path to current database file
    :param which_db: return main db file path or the pools db file path?
    :return: String the path to the current database file being used
    """
    global CURRENT_DB_NAME
    global CURRENT_POOLS_DB_NAME
    if which_db == "pools":
        return get_data_file_path(CURRENT_POOLS_DB_NAME)
    else:
        return get_data_file_path(CURRENT_DB_NAME)


def get_main_db_file_path(which_db="main"):
    """
    Returns the path to current database file
    :param which_db: main or pools; default to main
    :return: String the path to the current database file being used
    """
    global MAIN_DB_NAME
    global MAIN_POOLS_DB_NAME
    if which_db == "pools":
        return get_data_file_path(MAIN_POOLS_DB_NAME)
    else:
        return get_data_file_path(MAIN_DB_NAME)


def get_new_temp_db_file_name_and_path(name=None):
    """
    Initializes the block data SQLite database
    :param name: the name of the new temp database
    :return: String the path to a new database file name to be used temporarily
    """
    new_name = datetime.now().strftime("%d-%m-%Y-%H.db") if name is None else name
    return new_name, get_data_file_path(new_name)


def is_main_copy_in_use(which_db="main"):
    """
    :return: Boolean Whether or not the current copy of the database is the main copy
    :param which_db: main or pools; default to main
    """
    global CURRENT_DB_NAME, MAIN_DB_NAME, CURRENT_POOLS_DB_NAME, MAIN_POOLS_DB_NAME
    if which_db == "pools":
        return CURRENT_POOLS_DB_NAME is MAIN_POOLS_DB_NAME
    else:
        return CURRENT_DB_NAME is MAIN_DB_NAME


def init_pools_data_tables():
    """
    Initializes the pools SQLite database
    :return: None
    """
    '''
    Create database elements if they do not exist
    '''
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        # pool block occurrence data
        c.execute('''
        CREATE TABLE IF NOT EXISTS pools(name TEXT,
        share DOUBLE PRECISION,
        id INTEGER PRIMARY KEY AUTOINCREMENT);
        ''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS pool_block_occurrences(date_found_unix BIGINT,
        pool_id INT NOT NULL,
        block_no INT NOT NULL PRIMARY KEY,
        FOREIGN KEY(pool_id) REFERENCES pools(id));
        ''')
        c.execute('''
        CREATE INDEX IF NOT EXISTS pool_block_occurrences_index ON pool_block_occurrences(date_found_unix);
        ''')
        conn.commit()
    finally:
        conn.close()


def init_multi_pool_table(table_name, pool_ids, column_prefix="luck"):
    """
    Initializes the lucks tables in SQLite database
    :param column_prefix: prefix to be used with pool ids to make column names
    :param table_name: table name
    :param pool_ids: pool ids to be used as luck column names
    :return: None
    """
    '''
    Create database elements if they do not exist
    '''
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        columns_def_str = ", ".join(["{}_{} DOUBLE PRECISION".format(column_prefix, pool_id) for pool_id in pool_ids])
        c.execute('''
        CREATE TABLE IF NOT EXISTS {}(window_start BIGINT PRIMARY KEY, {});
        '''.format(table_name, columns_def_str))
        conn.commit()
    finally:
        conn.close()


def init_block_data_db():
    """
    Initializes the block data SQLite database
    :return: None
    """
    '''
    Create database elements if they do not exist
    '''
    try:
        conn = sqlite3.connect(get_current_db_file_path())
        c = conn.cursor()
        # Create main data tables
        # raw block data of the main pool in use
        c.execute('''
        CREATE TABLE IF NOT EXISTS raw_data(date_found_unix BIGINT,
        found_at_date TEXT,
        found_at_time TEXT,
        duration INT,
        hash_rate INT,
        block_no INT NOT NULL PRIMARY KEY,
        block_value DOUBLE PRECISION);
        ''')
        # Create views
        c.execute('''
            CREATE VIEW IF NOT EXISTS solved_blocks_per_day AS 
            SELECT COUNT(found_at_date), found_at_date FROM raw_data GROUP BY found_at_date ORDER BY found_at_date;
            ''')
        c.execute('''
        CREATE VIEW IF NOT EXISTS avg_duration_per_day AS 
        SELECT AVG(duration), found_at_date FROM raw_data GROUP BY found_at_date ORDER BY found_at_date;
        ''')
        c.execute('''
        CREATE VIEW IF NOT EXISTS grouped_per_day AS 
        SELECT 
        found_at_date, 
        COUNT(found_at_date) AS block_count, 
        AVG(duration) AS duration, 
        AVG(hash_rate) AS hash_rate,
        AVG(block_value) AS block_value 
        FROM raw_data GROUP BY found_at_date ORDER BY found_at_date;
        ''')
        conn.commit()
    finally:
        conn.close()


def print_all_raw_data():
    """
    Selects and prints all records from the main table in the mine database
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path())
        c = conn.cursor()
        results = c.execute("SELECT * FROM raw_data;")
        rows = results.fetchall()
        for row in rows:
            print(row)
        print("Count: ", len(rows))
    finally:
        conn.close()


def get_all_day_counts():
    """
    Selects and prints all records from the main table in the mine database
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path())
        c = conn.cursor()
        results = c.execute("SELECT found_at_date, COUNT(found_at_date) FROM raw_data GROUP BY found_at_date;")
        resultsList = []
        rows = results.fetchall()
        for row in rows:
            resultsList.append([r for r in row])
        return resultsList
    finally:
        conn.close()
    return None


def print_all_pools_data():
    """
    Selects and prints all records from the main table in the mine database
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path("pools"))
        c = conn.cursor()
        results = c.execute("SELECT * FROM pool_block_occurrences;")
        rows = results.fetchall()
        for row in rows:
            print(row)
        print("Count: ", len(rows))
    finally:
        conn.close()


def print_all_raw_data_in_tsdb_format2():
    """
    Selects and prints all records from the main table in the mine database into TSDB format
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path())
        c = conn.cursor()
        results = c.execute("SELECT * FROM raw_data;")
        rows = results.fetchall()
        for row in rows:
            ts = row[0]
            year_ts = row[1].split("-")[0]
            month_ts = row[1].split("-")[1]
            day_ts = row[1].split("-")[2]
            hour_ts = row[2].split(":")[0]
            minute_ts = row[2].split(":")[1]
            second_ts = row[2].split(":")[2]
            dur = row[3]
            hash_rate = row[4]
            block_no = row[5]
            block_value = row[6]
            print(
                "duration {0} {7} year_ts={1} month_ts={2} day_ts={3} hour_ts={4} minute_ts={5} second_ts={6} block_no={9}".format(
                    ts,
                    year_ts, month_ts, day_ts,
                    hour_ts, minute_ts, second_ts,
                    dur, hash_rate, block_no, block_value))
            print(
                "hash_rate {0} {8} year_ts={1} month_ts={2} day_ts={3} hour_ts={4} minute_ts={5} second_ts={6} block_no={9}".format(
                    ts,
                    year_ts, month_ts, day_ts,
                    hour_ts, minute_ts, second_ts,
                    dur, hash_rate, block_no, block_value))
            print(
                "block_value {0} {10} year_ts={1} month_ts={2} day_ts={3} hour_ts={4} minute_ts={5} second_ts={6} block_no={9}".format(
                    ts,
                    year_ts, month_ts, day_ts,
                    hour_ts, minute_ts, second_ts,
                    dur, hash_rate, block_no, block_value))
        print("Count: ", len(rows))
    finally:
        conn.close()


def print_all_raw_data_in_tsdb_format():
    """
    Selects and prints all records from the main table in the mine database into TSDB format
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path())
        c = conn.cursor()
        results = c.execute("SELECT * FROM raw_data;")
        rows = results.fetchall()
        for row in rows:
            ts = row[0]
            year_ts = row[1].split("-")[0]
            month_ts = row[1].split("-")[1]
            day_ts = row[1].split("-")[2]
            hour_ts = row[2].split(":")[0]
            minute_ts = row[2].split(":")[1]
            second_ts = row[2].split(":")[2]
            dur = row[3]
            hash_rate = row[4]
            block_no = row[5]
            block_value = row[6]
            print(
                "duration{{year_ts=\"{1}\",month_ts=\"{2}\",day_ts=\"{3}\",hour_ts=\"{4}\",minute_ts=\"{5}\",second_ts=\"{6}\",block_no=\"{9}\"}} {7} {0}".format(
                    ts,
                    year_ts, month_ts, day_ts,
                    hour_ts, minute_ts, second_ts,
                    dur, hash_rate, block_no, block_value))
            print(
                "hash_rate{{year_ts=\"{1}\",month_ts=\"{2}\",day_ts=\"{3}\",hour_ts=\"{4}\",minute_ts=\"{5}\",second_ts=\"{6}\",block_no=\"{9}\"}} {8} {0}".format(
                    ts,
                    year_ts, month_ts, day_ts,
                    hour_ts, minute_ts, second_ts,
                    dur, hash_rate, block_no, block_value))
            print(
                "block_value{{year_ts=\"{1}\",month_ts=\"{2}\",day_ts=\"{3}\",hour_ts=\"{4}\",minute_ts=\"{5}\",second_ts=\"{6}\",block_no=\"{9}\"}} {10} {0}".format(
                    ts,
                    year_ts, month_ts, day_ts,
                    hour_ts, minute_ts, second_ts,
                    dur, hash_rate, block_no, block_value))
        print("Count: ", len(rows))
    finally:
        conn.close()


def get_columns(columns, table_name="raw_data", which_db="main", limit=100):
    """
    Selects and returns the requested columns
    :param columns: list of names of columns
    :param table_name: table name
    :param which_db: which sqlite db? main or pools
    :param limit: limit of number of rows to return
    :return: A list of rows
    """
    data_rows = []
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db=which_db))
        c = conn.cursor()
        results = c.execute(("SELECT " + ",".join(columns) + " FROM {} LIMIT {};").format(table_name, limit))
        rows = results.fetchall()
        for row in rows:
            data_rows.append(row)
        return data_rows
    finally:
        conn.close()


def get_table_size(table_name="raw_data", which_db="main"):
    """
    Selects and returns the number of rows in the given table
    :param table_name: table name
    :param which_db: which sqlite db? main or pools
    :return: Size of table
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db=which_db))
        c = conn.cursor()
        results = c.execute("SELECT count(*) FROM {};".format(table_name))
        rows = results.fetchall()
        for row in rows:
            return row[0]
        return None
    finally:
        conn.close()


def truncate_table(table_name="raw_data", which_db="main"):
    """
    Truncates the given table
    :param table_name: table name
    :param which_db: which sqlite db? main or pools
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db=which_db))
        c = conn.cursor()
        results = c.execute("DELETE FROM {};".format(table_name))
        conn.commit()
    finally:
        conn.close()


def insert_raw_data(date_found_unix, found_at_date, found_at_time, duration,
                    hash_rate, block_no, block_value):
    """
    Insert a new raw record
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path())
        c = conn.cursor()
        c.execute("INSERT INTO raw_data VALUES (?, ?, ?, ?, ?, ?, ?);",
                  [date_found_unix, found_at_date, found_at_time, duration, hash_rate, block_no, block_value]
                  )
        conn.commit()
    finally:
        conn.close()


def insert_pool(name, share):
    """
    Insert a new pool
    :return: The id of the pool
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        c.execute("INSERT INTO pools(name, share) VALUES (?, ?);",
                  [name, share]
                  )
        pool_id = c.lastrowid
        conn.commit()
        return pool_id
    finally:
        conn.close()


def get_pools():
    """
    Selects and returns all the pool records
    :return: None
    """
    try:
        data_rows = []
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        results = c.execute("SELECT * FROM pools")
        rows = results.fetchall()
        for row in rows:
            data_rows.append(row)
        return data_rows
    finally:
        conn.close()


def insert_pool_block_occurrence(date_found_unix, pool_id, block_no):
    """
    Insert a new block win by a pool
    :param date_found_unix: time of occurrence
    :param pool_id: id of the pool in which it occurred
    :param block_no: block no of the occurred block
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        c.execute("INSERT INTO pool_block_occurrences VALUES (?, ?, ?);",
                  [date_found_unix, pool_id, block_no]
                  )
        conn.commit()
    finally:
        conn.close()


def insert_pool_assessments(table_name, data_point_timestamp, lucks, column_prefix="assessment"):
    insert_pool_lucks(table_name, data_point_timestamp, lucks, column_prefix)


def insert_pool_lucks(table_name, data_point_timestamp, lucks, column_prefix="luck"):
    """
    Insert a new block win by a pool
    :param column_prefix: prefix to be used for columns
    :param table_name: the table name of the specific luck table
    :param data_point_timestamp: the data point at which we calculate luck
    :param lucks: luck values to record in the table in its columns in order
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        lucks_names_part = ", ".join(["{}" for l in lucks])
        lucks_values_part = ", ".join(["?" for l in lucks])
        luck_column_names = []
        luck_column_values = []
        for pid in lucks:
            luck_column_names.append("{}_{}".format(column_prefix, pid))
            luck_column_values.append(lucks[pid])
        c.execute(("INSERT INTO {} (window_start, " + lucks_names_part + ") VALUES (?, " + lucks_values_part + ");")
                  .format(table_name, *luck_column_names),
                  [data_point_timestamp, *luck_column_values]
                  )
        conn.commit()
    finally:
        conn.close()


def get_all_block_occurrences(return_oldest_first=False, pool_id=None):
    order = "ASC" if return_oldest_first else "DESC"
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        if pool_id is None:
            results = c.execute("SELECT date_found_unix, pool_id FROM pool_block_occurrences "
                                + "ORDER BY date_found_unix {}".format(order))
            rows = results.fetchall()
            result_rows = []
            for row in rows:
                result_rows.append([e for e in row])
            return result_rows
        else:
            results = c.execute(("SELECT date_found_unix FROM pool_block_occurrences "
                                 + "WHERE pool_id={} "
                                 + "ORDER BY date_found_unix {}").format(pool_id, order))
            rows = results.fetchall()
            result_rows = []
            for row in rows:
                result_rows.append(row[0])
            return result_rows
    finally:
        conn.close()


def get_latest_pool_block_occurrence_timestamp(return_oldest=False):
    """
    Return the latest timestamp or None if table is empty
    :param return_oldest: returns the oldest if True
    :return: None
    """
    order = "ASC" if return_oldest else "DESC"
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        results = c.execute("SELECT date_found_unix FROM pool_block_occurrences "
                            + "ORDER BY date_found_unix {} LIMIT 1;".format(order))
        rows = results.fetchall()
        for row in rows:
            return row[0]
        return None
    finally:
        conn.close()


def get_latest_luck_window_start_timestamp(table_name, earlier_than=None):
    """
    Return the latest timestamp or None if table is empty
    :param table_name: the luck table name
    :param earlier_than: if given, the latest point earlier than the given timestamp is returned
    :return: None
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        results = None
        if earlier_than is None:
            results = c.execute(("SELECT window_start FROM {} " +
                                 "ORDER BY window_start DESC LIMIT 1;").format(table_name))
        else:
            results = c.execute(("SELECT window_start FROM {} WHERE window_start < {} " +
                                 "ORDER BY window_start DESC LIMIT 1;").format(table_name, earlier_than))
        rows = results.fetchall()
        for row in rows:
            return row[0]
        return None
    finally:
        conn.close()


def get_pool_luck_values(pool_id, table_names, oldest_ts, newest_ts, column_prefix="luck",
                         filter_by_block_occurrence=False, round_to_n_decimal_points=6):
    """

    :param round_to_n_decimal_points: All numbers are rounded up to this number of decimal points before being returned
    :param filter_by_block_occurrence: If True, only block occurrence timestamps are returned
    :param column_prefix: prefix to use to make column name
    :param pool_id: the id of the pool whose luck is to be returned
    :param table_names: list of names of all tables that contain this pool's luck
    :param oldest_ts: lower bound of timestamp
    :param newest_ts: upper bound of timestamp
    :return: a list of lists like [[window_start, luck in table 1, luck in table 2, ...], ...]
    """
    data_points = []
    column_name = "{}_{}".format(column_prefix, pool_id)
    for table_name in table_names:
        query = "SELECT window_start, {} FROM {} WHERE window_start BETWEEN {} AND {} ORDER BY window_start ASC;".format(
            column_name, table_name, oldest_ts, newest_ts)
        try:
            conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
            c = conn.cursor()
            results = c.execute(query)
            rows = results.fetchall()
            # if it's the first table, first prepare timestamp of the data points
            if len(data_points) == 0:
                for row in rows:
                    data_points.append([row[0]])
            row_index = 0
            for row in rows:
                # round the returning values
                rounded_value = round(row[1], ndigits=round_to_n_decimal_points)
                data_points[row_index].append(rounded_value)
                row_index += 1
        finally:
            conn.close()
    if filter_by_block_occurrence:
        data_points_filtered = []
        pool_occurrence_timestamps = get_all_block_occurrences(pool_id=pool_id)
        for d in data_points:
            if d[0] in pool_occurrence_timestamps:
                data_points_filtered.append(d)
        data_points = data_points_filtered
    return data_points


def get_total_block_occurrences_of_pool(pool_id, start_timestamp=0, end_timestamp=int(datetime.now().timestamp())):
    """
    Returns the number of data points of a pool from start timestamp to end timestamp
    :param pool_id: the pool id from the pools table
    :param start_timestamp: left side of the interval; default is the beginning of time
    :param end_timestamp: right side of the interval; default is now which is assumed to be the latest time
    :return: the number of block occurrences of this pool in the given time interval
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        results = c.execute(("SELECT count(*) FROM pool_block_occurrences WHERE pool_id = {}"
                             + " AND date_found_unix BETWEEN {} AND {};").format(pool_id, start_timestamp,
                                                                                 end_timestamp))
        rows = results.fetchall()
        for row in rows:
            return row[0]
        return None
    finally:
        conn.close()


def get_list_of_table_names(which_db="main"):
    """
    Returns the list of table names in the database
    """
    try:
        conn = sqlite3.connect(get_current_db_file_path(which_db="pools"))
        c = conn.cursor()
        results = c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        rows = results.fetchall()
        result = []
        for row in rows:
            result.append(row[0])
        return result
    finally:
        conn.close()


def switch_to_temporary_copy(which_db="main"):
    """
    Makes a copy of the mine database and the rest of requests
    to this package will be applied to the copy.
    Note: the function does nothing (and logs a warning) if the current copy is not the main copy
    :param which_db: main or pools; default is main
    :return: None
    """
    # check if the current copy is the main copy
    if not is_main_copy_in_use(which_db=which_db):
        logger("db").warning("Trying to switch to temporary copy while we are NOT on the main copy."
                             + " Ignoring the call. Which db: {}".format(which_db))
        return
    # make copy of the database file
    new_db_name, new_db_full_path = get_new_temp_db_file_name_and_path()
    shutil.copy(get_main_db_file_path(which_db=which_db), new_db_full_path)
    # set the database pointer to the copy
    set_current_db_pointer(new_db_name, which_db=which_db)


def switch_to_main_copy(save_temporary_copy=False, remove_temporary_copy=False, which_db="main"):
    """
    Does nothing if the current copy is the main copy
    Otherwise,
    Sets the current database pointer to the main copy
    :param save_temporary_copy: If True, the main copy will be overwritten from the current copy
    :param remove_temporary_copy: If True, the current copy will be deleted
    :param which_db: main or pools; default is main
    :return: None
    """
    # Do nothing if the main copy is the current copy
    if is_main_copy_in_use(which_db=which_db):
        return
    # Overwrite the main copy from the current copy if requested
    if save_temporary_copy:
        # Get a backup from the main copy
        backup_db_name, backup_db_full_path = get_new_temp_db_file_name_and_path("backup")
        shutil.copy(get_main_db_file_path(which_db=which_db), backup_db_full_path)
        # Overwrite the main copy
        shutil.copy(get_current_db_file_path(which_db=which_db), get_main_db_file_path(which_db=which_db))
        # Remove the backup
        os.remove(backup_db_full_path)
    # Remove the current temporary copy
    if remove_temporary_copy:
        os.remove(get_current_db_file_path(which_db=which_db))
    # set the database pointer to the main copy
    set_current_db_pointer_to_main_copy(which_db=which_db)


def get_last_block_no():
    """
    Gets block value for the most recent block added to database
    :return: an integer representing block value
    """
    conn = sqlite3.connect(get_current_db_file_path())
    c = conn.cursor()
    results = c.execute("SELECT block_no FROM raw_data ORDER BY block_no DESC LIMIT 1;")
    row = results.fetchall()
    return row[0][0]


DATA_DIRECTORY = "data"
MAIN_DB_NAME = "pool_analyzer.db"
MAIN_POOLS_DB_NAME = "pool_analyzer_pools.db"
CURRENT_DB_NAME = None
CURRENT_POOLS_DB_NAME = None
# Set current db pointer to the main db pointer
set_current_db_pointer(MAIN_DB_NAME, which_db="main")
set_current_db_pointer(MAIN_POOLS_DB_NAME, which_db="pools")

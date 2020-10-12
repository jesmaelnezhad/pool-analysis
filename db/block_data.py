import sqlite3
import shutil
import os
from datetime import datetime

from utility import logger

COL_BLOCK_NO = "block_no"
COL_DURATION = "duration"
COL_BLOCK_VALUE = "block_value"


def set_current_db_pointer(db_name):
    """
    Sets the value of the current db to the given name
    :db_name: the new name of db to set as the current db
    :return: None
    """
    global CURRENT_DB_NAME
    CURRENT_DB_NAME = db_name


def set_current_db_pointer_to_main_copy():
    """
    Sets the value of the current db to name of the main copy
    :return: None
    """
    global MAIN_DB_NAME
    set_current_db_pointer(MAIN_DB_NAME)


def get_data_file_path(db_name):
    """
    Returns the path to a data file whose name is given
    :db_name: the name of the datafile
    :return: String the path to the database file given
    """
    return os.path.join(".", DATA_DIRECTORY, db_name)


def get_current_db_file_path():
    """
    Returns the path to current database file
    :return: String the path to the current database file being used
    """
    global CURRENT_DB_NAME
    return get_data_file_path(CURRENT_DB_NAME)


def get_main_db_file_path():
    """
    Returns the path to current database file
    :return: String the path to the current database file being used
    """
    global MAIN_DB_NAME
    return get_data_file_path(MAIN_DB_NAME)


def get_new_temp_db_file_name_and_path(name=None):
    """
    Initializes the block data SQLite database
    :param name: the name of the new temp database
    :return: String the path to a new database file name to be used temporarily
    """
    new_name = datetime.now().strftime("%d-%m-%Y-%H.db") if name is None else name
    return new_name, get_data_file_path(new_name)


def is_main_copy_in_use():
    """
    :return: Boolean Whether or not the current copy of the database is the main copy
    """
    global CURRENT_DB_NAME, MAIN_DB_NAME
    return CURRENT_DB_NAME is MAIN_DB_NAME


def init_block_data_db():
    """
    Initializes the block data SQLite database
    :return: None
    """
    '''
    Create database elements if they do not exist
    '''
    conn = sqlite3.connect(get_current_db_file_path())
    c = conn.cursor()
    # Create main data table
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
    conn.close()


def print_all_raw_data():
    """
    Selects and prints all records from the main table in the mine database
    :return: None
    """
    conn = sqlite3.connect(get_current_db_file_path())
    c = conn.cursor()
    results = c.execute("SELECT * FROM raw_data;")
    rows = results.fetchall()
    for row in rows:
        print(row)
    print("Count: ", len(rows))


def print_all_raw_data_in_tsdb_format2():
    """
    Selects and prints all records from the main table in the mine database into TSDB format
    :return: None
    """
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
        print("duration {0} {7} year_ts={1} month_ts={2} day_ts={3} hour_ts={4} minute_ts={5} second_ts={6} block_no={9}".format(ts,
                            year_ts, month_ts, day_ts,
                            hour_ts, minute_ts, second_ts,
                            dur, hash_rate, block_no, block_value))
        print("hash_rate {0} {8} year_ts={1} month_ts={2} day_ts={3} hour_ts={4} minute_ts={5} second_ts={6} block_no={9}".format(ts,
                            year_ts, month_ts, day_ts,
                            hour_ts, minute_ts, second_ts,
                            dur, hash_rate, block_no, block_value))
        print("block_value {0} {10} year_ts={1} month_ts={2} day_ts={3} hour_ts={4} minute_ts={5} second_ts={6} block_no={9}".format(ts,
                            year_ts, month_ts, day_ts,
                            hour_ts, minute_ts, second_ts,
                            dur, hash_rate, block_no, block_value))
    print("Count: ", len(rows))


def print_all_raw_data_in_tsdb_format():
    """
    Selects and prints all records from the main table in the mine database into TSDB format
    :return: None
    """
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
        print("duration{{year_ts=\"{1}\",month_ts=\"{2}\",day_ts=\"{3}\",hour_ts=\"{4}\",minute_ts=\"{5}\",second_ts=\"{6}\",block_no=\"{9}\"}} {7} {0}".format(ts,
                            year_ts, month_ts, day_ts,
                            hour_ts, minute_ts, second_ts,
                            dur, hash_rate, block_no, block_value))
        print("hash_rate{{year_ts=\"{1}\",month_ts=\"{2}\",day_ts=\"{3}\",hour_ts=\"{4}\",minute_ts=\"{5}\",second_ts=\"{6}\",block_no=\"{9}\"}} {8} {0}".format(ts,
                            year_ts, month_ts, day_ts,
                            hour_ts, minute_ts, second_ts,
                            dur, hash_rate, block_no, block_value))
        print("block_value{{year_ts=\"{1}\",month_ts=\"{2}\",day_ts=\"{3}\",hour_ts=\"{4}\",minute_ts=\"{5}\",second_ts=\"{6}\",block_no=\"{9}\"}} {10} {0}".format(ts,
                            year_ts, month_ts, day_ts,
                            hour_ts, minute_ts, second_ts,
                            dur, hash_rate, block_no, block_value))
    print("Count: ", len(rows))


def get_columns(columns, limit=100):
    """
    Selects and returns the duration and block_number of all records
    :return: None
    """

    data_rows = []

    conn = sqlite3.connect(get_current_db_file_path())
    c = conn.cursor()
    results = c.execute(("SELECT " + ",".join(columns) + " FROM raw_data LIMIT {};").format(limit))
    rows = results.fetchall()
    for row in rows:
        data_rows.append(row)
    return data_rows


def insert_raw_data(date_found_unix, found_at_date, found_at_time, duration,
                    hash_rate, block_no, block_value):
    """
    Insert a new raw record
    :return: None
    """
    conn = sqlite3.connect(get_current_db_file_path())
    c = conn.cursor()
    c.execute("INSERT INTO raw_data VALUES (?, ?, ?, ?, ?, ?, ?);",
              [date_found_unix, found_at_date, found_at_time, duration, hash_rate, block_no, block_value]
              )
    conn.commit()
    conn.close()


def switch_to_temporary_copy():
    """
    Makes a copy of the mine database and the rest of requests
    to this package will be applied to the copy.
    Note: the function does nothing (and logs a warning) if the current copy is not the main copy
    :return: None
    """
    # check if the current copy is the main copy
    if not is_main_copy_in_use():
        logger("db").warning("Trying to switch to temporary copy while we are NOT on the main copy. Ignoring the call.")
        return
    # make copy of the database file
    new_db_name, new_db_full_path = get_new_temp_db_file_name_and_path()
    shutil.copy(get_main_db_file_path(), new_db_full_path)
    # set the database pointer to the copy
    set_current_db_pointer(new_db_name)


def switch_to_main_copy(save_temporary_copy=False, remove_temporary_copy=False):
    """
    Does nothing if the current copy is the main copy
    Otherwise,
    Sets the current database pointer to the main copy
    :save_temporary_copy: If True, the main copy will be overwritten from the current copy
    :remove_temporary_copy: If True, the current copy will be deleted
    :return: None
    """
    # Do nothing if the main copy is the current copy
    if is_main_copy_in_use():
        return
    # Overwrite the main copy from the current copy if requested
    if save_temporary_copy:
        # Get a backup from the main copy
        backup_db_name, backup_db_full_path = get_new_temp_db_file_name_and_path("backup")
        shutil.copy(get_main_db_file_path(), backup_db_full_path)
        # Overwrite the main copy
        shutil.copy(get_current_db_file_path(), get_main_db_file_path())
        # Remove the backup
        os.remove(backup_db_full_path)
    # Remove the current temporary copy
    if remove_temporary_copy:
        os.remove(get_current_db_file_path())
    # set the database pointer to the main copy
    set_current_db_pointer_to_main_copy()
    
    
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
CURRENT_DB_NAME = None
# Set current db pointer to the main db pointer
set_current_db_pointer(MAIN_DB_NAME)

# The databases are initialized here:
# Creating the databases, their tables, and their views happen here.
######################################################################
from utility import logger
from .block_data import init_block_data_db
from .prediction_data import init_prediction_data_db
######################################################################
logger("db").info("Setting up mine data SQLite database")
init_block_data_db()
######################################################################
logger("db").info("Setting up prediction data SQLite database")
init_prediction_data_db()
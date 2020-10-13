from db import block_data
from prediction import Pool
from prediction.random_data_predictor import RandomPoolDataHandler
from utility import logger


def extend_mine_data_by_prediction(how_many):
    """
    Predicts new block information and appends it to the mine database table
    :param how_many: how many new rows to predict
    :return: None
    """
    logger("prediction").info("Predicting {0} new records and "
                              "adding them to the mine database main table".format(how_many))
    '''
    Create database elements if they do not exist
    '''
    pass


logger("prediction").info("Setting up the predictor")


def populate_db_with_random():
    block_data.switch_to_temporary_copy(which_db="pools")
    data_handler = RandomPoolDataHandler([Pool("A", 0.2), Pool("B", 0.25), Pool("C", 0.25), Pool("D", 0.3)],
                                         [i for i in range(2, 101)])
    data_handler.initialize()
    data_handler.update_pools_db_with_occurrences()
    data_handler.update_luck_tables()
    # block_data.switch_to_main_copy(save_temporary_copy=True, remove_temporary_copy=True, which_db="pools")
    block_data.print_all_pools_data()

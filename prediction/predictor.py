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


def populate_db_with_random(pools, luck_average_windows, assessment_average_windows):
    block_data.switch_to_temporary_copy(which_db="pools")
    data_handler = RandomPoolDataHandler(
        pools,
        luck_average_windows,
        assessment_average_windows)
    data_handler.initialize()
    data_handler.update_pools_db_with_occurrences()
    # data_handler.update_luck_tables()
    block_data.switch_to_main_copy(save_temporary_copy=True, remove_temporary_copy=True, which_db="pools")
    # block_data.print_all_pools_data()

def create_data_handler(pools, luck_average_windows, assessment_average_windows):
    data_handler = RandomPoolDataHandler(
        pools,
        luck_average_windows,
        assessment_average_windows)
    data_handler.initialize()
    return data_handler


def export_pool_data_points_for_training(data_handler, pool_name, filter_by_block_occurrence=False):
    p = data_handler.get_pool_by_name(pool_name)
    x = data_handler.export_prediction_x(p.id, filter_by_block_occurrence=filter_by_block_occurrence)
    y = data_handler.export_assessments_y(p.id, filter_by_block_occurrence=filter_by_block_occurrence)
    # print(str(x))
    return x, y


def export_pool_block_occurrences(data_handler, pool_name):
    p = data_handler.get_pool_by_name(pool_name)
    return data_handler.export_block_occurrence_timestamps(p.id)


def get_nth_column(x, n):
    return [row[n] for row in x]
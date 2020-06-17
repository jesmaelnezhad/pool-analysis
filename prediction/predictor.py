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

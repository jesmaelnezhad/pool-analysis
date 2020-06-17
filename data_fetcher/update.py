from utility import logger


def update():
    """
    Updates the mine data by fetching new records from the pool web API
    :return: None
    """
    logger("data_fetcher").info("Updating the data from the pool web API")

from data_fetcher import get_last_block_no_seen, update_last_block_no_seen
from utility import logger


def update():
    """
    Updates the mine data by fetching new records from the pool web API
    :return: None
    """
    logger("data_fetcher").info("Updating the data from the pool web API")
    last_block_no = get_last_block_no_seen()
    # FIXME Update logic
    update_last_block_no_seen()

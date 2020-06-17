from utility import logger

logger("data_fetcher").info("Setting up the data fetcher")

LAST_BLOCK_NO_SEEN = None
# TODO Find the most recent block no seen


def update_last_block_no_seen():
    # TODO read the block no of the most recent block found in the database
    pass


def get_last_block_no_seen():
    return LAST_BLOCK_NO_SEEN

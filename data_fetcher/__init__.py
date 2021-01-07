from utility import logger
from db import block_data


def get_last_block_no_seen():
    LAST_BLOCK_NO_SEEN = block_data.get_last_block_no()
    # print("Return No Seen as ", LAST_BLOCK_NO_SEEN)
    return LAST_BLOCK_NO_SEEN


def get_slush_account_token():
    token = "ytvXZLN3nCHwYWlz"
    return token


logger("data_fetcher").info("Setting up the data fetcher")

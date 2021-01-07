from data_fetcher import get_last_block_no_seen, get_slush_account_token
from utility import logger
import requests
import json
from db import block_data as mine_database


def update():
    """
    Updates the mine data by fetching new records from the pool web API
    :return: None
    """
    logger("data_fetcher").info("Updating the data")
    
    # get last block data from database
    last_block_no = get_last_block_no_seen()
    
    # use slushpool api to update database
    result = update_with_api(last_block_no)
    print(result)
    
    # print("last block No = ", last_block_no)
    # FIXME Update logic


def update_with_api(last_block_no):
    """
    Gets blocks data from slushpool API, if last block is included in 15 blcok data, update database accordingly, 
    otherwise the last block value to be updated using web scrapping
    :return: last block value to be scrapped, if none required returns 0
    """
    # temporary
    last_block_no = 641547
    # using slushpool api, get data of last 15 blocks
    url = "https://slushpool.com/stats/json/btc/"
    token = get_slush_account_token()
    headerVar = {"X-SlushPool-Auth-Token" : token}
    result = requests.get(url, headers=headerVar)
    data = result.json()
    
    # parse json data, check if last_block_no is included in data retrieved
    data = data["btc"]
    blocks = data["blocks"]
    isIncluded = False
    blockNoList = []
    for key in blocks.keys():
        if last_block_no == int(key):
            isIncluded = True
        blockNoList.append(int(key))
    
    # take action based on the fact if last block No. exist in api response or not
    if (not isIncluded):
        return (min(blockNoList))
    else:
        # add data retrieved from API to the database
        logger("data_fetcher").info("Updating the data from pool web API")
        mine_database.switch_to_temporary_copy()
        for key in blocks.keys():
            if int(key) > last_block_no:
               blockData = blocks[key]
               dbRecord = dict()
               dbRecord["date_found"] = blockData["date_found"]
               dbRecord["duration"] = blockData["mining_duration"]
               dbRecord["hash_rate"] = blockData["pool_scoring_hash_rate"]
               dbRecord["difficulty"] = 111111111111111
               # FIXME get block difficulty
               dbRecord["block_no"] = int(key)
               dbRecord["block_value"] = blockData["value"]
               print(dbRecord)
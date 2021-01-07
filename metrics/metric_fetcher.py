import requests
import json


def get_slush_account_token():
    token = "ytvXZLN3nCHwYWlz"
    return token

def get_slushpool_data():
    # request data from slushpool api
    token = get_slush_account_token()
    headerVar = {"X-SlushPool-Auth-Token" : token}
    url = "https://slushpool.com/stats/json/btc/"
    result = requests.get(url, headers=headerVar)
    data = result.json()
    data = data["btc"]
    
    # prepare a dictionary as function output including usable data from api
    result = dict()
    result["slush_luck_b10"] = data["luck_b10"]
    result["slush_luck_b50"] = data["luck_b50"]
    result["slush_luck_b250"] = data["luck_b250"]
    scoringHash = data["pool_scoring_hash_rate"]
    hashUnit = data["hash_rate_unit"]
    # reported hash rate unit is Ph/s
    convFact = 1
    if hashUnit == "Gh/s":
        convFact = 1e6
    elif hashUnit == "Th/s":
        convFact = 1e3
    elif hashUnit == "Eh/s":
        convFact = 1e-3
    result["slush_pool_scoring_hash_rate"] = scoringHash / convFact
    result["slush_pool_active_workers"] = data["pool_active_workers"]
    result["slush_round_probability"] = data["round_probability"]
    result["slush_round_duration"] = data["round_duration"]
    
    # result
    return result
    
def get_slush_user_data():
    # request data from slushpool api
    token = get_slush_account_token()
    headerVar = {"X-SlushPool-Auth-Token" : token}
    url = "https://slushpool.com/accounts/profile/json/btc/"
    result = requests.get(url, headers=headerVar)
    data = result.json()
    data = data["btc"]
    
    # prepare dictionary as function output including usable data from api
    # hash rate data unit is Th/s
    hashUnit = data["hash_rate_unit"]
    convFact = 1
    if hashUnit == "Gh/s":
        convFact = 1e3
    elif hashUnit == "Ph/s":
        convFact = 1e-3
    elif hashUnit == "Eh/s":
        convFact = 1e-6
    result = dict()
    result["slush_user_hash_rate_5m"] = data["hash_rate_5m"] / convFact
    result["slush_user_hash_rate_60m"] = data["hash_rate_60m"] / convFact
    result["slush_user_hash_rate_24h"] = data["hash_rate_24h"] / convFact
    result["slush_user_hash_rate_scoring"] = data["hash_rate_scoring"] / convFact

    # result 
    return result

def get_all_pools_data():
    # request data from BTC.com
    url = "https://btc.com/stats/api/realtime/poolHashrate"
    params = dict()
    params["count"] = 15
    header = dict()
    header["User-agent"] = "your bot 0.1"
    result = requests.get(url, params=params, headers=header)
    data = result.json()["data"]
    print(data.keys())
    
    # prepare dictionary for each pool, then add it to general dictionary for all pools
    allPools = dict()
    realData = data["pools_hashrate"]
    for dataDict in realData:
        # get pool name
        poolName = dataDict["relayed_by"]
        # check if main dictionary contains this pool name
        if not poolName in allPools.keys():
            allPools[poolName] = dict()
        # add block count to dictionary
        allPools[poolName]["blocks_count"] = dataDict["count"]
        # add pool share to dictionary
        allPools[poolName]["pool_share"] = dataDict["pool_share"]
        # add hash share to dictionary
        allPools[poolName]["hash_share"] = dataDict["hash_share"]
        # add real hash rate (Ph/s) to dictionary
        hashUnit = dataDict["hashrate_unit"]
        factor = 1
        if hashUnit == "E":
            factor = 1e3
        allPools[poolName]["real_hash_rate"] = dataDict["hashrate"] * factor
        # add 24h difference to dictionary
        allPools[poolName]["diff_24h"] = dataDict["diff_24h"]
        # add 3-days lucky to dictionary
        allPools[poolName]["3days_lucky"] = dataDict["lucky"]
        # add block count current to max to dictionary
        allPools[poolName]["block_count_cur2max"] = dataDict["cur2max_percent"]
    computedData = data["pools_compute_hashrate"]
    for dataDict in computedData:
        # get pool name
        poolName = dataDict["relayed_by"]
        # check if main dictionary contains this pool name
        if not poolName in allPools.keys():
            allPools[poolName] = dict()
        # add computed hash rate to dictionary
        hashUnit = dataDict["hashrate_unit"]
        factor = 1
        if hashUnit == "E":
            factor = 1e3
        allPools[poolName]["compute_hash_rate"] = dataDict["hashrate"] * factor
        # add computed cur2max value (not understood yet)
        allPools[poolName]["compute_cur2max_percent"] = dataDict["cur2max_percent"]
    
    # output
    return allPools

if __name__ == "__main__":
    # print(get_slushpool_data())
    # print(get_slush_user_data())
    print(get_all_pools_data()["SlushPool"])
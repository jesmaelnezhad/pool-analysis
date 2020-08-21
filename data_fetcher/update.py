from data_fetcher import get_last_block_no_seen, get_slush_account_token
from utility import logger
import requests
import json
from db import block_data as mine_database
import datetime, time
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys


def update():
    """
    Updates the mine data by fetching new records from the pool web API
    :return: None
    """
    logger("data_fetcher").info("Updating the data")
    
    # get last block data from database
    # last_block_no = get_last_block_no_seen()
    last_block_no = 0
    
    # use slushpool api to update database
    result = update_with_api(last_block_no)
    print(result)
    
    # print("last block No = ", last_block_no)
    # FIXME Update logic


def update_with_api(last_block_no):
    """
    Gets blocks data from slushpool API, if last block is included in 15 blcok data, update database accordingly, 
    otherwise the last block value to be updated using web scrapping
    :last_block_no: last block no included in database
    :return: None
    """
    # temporary
    last_block_no = 322741
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

    # add data retrieved from API to the database
    logger("data_fetcher").info("Updating the data from pool web API")
    mine_database.switch_to_temporary_copy()
    for key in blocks.keys():
        if int(key) > last_block_no:
           print(int(key))
           blockData = blocks[key]
           dbRecord = dict()
           dbRecord["date_found_unix"] = blockData["date_found"]
           timestamp = datetime.datetime.fromtimestamp(blockData["date_found"])
           dbRecord["found_at_date"] = timestamp.strftime('%Y-%m-%d')
           dbRecord["found_at_time"] = timestamp.strftime('%H:%M:%S')
           dbRecord["duration"] = blockData["mining_duration"]
           dbRecord["hash_rate"] = int(blockData["pool_scoring_hash_rate"] / 1000)
           dbRecord["block_no"] = int(key)
           dbRecord["block_value"] = blockData["value"]
           mine_database.insert_raw_data_dict(dbRecord)
    
    # take action based on the fact if last block No. exist in api response or not
    if (not isIncluded):
        update_with_scrap(last_block_no, min(blockNoList))
        
    mine_database.switch_to_main_copy(True, True)
    
    
    
def update_with_scrap(last_block_no, until_block_no):
    """
    Scrap data from website and add them to database
    :last_block_no: Last block no that includes in database. All blocks after that shall be scrapped
    :until_block_no: All block No.s below this value shall be scrapped
    :return: None
    """
    # initiate web driver
    firefoxProfile = webdriver.FirefoxProfile()
    firefoxProfile.set_preference('permissions.default.stylesheet', 2)
    firefoxProfile.set_preference('permissions.default.image', 2)
    firefoxProfile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so','false')
    firefoxProfile.set_preference("http.response.timeout", 20)
    firefoxProfile.set_preference("dom.max_script_run_time", 10)
    driver = webdriver.Firefox(firefox_profile=firefoxProfile)
    url = "https://slushpool.com/stats/blocks/?c=btc"
    driver.get(url)
    
    # wait for page to completely be loaded
    xpath = "//table[@id='block-history']/tbody/tr[2]"
    loaded = check_page_loaded(driver, "xpath", xpath, 30)
    if not loaded:
        return
    
    # update data in temporary copy
    mine_database.switch_to_temporary_copy()
    
    # navigate to a page with last block no 
    while True:
        blockNoList = get_page_block_nos(driver)
        if (last_block_no in blockNoList):
            break
        isLoaded = page_navigate(driver, True)
        if not isLoaded:
            break
    
    # scrap page data, check for until_block_no, if found break, else update the database
    while True:
        scrap_page_data(driver, last_block_no, until_block_no)
        blockNoList = get_page_block_nos(driver)
        if (until_block_no in blockNoList):
            break
        isLoaded = page_navigate(driver, False)
    # input('x = ')
    
        
def scrap_page_data(driver, last_block_no, until_block_no):
    """
    scrap data in a page, check block no to be between two parameters and update database accordingly
    :driver: web driver to be used for data scrapping
    :last_block_no: Last block no that includes in database. All blocks after that shall be scrapped
    :until_block_no: All block No.s below this value shall be scrapped
    :return: True, if last_block_no is found inside this page
    """

    # get table body
    xpath = "//table[@id='block-history']/tbody"
    dataTbl = driver.find_element_by_xpath(xpath)
    
    # loop on every row data and scrap data
    xpath = "tr[@tabindex='-1']"
    totalDataRows = dataTbl.find_elements_by_xpath(xpath)
    for dataRow in totalDataRows:
        blockNo = (int)(dataRow.find_element_by_xpath("td[6]/a[@role='button']").get_attribute("innerText"))
        if blockNo <= last_block_no:
            continue
        if blockNo >= until_block_no:
            continue
            
        blockFoundTime = dataRow.find_element_by_xpath("td[1]/span[1]").get_attribute("innerText")
        day = int(blockFoundTime[0:2])
        month = int(blockFoundTime[3:5])
        year = int(blockFoundTime[6:10])
        hour = int(blockFoundTime[11:13])
        minute = int(blockFoundTime[14:16])
        print(day, month, year, hour, minute)
        
        timestamp = time.mktime(datetime.datetime(year, month, day, hour, minute).timetuple()) + 3600 * 3.5
        dbRecord = dict()
        dbRecord["date_found_unix"] = timestamp
        timestamp = datetime.datetime.fromtimestamp(timestamp)
        dbRecord["found_at_date"] = timestamp.strftime('%Y-%m-%d')
        dbRecord["found_at_time"] = timestamp.strftime('%H:%M:%S')
        duration = dataRow.find_element_by_xpath("td[2]").get_attribute("innerText")
        durationHour = int(duration[0:2])
        durationMin = int(duration[3:5])
        durationSec = int(duration[6:8])
        dbRecord["duration"] = durationHour * 3600 + durationMin * 60 + durationSec
        hashRateUnit = dataRow.find_element_by_xpath("td[3]/span/span[2]").get_attribute("innerText")
        factor = 1e6
        if (hashRateUnit == "Ph/s"):
            factor = 1e3
        dbRecord["hash_rate"] = (float)(dataRow.find_element_by_xpath("td[3]/span/span[1]").get_attribute("innerText")) * factor
        dbRecord["block_no"] = blockNo
        dbRecord["block_value"] = dataRow.find_element_by_xpath("td[7]/span/span[@class='value']").get_attribute("innerText")
        mine_database.insert_raw_data_dict(dbRecord)
        
        
def check_page_loaded(driver, propertyName, propertyValue, timeOut):
    pageLoaded = True
    try:
        property = By.NAME
        if propertyName == "id":
            property = By.ID
        elif propertyName == "class":
            property = By.CLASS_NAME
        elif propertyName == "name":
            property = By.NAME
        elif propertyName == "xpath":
            property = By.XPATH
        elementPresent = EC.presence_of_element_located((property, propertyValue))
        WebDriverWait(driver, timeOut).until(elementPresent)
    except TimeoutException:
        pageLoaded = False
        # print("Timed out waiting for page to load")
    return pageLoaded

def page_navigate(driver, higherNo = True):
    """
    load next page means higher page number
    :driver: web driver to be used for data scrapping
    :return: True, If successfully load new page
    """
    
    # assume that page is loaded completely
    # get id of the first and last row of data, then set new id to be found
    xpath = "//table[@id='block-history']/tbody/tr[@tabindex='-1']"
    allRows = driver.find_elements_by_xpath(xpath)
    firstEle = allRows[0]
    lastEle = allRows[-1]
    firstId = (int)(firstEle.find_element_by_xpath("td[1]").get_attribute("title")[1:])
    lastId = (int)(lastEle.find_element_by_xpath("td[1]").get_attribute("title")[1:])
    newId = lastId - 1
    if not higherNo:
        newId = firstId + 1
    
    # click on next page button and wait for new element with new id
    xpath = "//*[contains(@class, 'angle-right-thin')]/../.."
    if not higherNo:
        xpath = "//*[contains(@class, 'angle-left-thin')]/../.."
    nextButt = driver.find_element_by_xpath(xpath)
    nextButt.click()
    xpath = "//table[@id='block-history']/tbody/tr[1]/td[@title='#{0}']".format(newId)
    loaded = check_page_loaded(driver, "xpath", xpath, 15)
    return(loaded)

def get_page_block_nos(driver):
    """
    get a list of all block no whitin a page
    :driver: web driver to be used for data scrapping
    :return: list of integers of block No list
    """
    # get table body
    xpath = "//table[@id='block-history']/tbody"
    dataTbl = driver.find_element_by_xpath(xpath)
    
    # loop on every row data and scrap data
    xpath = "tr[@tabindex='-1']"
    totalDataRows = dataTbl.find_elements_by_xpath(xpath)
    blockNoList = []
    for dataRow in totalDataRows:
        blockNo = dataRow.find_element_by_xpath("td[6]/a[@role='button']").get_attribute("innerText")
        blockNoList.append(int(blockNo))
    
    return(blockNoList)




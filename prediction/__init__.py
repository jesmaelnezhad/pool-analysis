from random import randint

from utility import logger
from db import block_data
from datetime import datetime


########################
TIME_CONSTANT_10minutes = 60 * 10
TIME_CONSTANT_1months  = 60 * 60 * 24 * 30
########################


class Pool:
    def __init__(self, name, share, pool_id=None):
        self.name = name
        self.share = share
        self.id = pool_id
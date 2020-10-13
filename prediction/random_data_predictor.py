from datetime import datetime
from random import randint, seed

from db import block_data
from prediction import TIME_CONSTANT_1months, TIME_CONSTANT_10minutes, Pool
from utility import logger


def generate_luck_table_name(pool, delta_window):
    return "luck_b{}_w{}".format(pool.id, delta_window)


def get_now_timestamp():
    return int(datetime.now().timestamp())


class RandomPoolDataHandler:
    def __init__(self, pools, average_windows,
                 all_time_range=(3 * TIME_CONSTANT_1months),
                 step_size=TIME_CONSTANT_10minutes,
                 random_granularity=1000):
        self.all_time_range = all_time_range
        self.step_size = step_size
        self.random_granularity = random_granularity
        self.pools = pools
        self.average_windows = average_windows

    def initialize(self, truncate_luck_tables=False):
        # init the pools and occurrences tables first if not initialized
        block_data.init_pools_data_tables()
        # make sure database is consistent with the in memory list of pools
        self.initialize_pools()
        # initialize the luck tables if not initialized yet
        for p in self.pools:
            for w in self.average_windows:
                table_name = generate_luck_table_name(p, w)
                block_data.init_pools_luck_tables(table_name)
                if truncate_luck_tables:
                    block_data.truncate_table(table_name, which_db="pools")

    def update_pools_db_with_occurrences(self):
        now_timestamp = get_now_timestamp()
        start_timestamp = now_timestamp - self.all_time_range
        # if there are occurrences inserted from before, continue on top of that
        latest_timestamp = block_data.get_latest_pool_block_occurrence_timestamp()
        if latest_timestamp is not None:
            start_timestamp = latest_timestamp + self.step_size
        seed(int(datetime.now().timestamp()))
        block_no = 10000
        while start_timestamp < now_timestamp:
            new_random = randint(1, self.random_granularity)
            matching_pool = None
            for p in self.pools:
                if self.does_pool_match(p, new_random):
                    matching_pool = p
                    break
            block_data.insert_pool_block_occurrence(start_timestamp, matching_pool.id, block_no)
            start_timestamp += self.step_size
            block_no += 1

    def update_luck_tables(self):
        for p in self.pools:
            for w in self.average_windows:
                self.update_luck_table(p, w)
                rows = block_data.get_columns(["*"],
                                              table_name=generate_luck_table_name(p, w),
                                              which_db="pools",
                                              limit=100000)
                for r in rows:
                    logger("update_luck_table_{}_{}".format(p.id, w)).info(str(r))

    # ############# Private methods ############ #
    def does_pool_match(self, pool, new_random):
        start = 1
        end = -1
        for p in self.pools:
            if pool.id == p.id:
                end = start + int(p.share * self.random_granularity)
                break
            start += int(p.share * self.random_granularity)
        if end == -1:
            logger("random-data-generator").warn("Generated random value did not fit in any pools!!!!")
        return start <= new_random <= end

    def initialize_pools(self):
        db_pools = block_data.get_pools()
        # check all db pools exist in memory
        self.sync_in_memory_pools(db_pools)
        # check all memory pools exist in db
        self.sync_db_pools(db_pools)

    def sync_db_pools(self, db_pools):
        for p in self.pools:
            pool_found = False
            for pool_info in db_pools:
                if p.name == pool_info[0]:
                    pool_found = True
                    break
            if not pool_found:
                p.id = block_data.insert_pool(p.name, p.share)

    def sync_in_memory_pools(self, db_pools):
        for pool_info in db_pools:
            name = pool_info[0]
            share = pool_info[1]
            pool_id = pool_info[2]
            pool_found = False
            for p in self.pools:
                if name == p.name:
                    p.id = pool_id
                    p.share = share
                    pool_found = True
                    break
            if not pool_found:
                self.pools.append(Pool(name, share, pool_id))

    def update_luck_table(self, pool, delta_window):
        time_interval = delta_window * self.step_size
        table_name = generate_luck_table_name(pool, delta_window)
        start_timestamp = block_data.get_latest_pool_block_occurrence_timestamp(return_oldest=True)
        if start_timestamp is None:
            return
        latest_window_start = block_data.get_latest_luck_window_start_timestamp(table_name)
        if latest_window_start is not None:
            start_timestamp = latest_window_start + self.step_size
        now_timestamp = get_now_timestamp()
        while start_timestamp < now_timestamp:
            end = start_timestamp
            begin = end - time_interval
            delta_count = block_data.get_total_block_occurrences_of_pool(pool.id, begin, end)
            expected_total = self.get_total_expected_number_of_occurrences(time_interval)
            expected_share = expected_total * pool.share
            luck = delta_count / expected_share
            block_data.insert_pool_luck(table_name, end, luck)
            start_timestamp += self.step_size

    def get_total_expected_number_of_occurrences(self, time_interval):
        return (time_interval * 1.0) / self.step_size

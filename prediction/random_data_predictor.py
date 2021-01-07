from datetime import datetime
from random import randint, seed

from db import block_data
from prediction import TIME_1_MONTHS, TIME_10_MINUTES, Pool
from utility import logger

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FeatureUnion

from skits.feature_extraction import AutoregressiveTransformer
from skits.pipeline import ForecasterPipeline
from skits.preprocessing import ReversibleImputer, HorizonTransformer
import numpy as np


def generate_luck_table_name(delta_window):
    return "luck_w{}".format(delta_window)


def generate_assessment_table_name(delta_window):
    return "assessment_w{}".format(delta_window)


def get_now_timestamp():
    return int(datetime.now().timestamp())


class PoolStatCollector:
    def __init__(self, step_size, average_windows, mode="luck"):
        """

        :param step_size:
        :param average_windows:
        :param mode: luck or assessment / luck is stats on past, assessment is stats on future
        """
        self.mode = mode
        self.step_size = step_size
        self.max_window = average_windows[0]
        self.valid_timestamps = []
        for w in average_windows:
            if w > self.max_window:
                self.max_window = w

    def add_point_and_update(self, new_timestamp):
        self.valid_timestamps.append(new_timestamp)
        tmp_valid_timestamps = self.valid_timestamps
        self.valid_timestamps = []
        from_idx = 0
        for i, ts in enumerate(tmp_valid_timestamps):
            if self.mode == "luck":
                if new_timestamp - ts < self.max_window * self.step_size:
                    from_idx = i
                    break
            elif self.mode == "assessment":
                if ts - new_timestamp < self.max_window * self.step_size:
                    from_idx = i
                    break
        self.valid_timestamps = tmp_valid_timestamps[from_idx:]

    def get_latest_total_count_for_all_windows(self, windows, last_timestamp):
        if len(self.valid_timestamps) == 0:
            return [0 for w in windows]

        from_idx = len(self.valid_timestamps) - 1
        counts = [0 for w in windows]
        while from_idx > 0:
            range_from_end = None
            if self.mode == "luck":
                range_from_end = last_timestamp - self.valid_timestamps[from_idx]
            elif self.mode == "assessment":
                range_from_end = self.valid_timestamps[from_idx] - last_timestamp
            any_window_still_active = False
            for wi, window in enumerate(windows):
                if range_from_end < window * self.step_size:
                    counts[wi] += 1
                    any_window_still_active = True
            if not any_window_still_active:
                break
            from_idx -= 1

        return counts


class RandomPoolDataHandler:
    def __init__(self, pools, average_windows, assessment_windows,
                 all_time_range=6 * TIME_1_MONTHS,
                 step_size=TIME_10_MINUTES,
                 random_granularity=10000):
        self.all_time_range = all_time_range
        self.step_size = step_size
        self.random_granularity = random_granularity
        self.pools = pools
        self.pool_stats = {}
        for p in self.pools:
            self.pool_stats[p] = PoolStatCollector(self.step_size, average_windows)
        self.pool_assessment_stats = {}
        for p in self.pools:
            self.pool_assessment_stats[p] = PoolStatCollector(self.step_size, assessment_windows, mode="assessment")
        self.average_windows = average_windows
        self.assessment_windows = assessment_windows
        #
        self.prediction_horizon_seconds_max = 48 * 6 * TIME_10_MINUTES
        self.prediction_history_length_seconds = 42 * 24 * 6 * TIME_10_MINUTES
        self.newest_ts = 0
        self.oldest_ts = 0
        self.x_data_extractor = None
        self.block_occurrence_extractor = None

    def initialize(self, truncate_tables=False):
        # init the pools and occurrences tables first if not initialized
        block_data.init_pools_data_tables()
        # make sure database is consistent with the in memory list of pools
        self.initialize_pools()
        # initialize the luck tables if not initialized yet
        for w in self.average_windows:
            table_name = generate_luck_table_name(w)
            block_data.init_multi_pool_table(table_name, [p.id for p in self.pools], column_prefix="luck")
            if truncate_tables:
                block_data.truncate_table(table_name, which_db="pools")
        for w in self.assessment_windows:
            table_name = generate_assessment_table_name(w)
            block_data.init_multi_pool_table(table_name, [p.id for p in self.pools], column_prefix="assessment")
            if truncate_tables:
                block_data.truncate_table(table_name, which_db="pools")
        print(str(block_data.get_list_of_table_names(which_db="pools")))

    def get_pool_by_name(self, pool_name):
        for p in self.pools:
            if p.name == pool_name:
                return p
        return None

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
            logger("random-data-generator").debug("Block # {} processed.".format(block_no))
            new_random = randint(1, self.random_granularity)
            matching_pool = None
            for p in self.pools:
                if self.does_pool_match(p, new_random):
                    matching_pool = p
                    break
            # Update occurrence
            block_data.insert_pool_block_occurrence(start_timestamp, matching_pool.id, block_no)
            self.pool_stats[matching_pool].add_point_and_update(start_timestamp)

            # Update luck tables
            self.update_luck_tables_after_one_step(start_timestamp)

            start_timestamp += self.step_size
            block_no += 1
        # Update assessments
        all_block_occurrences = block_data.get_all_block_occurrences()
        for row in all_block_occurrences:
            matching_pool = None
            for p in self.pools:
                if p.id == row[1]:
                    matching_pool = p
                    break
            self.pool_assessment_stats[matching_pool].add_point_and_update(row[0])
            # Update assessment tables
            self.update_luck_tables_after_one_step(row[0], mode="assessments")

    def update_luck_tables_after_one_step(self, timestamp, mode="lucks"):
        pools_window_delta_counts = {}
        windows = self.average_windows
        for p in self.pools:
            if mode == "lucks":
                pools_window_delta_counts[p] = self.pool_stats[p].get_latest_total_count_for_all_windows(
                    self.average_windows, timestamp)
            elif mode == "assessments":
                pools_window_delta_counts[p] = self.pool_assessment_stats[p].get_latest_total_count_for_all_windows(
                    self.assessment_windows, timestamp)
                windows = self.assessment_windows

        for i, w in enumerate(windows):
            expected_total = self.get_total_expected_number_of_occurrences(w * self.step_size)
            pool_lucks = {}
            for p in self.pools:
                delta_count = pools_window_delta_counts[p][i]
                expected_share = expected_total * p.share
                pool_lucks[p.id] = delta_count / expected_share
            if mode == "lucks":
                table_name = generate_luck_table_name(w)
                block_data.insert_pool_lucks(table_name, timestamp, pool_lucks)
                # print(str(timestamp) + " / lucks / " + str(pool_lucks))
            elif mode == "assessments":
                table_name = generate_assessment_table_name(w)
                block_data.insert_pool_assessments(table_name, timestamp, pool_lucks)
                # print(str(timestamp) + " / assessments / " + str(pool_lucks))

    def update_luck_tables(self):
        for w in self.average_windows:
            self.update_luck_table(w)
            # rows = block_data.get_columns(["*"],
            #                               table_name=generate_luck_table_name(w),
            #                               which_db="pools",
            #                               limit=100000)
            # for r in rows:
            #     logger("update_luck_table_{}".format(w)).info(str(r))

    def set_main_configs_for_input_data_preparation(self, no_days_offset=0):
        """
        Sets the following configurations to be used in training a prediction model:
        oldest_ts : the timestamp of the oldest data point that can be used in model fitting
        newest_ts : the timestamp of the most recent data point that can be used in model fitting
        train_percentage: the percentage of data points to be used in training (the rest until the
        newest will be used in testing)
        x_query: the query which returns a row for each timestamp which is the dimensions of the X data points
        :return:
        """
        table_name = generate_luck_table_name(self.average_windows[0])
        latest_data_ts = block_data.get_latest_luck_window_start_timestamp(table_name)
        self.newest_ts = block_data.get_latest_luck_window_start_timestamp(table_name,
                                                                           earlier_than=(
                                                                                   latest_data_ts -
                                                                                   self.prediction_horizon_seconds_max -
                                                                                   no_days_offset * 24 * 6 * TIME_10_MINUTES))
        self.oldest_ts = block_data.get_latest_luck_window_start_timestamp(table_name,
                                                                           earlier_than=(
                                                                                   self.newest_ts -
                                                                                   self.prediction_history_length_seconds))

        self.x_data_extractor = block_data.get_pool_luck_values
        self.block_occurrence_extractor = block_data.get_all_block_occurrences

    # ############# Private methods ############ #
    def does_pool_match(self, pool, new_random):
        start = 0
        end = -1
        for p in self.pools:
            if pool.id == p.id:
                end = start + int(p.share * self.random_granularity)
                break
            start += int(p.share * self.random_granularity)
        if end == -1:
            logger("random-data-generator").warn("Generated random value did not fit in any pools!!!!")
        return start < new_random <= end

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

    def get_pool_luck(self, pool, from_ts, time_interval):
        delta_count = block_data.get_total_block_occurrences_of_pool(pool.id, from_ts, from_ts + time_interval)
        total_expected = self.get_total_expected_number_of_occurrences(time_interval)
        expected_share = total_expected * pool.share
        return delta_count * 1.0 / expected_share

    def update_luck_table(self, delta_window):
        time_interval = delta_window * self.step_size
        start_timestamp = block_data.get_latest_pool_block_occurrence_timestamp(return_oldest=True)
        table_name = generate_luck_table_name(delta_window)
        if start_timestamp is None:
            return
        latest_window_start = block_data.get_latest_luck_window_start_timestamp(table_name)
        if latest_window_start is not None:
            start_timestamp = latest_window_start + self.step_size
        now_timestamp = get_now_timestamp()
        while start_timestamp < now_timestamp:
            end = start_timestamp
            begin = end - time_interval
            expected_total = self.get_total_expected_number_of_occurrences(time_interval)
            pool_lucks = {}
            for p in self.pools:
                delta_count = block_data.get_total_block_occurrences_of_pool(p.id, begin, end)
                expected_share = expected_total * p.share
                pool_lucks[p.id] = delta_count / expected_share
            block_data.insert_pool_lucks(table_name, end, pool_lucks)
            start_timestamp += self.step_size

    def get_total_expected_number_of_occurrences(self, time_interval):
        return (time_interval * 1.0) / self.step_size

    def export_prediction_x(self, pool_id, filter_by_block_occurrence=False, round_to_n_decimal_points=5):
        """
        :return: List of data points to be used in training
        """
        table_names = [generate_luck_table_name(dw) for dw in self.average_windows]
        return self.x_data_extractor(pool_id, table_names, self.oldest_ts, self.newest_ts,
                                     filter_by_block_occurrence=filter_by_block_occurrence,
                                     round_to_n_decimal_points=round_to_n_decimal_points)

    def export_block_occurrence_timestamps(self, pool_id):
        return self.block_occurrence_extractor(pool_id=pool_id)

    def export_assessments_y(self, pool_id, filter_by_block_occurrence=False, round_to_n_decimal_points=5):
        """
        :return: List of data points to be used in training
        """
        table_names = [generate_assessment_table_name(dw) for dw in self.assessment_windows]
        return self.x_data_extractor(pool_id, table_names, self.oldest_ts, self.newest_ts, column_prefix="assessment",
                                     filter_by_block_occurrence=filter_by_block_occurrence,
                                     round_to_n_decimal_points=round_to_n_decimal_points)

from db.block_data import COL_DURATION, COL_BLOCK_NO, COL_BLOCK_VALUE
from utility import logger
import math

TICK_DURATION_SECONDS = 4
VALID_PRICE_CHANGE_GAP = 10 * 60
LIMIT_CHANGE_DELAY = 10 * 60

COST_PER_TERA_HASH_PER_HOUR = 0.0000084 / 24
COST_PER_TERA_HASH_PER_TICK = COST_PER_TERA_HASH_PER_HOUR / (60 * 60 / TICK_DURATION_SECONDS)

REWARD_PER_SOLVE_PER_TERA_HASH_PER_BLOCK_VALUE = 1 / (4.53 * 1000 * 1000)

TIME_UNIT_HOURS = "hours"
TIME_UNIT_MINUTES = "minutes"
TIME_UNIT_SECONDS = "seconds"


def hours(h):
    return h * 60 * 60


def minutes(m):
    return m * 60


TICK_INFO_NEEDED_COLUMNS = [COL_DURATION, COL_BLOCK_NO, COL_BLOCK_VALUE]


class RuntimeTickInfo:
    def __init__(self, block_number, block_value, time_since_block_start_in_seconds, time_since_start_in_seconds,
                 is_block_starting_point=False, is_block_ending_point=False):
        """
        Initializes an object which represents a tick (a tiny step) in the execution of an algorithm
        """
        self.block_number = block_number
        self.block_value = block_value
        self.time_since_block_start_in_seconds = time_since_block_start_in_seconds
        self.time_since_start_in_seconds = time_since_start_in_seconds
        self.is_block_starting_point = is_block_starting_point
        self.is_block_ending_point = is_block_ending_point
        # operation is a list of functions that must be called when we are in that tick
        self.operations = []
        # tags that can be used in the operation
        self.tags = []
        # price and limit changes to be applied in this tick
        self.limit_change = 0

    def __str__(self):
        return "Tick info: block no={0} / time since start={1} / time of block passed={2} " \
               "/ block start={3} / block end={4} / tags={5}".format(self.block_number,
                                                                     self.time_since_start_in_seconds,
                                                                     self.time_since_block_start_in_seconds,
                                                                     self.is_block_starting_point,
                                                                     self.is_block_ending_point,
                                                                     self.tags)

    def add_operation(self, operation):
        """
        Adds a new operation (a function) to be called when we are in that tick
        :param operation: new operation to add
        :return: None
        """
        self.operations.append(operation)

    def add_tag(self, tag):
        """
        Adds a new tag
        :param tag: new tag to add
        :return: None
        """
        self.tags.append(tag)

    def run(self, tester):
        """
        Runs the operations in the current tick. The operation is in the form of a function
        that takes the tester object and the runtime tick info
        :return: None
        """
        for op in self.operations:
            op(tester, self)

    def has_passed_exactly(self, time, time_unit=TIME_UNIT_HOURS):
        """
        Returns whether or not exactly 'time' seconds has passed since the beginning of this block at this tick
        :param time_unit: time unit
        :param time: time to check in seconds
        :return: True if we are in the tick where exactly time seconds has passed since the beginning of the block
        """
        time_in_seconds = None
        if time_unit == TIME_UNIT_HOURS:
            time_in_seconds = hours(time)
        elif time_unit == TIME_UNIT_MINUTES:
            time_in_seconds = minutes(time)
        elif time_unit == TIME_UNIT_SECONDS:
            time_in_seconds = time

        return self.time_since_block_start_in_seconds <= time_in_seconds < self.time_since_block_start_in_seconds + TICK_DURATION_SECONDS


class RuntimeStats:
    def __init__(self):
        self.profit_min = float('inf')
        self.profit_max = float('-inf')

    def update_profit_stats(self, profit):
        if profit < self.profit_min:
            self.profit_min = profit
        if profit > self.profit_max:
            self.profit_max = profit


class Runtime:
    def __init__(self, start, end):
        """
        Initializes the object which holds the execution information of an i to j test where
        i and j are the indices indication the range of data on which the test should be performed
        """
        self.RuntimeTicks = []
        self.current_run_tick_index = 0
        # Nice hash order properties
        self.nice_hash_order_price = 0
        self.last_price_change_time_since_start = None
        self.nice_hash_order_limit = 0

        self.total_cost = 0
        self.total_reward = 0

        self.start, self.end = start, end

        self.statistics = RuntimeStats()

    def nice_hash_api_edit_order_price(self, price):
        """
        Sets the price on our imaginary order on nicehash
        :return: Boolean: True if successful in changing the order, and False if unable to change the order
        (for example if we have changed price less than 10 minutes ago we cannot change it now)
        """
        # Price change
        # Check the last price change is >= 10 minutes ago
        current_time_since_start = self.RuntimeTicks[self.current_run_tick_index].time_since_start_in_seconds
        if self.last_price_change_time_since_start is None:
            self.last_price_change_time_since_start = current_time_since_start
            self.nice_hash_order_price = price
        else:
            if current_time_since_start - self.last_price_change_time_since_start < VALID_PRICE_CHANGE_GAP:
                return False
            else:
                self.last_price_change_time_since_start = current_time_since_start
                self.nice_hash_order_price = price
        return True

    def nice_hash_api_edit_order_limit(self, limit):
        """
        Changes the following ticks so that we reach to the given limit in the following 10 minutes
        :param limit: the requested limit to reach to over 10 minutes
        :return: None
        """
        # 0. Special case: if this is the last tick, limit cannot change.
        if self.current_run_tick_index == len(self.RuntimeTicks) - 1:
            return
        # 1. check the limit after 10 minutes
        involved_ticks = []
        next_tick_time_since_start = self.RuntimeTicks[self.current_run_tick_index + 1].time_since_start_in_seconds
        for tick_index in range(self.current_run_tick_index + 1, len(self.RuntimeTicks)):
            tick_time_since_start = self.RuntimeTicks[tick_index].time_since_start_in_seconds
            if tick_time_since_start - next_tick_time_since_start > LIMIT_CHANGE_DELAY:
                break
            involved_ticks.append(self.RuntimeTicks[tick_index])
        order_limit = self.nice_hash_order_limit
        for involved_tick in involved_ticks:
            order_limit += involved_tick.limit_change
        # 2. calculate total needed limit change and limit_change in every tick if it is going to happen
        #    over 10 minutes
        total_limit_change = limit - order_limit
        number_of_ticks_to_distribute_change = math.ceil(LIMIT_CHANGE_DELAY / TICK_DURATION_SECONDS)
        change_per_tick = total_limit_change * 1.0 / number_of_ticks_to_distribute_change
        # 3. Add the limit_change to happen in the following ticks
        start_index = self.current_run_tick_index + 1
        end_index = min(len(self.RuntimeTicks), start_index + number_of_ticks_to_distribute_change)
        for tick_index in range(start_index, end_index):
            self.RuntimeTicks[tick_index].limit_change += change_per_tick

        return True

    def nice_hash_api_get_order_price(self):
        """
        :returns the price on our imaginary order on nicehash
        """
        return self.nice_hash_order_price

    def nice_hash_api_get_order_limit(self):
        """
        :returns the limit on our imaginary order on nicehash
        """
        return self.nice_hash_order_limit

    def runtime_update_order_on_tick(self, tester, tick_info):
        """
        If any gradual move on price or limit of the order exists in this tick, it is applied in this method.
        :param tester: usually equal to self. The tester object responsible for nice_hash api
        :param tick_info: the information of the current tick at which the function is called
        :return: None
        """
        self.nice_hash_order_limit = self.nice_hash_order_limit + tick_info.limit_change

    def runtime_cost_and_reward(self, tester, tick_info):
        """
        Updates the total cost and reward based on what happens in the given tick, also updates the statistics
        :param tester:
        :param tick_info:
        :return:
        """
        # update reward if a block is finished
        current_limit = tester.r.nice_hash_api_get_order_limit()
        if tick_info.is_block_ending_point:
            self.total_reward += current_limit * REWARD_PER_SOLVE_PER_TERA_HASH_PER_BLOCK_VALUE * tick_info.block_value
        # update cost based on the limit in this tick
        self.total_cost += current_limit * COST_PER_TERA_HASH_PER_TICK
        if self.total_cost != 0 and self.total_reward != 0:
            self.statistics.update_profit_stats((self.total_reward * 100) / self.total_cost)

    def set_tag_at(self, time_since_start, tag):
        """

        :param time_since_start: time_since_start_in_seconds that determines the tick in which the operation should be
        triggered
        :param tag:
        :return: the RuntimeTickInfo object in which the tag is set
        """
        for tick in self.RuntimeTicks:
            if tick.time_since_start_in_seconds <= time_since_start < tick.time_since_start_in_seconds + TICK_DURATION_SECONDS:
                tick.add_tag(tag)
                return tick
        return None


class AlgorithmTester:
    def __init__(self, data, algorithm):
        """
        Initializes the algorithm tester
        :param data: list of tuples (duration, block_id) Note: it is assumed the tuple list is ordered *recent to old*
        and duration is in seconds
        """
        self.AllData = data
        self.Algorithm = algorithm
        # initialize runtime
        self.r = None
        self.Data = None
        self.reset_runtime(0, len(data))

    def __str__(self):
        """
        :return: a string explaining the info of the tester object
        """
        return "Tester object on %d blocks with algorithm %s".format(len(self.Data), str(self.Algorithm))

    def test_range(self, range_size):
        """
        Tests all ranges of size range_size and for each such range yields a tuple like:
        (cost, reward, R/C%, R/C% lowest, R/C% highest)
        :param range_size: the size of slice of all data to take in each test
        :return: a tuple as (cost, reward, R/C%, R/C% lowest, R/C% highest)
        """
        for i in range(len(self.AllData) - range_size):
            self.reset_runtime(i, i + range_size)
            self.prepare_to_run()
            yield self.run()

    def reset_runtime(self, start, end):
        """

        :param start: runtime starting index in all data
        :param end:  runtime ending index in all data
        :return:
        """
        self.r = Runtime(start, end)
        self.Data = self.AllData[start:end]

    def run(self):
        """
        Executes the ticks one by one from the beginning until the end
        :return:
        """
        logger_object = logger("tester")

        # pre tick
        self.Algorithm.pre_ticks(self)
        # run ticks
        for tick_index in range(len(self.r.RuntimeTicks)):
            self.r.current_run_tick_index = tick_index
            tick = self.r.RuntimeTicks[tick_index]
            tick.run(self)
        # post tick
        self.Algorithm.post_ticks(self)

        # print cost and reward
        logger_object.info("Cost: {0:.3f} - Reward: {1:0.3f} - R/C%: [ {3:.3f} >> {2:.3f} << {4:.3f} ]".format(
            self.r.total_cost, self.r.total_reward, (self.r.total_reward * 100) / self.r.total_cost,
            self.r.statistics.profit_min, self.r.statistics.profit_max))
        return self.r.total_cost, self.r.total_reward, (self.r.total_reward * 100) / self.r.total_cost, \
            self.r.statistics.profit_min, self.r.statistics.profit_max

    def prepare_to_run(self):
        """
        Prepares the object for a run
        :return: None
        """
        # Prepare tick info objects
        # Move from the last data point backwards
        for block_data in reversed(self.Data):
            block_duration = block_data[0]
            block_number = block_data[1]
            block_value = block_data[2]
            duration = block_duration
            while duration > 0:
                new_tick = RuntimeTickInfo(block_number, block_value, block_duration - duration,
                                           len(self.r.RuntimeTicks) * TICK_DURATION_SECONDS,
                                           duration == block_duration, duration <= TICK_DURATION_SECONDS)
                # run operation
                new_tick.add_operation(self.r.runtime_update_order_on_tick)
                new_tick.add_operation(self.Algorithm.tick)
                new_tick.add_operation(self.r.runtime_cost_and_reward)
                # Record the tick
                self.r.RuntimeTicks.append(new_tick)
                # Move time forward
                duration -= TICK_DURATION_SECONDS

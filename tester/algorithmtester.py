from utility import logger
import math

TICK_DURATION_SECONDS = 4
VALID_PRICE_CHANGE_GAP = 10 * 60
LIMIT_CHANGE_DELAY = 10 * 60


class RuntimeTickInfo:
    def __init__(self, block_number, time_since_block_start_in_seconds, time_since_start_in_seconds,
                 is_block_starting_point=False, is_block_ending_point=False):
        """
        Initializes an object which represents a tick (a tiny step) in the execution of an algorithm
        """
        self.block_number = block_number
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
               "/ block start={3} / block end={4} / tags={5}".format(self.block_number, self.time_since_start_in_seconds,
                                                        self.time_since_block_start_in_seconds,
                                                        self.is_block_starting_point, self.is_block_ending_point,
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


class AlgorithmTester:
    def __init__(self, data, algorithm):
        """
        Initializes the algorithm tester
        :param data: list of tuples (duration, block_id) Note: it is assumed the tuple list is ordered *recent to old*
        and duration is in seconds
        """
        self.Data = data
        self.Algorithm = algorithm
        self.RuntimeTicks = []
        self.current_run_tick_index = 0
        # Nice hash order properties
        self.nice_hash_order_price = 0
        self.last_price_change_time_since_start = None
        self.nice_hash_order_limit = 0

    def __str__(self):
        """
        :return: a string explaining the info of the tester object
        """
        return "Tester object on %d blocks with algorithm %s".format(len(self.Data), str(self.Algorithm))

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
        if self.current_run_tick_index == len(self.RuntimeTicks)-1:
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

    def run(self):
        """
        Executes the ticks one by one from the beginning until the end
        :return:
        """
        logger_object = logger("tester")

        # pre tick
        self.Algorithm.pre_ticks(self)
        # run ticks
        for tick_index in range(len(self.RuntimeTicks)):
            self.current_run_tick_index = tick_index
            tick = self.RuntimeTicks[tick_index]
            tick.run(self)
        # post tick
        self.Algorithm.post_ticks(self)

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
            duration = block_duration
            while duration > 0:
                new_tick = RuntimeTickInfo(block_number, block_duration - duration,
                           len(self.RuntimeTicks) * TICK_DURATION_SECONDS,
                                           duration == block_duration, duration <= TICK_DURATION_SECONDS)
                # run operation
                new_tick.add_operation(self.runtime_update_order_on_tick)
                new_tick.add_operation(self.Algorithm.tick)
                # Record the tick
                self.RuntimeTicks.append(new_tick)
                # Move time forward
                duration -= TICK_DURATION_SECONDS

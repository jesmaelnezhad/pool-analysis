from utility import logger

TICK_DURATION_SECONDS = 4


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
        # Nice hash order properties
        self.nice_hash_order_price = 0
        self.nice_hash_order_limit = 0

    def __str__(self):
        """
        :return: a string explaining the info of the tester object
        """
        return "Tester object on %d blocks with algorithm %s".format(len(self.Data), str(self.Algorithm))

    def nice_hash_api_edit_order(self, price, limit):
        """
        Sets the price on our imaginary order on nicehash
        :return: None
        """
        self.nice_hash_order_price, self.nice_hash_order_limit = price, limit

    def nice_hash_api_get_order(self):
        """
        :returns the price on our imaginary order on nicehash
        """
        return self.nice_hash_order_price, self.nice_hash_order_limit

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
        for tick in self.RuntimeTicks:
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
                new_tick.add_operation(self.Algorithm.tick)
                # Record the tick
                self.RuntimeTicks.append(new_tick)
                # Move time forward
                duration -= TICK_DURATION_SECONDS

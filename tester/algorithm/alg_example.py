from utility import logger


class Algorithm3Hour:
    def __init__(self):
        self.logger = logger("Algorithm-3-Hour")

    def pre_ticks(self, tester):
        self.logger.info("pre-tick")

    def tick(self, tester, tick_info):
        # Example one:
        # If block started, wake up three hours later and print hello
        interval_length_in_seconds = 3 * 60 * 60
        # Act based on the existing tags in the tick
        if "printhello" in tick_info.tags:
            self.logger.info("tick acting on tag" + str(tick_info))
            self.logger.info("HELLOOOOOOOOOOOOOOOOOOOOOOOO")
        # Act on block start (set tag for three hours)
        if tick_info.is_block_starting_point:
            self.logger.info("tick setting tag" + str(tick_info))
            time_of_next_wakeup = tick_info.time_since_start_in_seconds + interval_length_in_seconds
            # set tag for 3 hours in future
            tester.set_tag_at(time_of_next_wakeup, "printhello")
        # Example two:
        # Act on block end (add 0.01 to price if block ends)
        if tick_info.is_block_ending_point:
            self.logger.info("tick block end " + str(tester.nice_hash_api_get_order()))
            price, limit = tester.nice_hash_api_get_order()
            tester.nice_hash_api_edit_order(price + 0.01, limit)

    def post_ticks(self, tester):
        self.logger.info("post-tick")
        # TODO: cost and benefit should be calculated here
        pass

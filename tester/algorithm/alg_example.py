from tester.algorithmtester import TICK_DURATION_SECONDS
from utility import logger


class Algorithm3Hour:
    def __init__(self):
        self.logger = logger("Algorithm-3-Hour")

    def pre_ticks(self, tester):
        self.logger.info("pre-tick")

    def tick(self, tester, tick_info):
        self.logger.info("Tick time: {0:.3f} total /// {1:.3f} of this block".format(tick_info.time_since_start_in_seconds / (60*60), tick_info.time_since_block_start_in_seconds / (60*60)))
        self.logger.info("     limit : {0:.3f}".format(tester.nice_hash_api_get_order_limit()))
        self.logger.info("     price : {0:.3f}".format(tester.nice_hash_api_get_order_price()))
        # Example one:
        # If block started, wake up three hours later and print hello
        interval_length_in_seconds = 3 * 60 * 60
        # Act based on the existing tags in the tick
        if "lowerlimit" in tick_info.tags:
            self.logger.info("--- tick acting on tag" + str(tick_info))
            # set the limit to 0 if three hours has passed since block start
            if interval_length_in_seconds < tick_info.time_since_block_start_in_seconds + TICK_DURATION_SECONDS:
                tester.nice_hash_api_edit_order_limit(0)
            else:
                time_of_next_wakeup = tick_info.time_since_start_in_seconds + (interval_length_in_seconds - tick_info.time_since_block_start_in_seconds)
                # set tag for 3 hours in future
                tester.set_tag_at(time_of_next_wakeup, "lowerlimit")
        # Act on block start (set tag for three hours)
        if tick_info.is_block_starting_point:
            self.logger.info("tick setting tag" + str(tick_info))
            time_of_next_wakeup = tick_info.time_since_start_in_seconds + interval_length_in_seconds
            # set tag for 3 hours in future
            tester.set_tag_at(time_of_next_wakeup, "lowerlimit")
            # set the limit to 500
            tester.nice_hash_api_edit_order_limit(500)
        # # Example two:
        # # Act on block end (add 0.01 to price if block ends)
        # if tick_info.is_block_ending_point:
        #     self.logger.info("tick block end " + str(tester.nice_hash_api_get_order_price()))
        #     price = tester.nice_hash_api_get_order_price()
        #     could_change_price = tester.nice_hash_api_edit_order_price(price + 0.01)
        #     if could_change_price:
        #         self.logger.info("Could increase price to " + str(tester.nice_hash_api_get_order_price()))
        #     else:
        #         self.logger.info("Could not change price -- too soon.")

    def post_ticks(self, tester):
        self.logger.info("post-tick")
        # TODO: cost and benefit should be calculated here
        pass

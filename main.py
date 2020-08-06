# Tool entry point. All processes start here.
######################################################################
from db import block_data as mine_database
from data_fetcher import update as mine_data_updater
from prediction import predictor, optimizer
from plotter import plots
from datetime import datetime
from datetime import timedelta
import random

######################################################################
from tester.algorithm.alg_example import Algorithm3Hour
from tester.algorithmtester import AlgorithmTester, TICK_INFO_NEEDED_COLUMNS

if __name__ == "__main__":
    # # Update the mine data in the database
    # mine_data_updater.update()
    #

    duration_block_no_data = mine_database.get_columns(TICK_INFO_NEEDED_COLUMNS, 100)
    alg = Algorithm3Hour()
    tester = AlgorithmTester(duration_block_no_data, alg)
    results = []
    for result in tester.test_range(range_size=90):
        results.append(result)
    # while True:
    #     try:
    #         results.append(next(range_iterator))
    #     except StopIteration:
    #         break
    print(results)



    # # Optimize the prediction parameters based on the mine data and tests
    # optimizer.optimize()
    # # Create a working copy of the mine database
    # mine_database.switch_to_temporary_copy()
    # # Insert 100 new predictions into the mine data working copy
    # predictor.extend_mine_data_by_prediction(100)
    # # Generate plots
    # plots.generate_plots()


def test_db():
    mine_database.print_all_raw_data()
    mine_database.switch_to_temporary_copy()
    mine_database.insert_raw_data(1234567, "a", "b", 12, 12.9, 12, 12.7)
    mine_database.print_all_raw_data()
    mine_database.switch_to_main_copy()
    mine_database.print_all_raw_data()

def insert_place_holder_data():
    mine_database.switch_to_temporary_copy()
    for i in range(1000):
        date_value = datetime.now() + timedelta(days=i)
        mine_database.insert_raw_data(int(date_value.timestamp()), date_value.strftime("%d-%m-%Y"),
                                      date_value.strftime("%H:%M:%S"), random.randint(10, 65000), random.randint(2000000, 2050000), i, 6.5)
    mine_database.print_all_raw_data()
    mine_database.switch_to_main_copy(save_temporary_copy=True, remove_temporary_copy=True)
    mine_database.print_all_raw_data()

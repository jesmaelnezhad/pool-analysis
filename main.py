# Tool entry point. All processes start here.
######################################################################
from db import block_data as mine_database
from data_fetcher import update as mine_data_updater
from prediction import predictor, optimizer
from plotter import plots

######################################################################

if __name__ == "__main__":
    # Update the mine data in the database
    mine_data_updater.update()
    # Optimize the prediction parameters based on the mine data and tests
    optimizer.optimize()
    # Create a working copy of the mine database
    mine_database.switch_to_temporary_copy()
    # Insert 100 new predictions into the mine data working copy
    predictor.extend_mine_data_by_prediction(100)
    # Generate plots
    plots.generate_plots()


def test_db():
    mine_database.print_all_raw_data()
    mine_database.switch_to_temporary_copy()
    mine_database.insert_raw_data("a", "b", 12, 12.0, 12, 12.9, 12, 12.7, "c")
    mine_database.print_all_raw_data()
    mine_database.switch_to_main_copy()
    mine_database.print_all_raw_data()
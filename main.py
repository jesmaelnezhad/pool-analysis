# Tool entry point. All processes start here.
######################################################################
import math

from numpy import double

from db import block_data as mine_database, block_data
from data_fetcher import update as mine_data_updater
from prediction import predictor, optimizer, TIME_1_HOURS, TIME_10_MINUTES, Pool
from plotter import plots
from datetime import datetime
from datetime import timedelta
import random

######################################################################
from prediction.algorithm_tester import AlgorithmTester
from prediction.algorithms import StrengthPredictor, Aggregator, StepPredictor, SciKitPredictor
from tester.statistical_analyzers import average_on_columns, min_on_columns, max_on_columns
from utility import logger

TICKS_HOUR = int(TIME_1_HOURS / TIME_10_MINUTES)


def prepare_average_luck_windows():
    # luck_average_windows = [n * TICKS_HOUR for n in range(1, 6)]  # every hour for 5 hours
    # luck_average_windows += [n * 3 * TICKS_HOUR for n in range(2, 8)]  # every 3 hours from hour 6 to 24
    # luck_average_windows += [n * 6 * TICKS_HOUR for n in range(4, 8)]  # every 6 hours from hour 24 to 48
    # luck_average_windows += [n * 24 * TICKS_HOUR for n in range(2, 7)]  # every day for 5 days from day 2
    luck_average_windows = [n * TICKS_HOUR for n in [6, 9, 12, 24, 7*24, 14*24]]  # for the first ensembel results
    # luck_average_windows += [n * 24 * 7 * TICKS_HOUR for n in range(1, 3)]  # every 7 days for two weeks
    return luck_average_windows


def get_average_luck_window_index(window, windows):
    for i, w in enumerate(windows):
        if window == w:
            return i
    return None


def prepare_average_assessment_windows():
    # assessment_windows = [n * TICKS_HOUR for n in range(1, 6)]  # every hour for 5 hours
    assessment_windows = [n * 3 * TICKS_HOUR for n in range(1, 4)]  # every 3 hours from hour 3 to 12
    # assessment_windows = [n * 1 * TICKS_HOUR for n in range(1, 73)]  # every hour until 3 days
    # assessment_windows += [n * 2 * TICKS_HOUR for n in range(6, 13)]  # every 2 hours until 24 hours
    return assessment_windows


def prepare_pools():
    return [Pool("BTCCOM", 0.1716), Pool("VIABTC", 0.0667), Pool("SLUSHPOOL", 0.0148), Pool("OTHERS", 0.7469)]


class OccurrenceFilter:
    def __init__(self, occurred_timestamps):
        self.occurred_timestamps = occurred_timestamps

    def init(self, all_data_points):
        pass

    def filter(self, timestamp):
        return timestamp in self.occurred_timestamps


class HighLuckFilter(OccurrenceFilter):
    def __init__(self, occurred_timestamps, check_dimensions_indices=[1]):
        super().__init__(occurred_timestamps)
        self.check_dimensions_indices = check_dimensions_indices

    def init(self, all_data_points):
        super().init(all_data_points)
        occurred_timestamps = []
        for d in all_data_points:
            data_point_is_accepted = True
            for index in self.check_dimensions_indices:
                if d[index] < 1:
                    data_point_is_accepted = False
                    break
            if data_point_is_accepted:
                occurred_timestamps.append(d[0])
        super().__init__(occurred_timestamps)


def case(data_handler, luck_average_windows, assessment_average_windows, pool_name,
         cases=None,
         method="linear", aggr_method="strength", aggr_avg_window_idx=6, lags=5, stride=0.5, no_estimators=50,
         too_late_to_predict_time_threshold=1.5 * TIME_10_MINUTES,
         positive_decision_occurrence_count_threshold=2,
         decision_aggregation_method="and",
         predictor_class="aggregation",
         data_filter=None,
         case_observation_size=24 * 6):
    """
    :param predictor_class: aggregation or step
    :return:
    """
    logger("==========================================").info("")
    if cases is None:
        logger("CASE").info("{}-{}-{}-{}-{}-{}-{}".format(predictor_class, method, aggr_method,
                                                          (
                                                              aggr_avg_window_idx if aggr_avg_window_idx is not None else ""),
                                                          lags, stride, no_estimators))
    else:
        logger("CASE").info("-- COMBINATION --")
        for test_case in cases:
            logger("CASE").info("{}-{}-{}-{}-{}-{}-{}".format(predictor_class, test_case[0], test_case[1],
                                                              (test_case[2] if test_case[2] is not None else ""),
                                                              test_case[3], test_case[4], test_case[5]))
        logger("CASE").info("-----------------")
    sum_results = None
    no_exp_repeats = 10
    for day_offset in range(no_exp_repeats, 0, -1):
        data_handler.set_main_configs_for_input_data_preparation(no_days_offset=(day_offset - 1) * 2)
        x, y = predictor.export_pool_data_points_for_training(data_handler, pool_name)
        data_points_filter = None
        if predictor_class == "step" or predictor_class == "scikit":
            data_points_filter = data_filter
            if data_points_filter is not None:
                data_points_filter.init(x)
        algorithm_tester = AlgorithmTester(luck_average_windows, assessment_average_windows, x, y)
        # Booster
        if cases is None:
            if predictor_class == "aggregation":
                algorithm_tester.add_algorithm([
                    StrengthPredictor(learning_method=method,
                                      aggregator=Aggregator(method=aggr_method, avg_window_idx=aggr_avg_window_idx),
                                      num_lags=lags, pred_stride=stride, fit_intercept=False,
                                      success_hardness_factor=1,
                                      no_estimators=no_estimators)])
            elif predictor_class == "step":
                algorithm_tester.add_algorithm([
                    StepPredictor(x, learning_method=method,
                                  aggregator=Aggregator(method=aggr_method, avg_window_idx=aggr_avg_window_idx),
                                  num_lags=lags, pred_stride=stride, fit_intercept=False,
                                  no_estimators=no_estimators,
                                  filter_object=data_points_filter,
                                  too_late_to_predict_time_threshold=too_late_to_predict_time_threshold,
                                  positive_decision_occurrence_count_threshold=positive_decision_occurrence_count_threshold)])
            elif predictor_class == "scikit":
                # prepare scikit friendly x
                x_without_timestamp = []
                x_only_timestamp = []
                for data_point in x:
                    x_without_timestamp.append(data_point[1:])
                    x_only_timestamp.append(data_point[0])
                # prepare classification labeling based on assessments
                decision_labels = []
                for data_point_assessments in y:
                    decision_labels.append([(assessment >= 1) for assessment in data_point_assessments[1:]])
                algorithm_tester.add_algorithm([
                    SciKitPredictor(x_only_timestamp, x_without_timestamp, decision_labels,
                                    no_estimators=no_estimators,
                                    filter_object=data_points_filter,
                                    case_observation_size=case_observation_size)])

        else:
            for test_case in cases:
                if predictor_class == "aggregation":
                    algorithm_tester.add_algorithm([
                        StrengthPredictor(learning_method=test_case[0],
                                          aggregator=Aggregator(method=test_case[1], avg_window_idx=test_case[2]),
                                          num_lags=test_case[3], pred_stride=test_case[4], fit_intercept=False,
                                          success_hardness_factor=1,
                                          no_estimators=test_case[5])])
                elif predictor_class == "step":
                    algorithm_tester.add_algorithm([
                        StepPredictor(x, learning_method=test_case[0],
                                      aggregator=Aggregator(method=test_case[1], avg_window_idx=test_case[2]),
                                      num_lags=test_case[3], pred_stride=test_case[4], fit_intercept=False,
                                      no_estimators=test_case[5],
                                      filter_object=data_points_filter,
                                      too_late_to_predict_time_threshold=test_case[6],
                                      positive_decision_occurrence_count_threshold=test_case[7])])
        max_horizon = 1000000
        results = algorithm_tester.test_algorithms(decision_aggregation_method=decision_aggregation_method,
                                                   max_horizon=max_horizon, test_size=150)
        if sum_results is None:
            sum_results = results
        else:
            new_results = []
            for i, w in enumerate(assessment_average_windows):
                if w > max_horizon:
                    continue
                last_window_sum = sum_results[i]
                current_result = results[i]
                new_sum = None
                for last_window_sum_idx in range(0, len(last_window_sum)):
                    if last_window_sum_idx == 0:
                        new_sum = (last_window_sum[last_window_sum_idx] + current_result[last_window_sum_idx],)
                    else:
                        new_sum = new_sum + (
                            last_window_sum[last_window_sum_idx] + current_result[last_window_sum_idx],)
                new_results.append(new_sum)
            sum_results = new_results
    for i, w in enumerate(assessment_average_windows):
        if w > max_horizon:
            continue
        logger("RESULTS-AVG").info(
            "Horizon {} : {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(w, sum_results[i][0] / no_exp_repeats,
                                                                         sum_results[i][1] / no_exp_repeats,
                                                                         sum_results[i][2] / no_exp_repeats,
                                                                         sum_results[i][3] / no_exp_repeats,
                                                                         sum_results[i][4] / no_exp_repeats))


def case_algorithm(algorithm, data_handler, luck_average_windows, assessment_average_windows, pool_name,
                   step_predictor=False):
    ## Scikit
    case(data_handler, luck_average_windows=luck_average_windows,
         assessment_average_windows=assessment_average_windows,
         pool_name=pool_name, no_estimators=100,
         predictor_class="scikit",
         case_observation_size=6 * TICKS_HOUR)
    # # Booster
    # ## Strength
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="strength", aggr_avg_window_idx=None,
    #      lags=5, stride=0.5, no_estimators=50, decision_aggregation_method="and")
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="strength", aggr_avg_window_idx=None,
    #      lags=10, stride=0.5, no_estimators=50, decision_aggregation_method="and")
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="strength", aggr_avg_window_idx=None,
    #      lags=5, stride=2, no_estimators=50, decision_aggregation_method="and")
    # if algorithm == "booster":
    #     case(data_handler, luck_average_windows=luck_average_windows,
    #          assessment_average_windows=assessment_average_windows,
    #          pool_name=pool_name, method=algorithm, aggr_method="strength", aggr_avg_window_idx=None,
    #          lags=5, stride=0.5, no_estimators=100, decision_aggregation_method="and")
    # ## Time accuracy
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="time_accuracy", aggr_avg_window_idx=None,
    #      lags=5, stride=0.5, no_estimators=50, decision_aggregation_method="and")
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="time_accuracy", aggr_avg_window_idx=None,
    #      lags=10, stride=0.5, no_estimators=50, decision_aggregation_method="and")
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="time_accuracy", aggr_avg_window_idx=None,
    #      lags=5, stride=2, no_estimators=50, decision_aggregation_method="and")
    # if algorithm == "booster":
    #     case(data_handler, luck_average_windows=luck_average_windows,
    #          assessment_average_windows=assessment_average_windows,
    #          pool_name=pool_name, method=algorithm, aggr_method="time_accuracy", aggr_avg_window_idx=None,
    #          lags=5, stride=0.5, no_estimators=100, decision_aggregation_method="and")
    # ## Select 6 hours
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="select", aggr_avg_window_idx=6,
    #      lags=5, stride=0.5, no_estimators=50, decision_aggregation_method="and")
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="select", aggr_avg_window_idx=6,
    #      lags=10, stride=0.5, no_estimators=50, decision_aggregation_method="and")
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="select", aggr_avg_window_idx=6,
    #      lags=5, stride=2, no_estimators=50, decision_aggregation_method="and")
    # if algorithm == "booster":
    #     case(data_handler, luck_average_windows=luck_average_windows,
    #          assessment_average_windows=assessment_average_windows,
    #          pool_name=pool_name, method=algorithm, aggr_method="select", aggr_avg_window_idx=6,
    #          lags=5, stride=0.5, no_estimators=100, decision_aggregation_method="and")
    # ## Select 9 hours
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="select", aggr_avg_window_idx=9,
    #      lags=5, stride=0.5, no_estimators=50, decision_aggregation_method="and")
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="select", aggr_avg_window_idx=9,
    #      lags=10, stride=0.5, no_estimators=50, decision_aggregation_method="and")
    # case(data_handler, luck_average_windows=luck_average_windows, assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, method=algorithm, aggr_method="select", aggr_avg_window_idx=9,
    #      lags=5, stride=2, no_estimators=50, decision_aggregation_method="and")
    # if algorithm == "booster":
    #     case(data_handler, luck_average_windows=luck_average_windows,
    #          assessment_average_windows=assessment_average_windows,
    #          pool_name=pool_name, method=algorithm, aggr_method="select", aggr_avg_window_idx=9,
    #          lags=5, stride=0.5, no_estimators=100, decision_aggregation_method="and")
    ## Combination cases
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name,
    #      cases=[(algorithm, "strength", None, 1, 0.25, 10),
    #             (algorithm, "time_accuracy", None, 1, 1, 10)],
    #      decision_aggregation_method="and")
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name,
    #      cases=[(algorithm, "strength", None, 15, 0.25, 100),
    #             (algorithm, "time_accuracy", None, 5, 1, 250)],
    #      decision_aggregation_method="any1")
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name,
    #      cases=[(algorithm, "strength", None, 5, 0.5, 50),
    #             (algorithm, "time_accuracy", None, 5, 1, 250),
    #             (algorithm, "select", 7, 15, 1, 50)],
    #      decision_aggregation_method="any1")


if __name__ == "__main__":
    luck_average_windows = prepare_average_luck_windows()
    assessment_average_windows = prepare_average_assessment_windows()
    pools = prepare_pools()
    # predictor.populate_db_with_random(pools, luck_average_windows, assessment_average_windows)
    table_names = block_data.get_list_of_table_names(which_db="pools")
    print(str(table_names))
    data_handler = predictor.create_data_handler(pools, luck_average_windows, assessment_average_windows)
    pool_names = ["SLUSHPOOL", "BTCCOM", "VIABTC"]
    for pool_name in pool_names[:]:
        logger("RESULTS").info("Pool: {}".format(pool_name))
        # Combination example
        # Booster
        # case_algorithm("booster", data_handler, luck_average_windows, assessment_average_windows, pool_name,
        #                step_predictor=True)
        # Linear
        case_algorithm("linear", data_handler, luck_average_windows, assessment_average_windows, pool_name,
                       step_predictor=True)
        # Linear
        # algorithm_tester.add_algorithm([
        #     StrengthPredictor(learning_method="linear", aggregator=Aggregator(method="strength"),
        #                       num_lags=10, pred_stride=1, fit_intercept=False,
        #                       success_hardness_factor=1)])
        # algorithm_tester.add_algorithm([
        #     StrengthPredictor(learning_method="linear", aggregator=Aggregator(method="time_accuracy"),
        #                       num_lags=15, pred_stride=1, fit_intercept=False,
        #                       success_hardness_factor=1)])
        # algorithm_tester.add_algorithm([
        #     StrengthPredictor(learning_method="linear", aggregator=Aggregator(method="select", avg_window_idx=6),
        #                       num_lags=10, pred_stride=0.25, fit_intercept=False,
        #                       success_hardness_factor=1)])
        # algorithm_tester.add_algorithm([
        #     StrengthPredictor(learning_method="linear", aggregator=Aggregator(method="select", avg_window_idx=7),
        #                       num_lags=15, pred_stride=1, fit_intercept=False,
        #                       success_hardness_factor=1)])

        # # Booster
        # algorithm_tester.add_algorithm([
        #     StrengthPredictor(learning_method="booster", aggregator=Aggregator(method="strength"),
        #                       num_lags=5, pred_stride=0.5, fit_intercept=False,
        #                       success_hardness_factor=1,
        #                       no_estimators=50)])
        # algorithm_tester.add_algorithm([
        #     StrengthPredictor(learning_method="booster", aggregator=Aggregator(method="time_accuracy"),
        #                       num_lags=5, pred_stride=1, fit_intercept=False,
        #                       success_hardness_factor=1,
        #                       no_estimators=50)])
        # algorithm_tester.add_algorithm([
        #     StrengthPredictor(learning_method="booster", aggregator=Aggregator(method="select", avg_window_idx=7),
        #                       num_lags=5, pred_stride=1, fit_intercept=False,
        #                       success_hardness_factor=1,
        #                       no_estimators=50)])
        # algorithm_tester.add_algorithm([
        #     StrengthPredictor(learning_method="booster", aggregator=Aggregator(method="select", avg_window_idx=6),
        #                       num_lags=5, pred_stride=1, fit_intercept=False,
        #                       success_hardness_factor=1,
        #                       no_estimators=50)])

        # # Deep learning
        # algorithm_tester.add_algorithm([
        #     StrengthPredictor(learning_method="deep", aggregator=Aggregator(method="strength"),
        #                       num_lags=1, pred_stride=0.5, fit_intercept=False,
        #                       success_hardness_factor=1,
        #                       no_channels=1)])

        # algorithm_tester.test_algorithms(decision_aggregation_method="any2")

        # test prediction on all columns
        # for n in range(len(x[0])-1, len(x[0])):
        # for n in range(1, len(x[0])):
        #     logger("RESULTS").info("Average window index: {}".format(n-1))
        #     nth_series = predictor.get_nth_column(x, n)
        #     ### horizon
        #     num_lags, pred_stride, fit_intercept, horizon = 0.1, 10, False, 0.5
        #     data_handler.test_model_on_series(nth_series, num_lags, pred_stride, fit_intercept, horizon)
        # ### num_lags
        # num_lags, pred_stride, fit_intercept, horizon = 7, 1, False, 1
        # data_handler.test_model_on_series(nth_series, num_lags, pred_stride, fit_intercept, horizon)
        # num_lags, pred_stride, fit_intercept, horizon = 5, 1, False, 1
        # data_handler.test_model_on_series(nth_series, num_lags, pred_stride, fit_intercept, horizon)
        # num_lags, pred_stride, fit_intercept, horizon = 3, 1, False, 1
        # data_handler.test_model_on_series(nth_series, num_lags, pred_stride, fit_intercept, horizon)
        # ### stride
        # num_lags, pred_stride, fit_intercept, horizon = 3, 2, False, 1
        # data_handler.test_model_on_series(nth_series, num_lags, pred_stride, fit_intercept, horizon)
        # num_lags, pred_stride, fit_intercept, horizon = 3, 4, False, 1
        # data_handler.test_model_on_series(nth_series, num_lags, pred_stride, fit_intercept, horizon)
        # num_lags, pred_stride, fit_intercept, horizon = 3, 8, False, 1
        # data_handler.test_model_on_series(nth_series, num_lags, pred_stride, fit_intercept, horizon)
        # ### fit intercept
        # num_lags, pred_stride, fit_intercept, horizon = 3, 8, True, 1
        # data_handler.test_model_on_series(nth_series, num_lags, pred_stride, fit_intercept, horizon)

    # # Update the mine data in the database
    # mine_data_updater.update()
    # #
    # mine_database.print_all_raw_data_in_tsdb_format2()
    # mine_database.print_all_raw_data()
    # duration_block_no_data = mine_database.get_columns(TICK_INFO_NEEDED_COLUMNS, 569)
    # alg = Algorithm3Hour()
    # tester = AlgorithmTester(duration_block_no_data, alg)
    # results = []
    # for result in tester.test_range(range_size=48):
    #     results.append(result)
    # logger("main").info("Testing ranges finished.")
    #
    # average = average_on_columns(results)
    # minimums = min_on_columns(results)
    # maximums = max_on_columns(results)
    #
    # logger("main").info("        \t\tCost\tReward\tLowest profit >> Profit << Highest profit")
    # logger("main").info("Averages\t|\t{0:.3f}\t{1:.3f}\t      {3:.3f} >> {2:.3f} << {4:.3f}".format(*average))
    # logger("main").info("Minimums\t|\t{0:.3f}\t{1:.3f}\t      {3:.3f} >> {2:.3f} << {4:.3f}".format(*minimums))
    # logger("main").info("Maximums\t|\t{0:.3f}\t{1:.3f}\t      {3:.3f} >> {2:.3f} << {4:.3f}".format(*maximums))

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
                                      date_value.strftime("%H:%M:%S"), random.randint(10, 65000),
                                      random.randint(2000000, 2050000), i, 6.5)
    mine_database.print_all_raw_data()
    mine_database.switch_to_main_copy(save_temporary_copy=True, remove_temporary_copy=True)
    mine_database.print_all_raw_data()

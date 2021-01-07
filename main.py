# Tool entry point. All processes start here.
######################################################################
import math

from numpy import double

from db import block_data as mine_database, block_data
from data_fetcher import update as mine_data_updater
from prediction import predictor, optimizer, TIME_1_HOURS, TIME_10_MINUTES, Pool
from plotter import plots
from datetime import timedelta
import random
from datetime import datetime

######################################################################
from prediction.SciKitPredictor import SciKitPredictor
from prediction.algorithm_tester import AlgorithmTester
from prediction.algorithms import StrengthPredictor, Aggregator, StepPredictor
from tester.statistical_analyzers import average_on_columns, min_on_columns, max_on_columns
from utility import logger

TICKS_HOUR = int(TIME_1_HOURS / TIME_10_MINUTES)


def prepare_average_luck_windows():
    # luck_average_windows = [n * TICKS_HOUR for n in range(1, 6)]  # every hour for 5 hours
    # luck_average_windows += [n * 3 * TICKS_HOUR for n in range(2, 8)]  # every 3 hours from hour 6 to 24
    # luck_average_windows += [n * 6 * TICKS_HOUR for n in range(4, 8)]  # every 6 hours from hour 24 to 48
    # luck_average_windows += [n * 24 * TICKS_HOUR for n in range(2, 7)]  # every day for 5 days from day 2
    luck_average_windows = [n * TICKS_HOUR for n in [12, 7 * 24]]  # for the first ensemble results
    # luck_average_windows += [n * 24 * 7 * TICKS_HOUR for n in range(1, 3)]  # every 7 days for two weeks
    return luck_average_windows


def get_every_nth_value_for_average_window(window):
    return 1
    # if window == 24 * TICKS_HOUR:
    #     return 6
    # if window == 14 * 24 * TICKS_HOUR:
    #     return 72
    # if window in [n * TICKS_HOUR for n in [24, 14 * 24]]:
    #     return 6
    # if 0 < window < 24 * TICKS_HOUR:
    #     return 1
    # elif 24 * TICKS_HOUR <= window < 7 * 24 * TICKS_HOUR:
    #     return 2
    # else:
    #     return 3


def get_average_luck_window_index(window, windows):
    for i, w in enumerate(windows):
        if window == w:
            return i
    return None


def prepare_average_assessment_windows():
    # assessment_windows = [n * TICKS_HOUR for n in range(1, 6)]  # every hour for 5 hours
    assessment_windows = [n * 3 * TICKS_HOUR for n in range(3, 4)]  # every 3 hours from hour 3 to 12
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
         case_observation_size=24 * 6,
         prediction_above_one_margin=0.5,
         round_to_n_decimal_points=5,
         class_weight=None,
         max_depth=5,
         criterion='entropy',
         min_samples_split=2,
         min_samples_leaf=1,
         bootstrap=False,
         oob_score=False):
    """
    :param predictor_class: aggregation or step or scikit
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
        data_handler.set_main_configs_for_input_data_preparation(no_days_offset=(day_offset - 1) * 3)
        x, y = predictor.export_pool_data_points_for_training(data_handler, pool_name,
                                                              round_to_n_decimal_points=round_to_n_decimal_points)
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
                    decision_labels.append(
                        [(assessment >= 1 + prediction_above_one_margin) for assessment in data_point_assessments[1:]])
                algorithm_tester.add_algorithm([
                    SciKitPredictor(x_only_timestamp, x_without_timestamp, decision_labels,
                                    no_estimators=no_estimators,
                                    filter_object=data_points_filter,
                                    case_observation_size=case_observation_size,
                                    every_m_observations_for_dimension=[
                                        get_every_nth_value_for_average_window(avg_window) for avg_window in
                                        luck_average_windows],
                                    class_weight=class_weight,
                                    max_depth=max_depth,
                                    criterion=criterion,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    bootstrap=bootstrap,
                                    oob_score=oob_score)])

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
                                                   max_horizon=max_horizon, test_size=100)
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
        logger("RESULTS-AVG").debug(
            "Day offset: \t Horizon  : S/T\tP/T\tRP/T\tPS/P\tPS/T\tT")
        for i, w in enumerate(assessment_average_windows):
            if w > max_horizon:
                continue
            logger("RESULTS-AVG").debug(
                "Day offset: {} \t Horizon {} : {:.2f}\t{:.2f}\t{}\t{:.2f}\t{:.2f}\t{}".format(
                    day_offset,
                    w, sum_results[i][0] / no_exp_repeats,
                       sum_results[i][1] / no_exp_repeats,
                       sum_results[i][2] / no_exp_repeats,
                       sum_results[i][3] / no_exp_repeats,
                       sum_results[i][4] / no_exp_repeats,
                       sum_results[i][5] / no_exp_repeats))
    logger("RESULTS-AVG").info(
        "Day offset: \t Horizon  : S/T\tP/T\tRP/T\tPS/P\tPS/T\tT")
    for i, w in enumerate(assessment_average_windows):
        if w > max_horizon:
            continue
        logger("RESULTS-AVG").info(
            "Horizon {} : {:.2f}\t{:.2f}\t{}\t{:.2f}\t{:.2f}\t{}".format(w, sum_results[i][0] / no_exp_repeats,
                                                                         sum_results[i][1] / no_exp_repeats,
                                                                         sum_results[i][2] / no_exp_repeats,
                                                                         sum_results[i][3] / no_exp_repeats,
                                                                         sum_results[i][4] / no_exp_repeats,
                                                                         sum_results[i][5] / no_exp_repeats))


def case_algorithm(algorithm, data_handler, luck_average_windows, assessment_average_windows, pool_name,
                   step_predictor=False):
    ## Scikit

    default_parameters = {
        "no_estimators": 150,
        "case_observation_size": 24 * TICKS_HOUR,
        "prediction_above_one_margin": 0,
        "round_to_n_decimal_points": 7,
        "class_weight": None,
        "max_depth": 3,
        "criterion": 'entropy',
        "min_samples_split": 10,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "oob_score": True,
    }

    logger("CASES").info("========================================================== base")
    case(data_handler, luck_average_windows=luck_average_windows,
         assessment_average_windows=assessment_average_windows,
         pool_name=pool_name, no_estimators=default_parameters["no_estimators"],
         predictor_class="scikit",
         case_observation_size=default_parameters["case_observation_size"],
         prediction_above_one_margin=default_parameters["prediction_above_one_margin"],
         round_to_n_decimal_points=default_parameters["round_to_n_decimal_points"],
         class_weight=default_parameters["class_weight"],
         max_depth=default_parameters["max_depth"],
         criterion=default_parameters["criterion"],
         min_samples_split=default_parameters["min_samples_split"],
         min_samples_leaf=default_parameters["min_samples_leaf"],
         bootstrap=default_parameters["bootstrap"],
         oob_score=default_parameters["oob_score"])

    # logger("CASES").info("========================================================== class_weight None")
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, no_estimators=default_parameters["no_estimators"],
    #      predictor_class="scikit",
    #      case_observation_size=default_parameters["case_observation_size"],
    #      prediction_above_one_margin=default_parameters["prediction_above_one_margin"],
    #      round_to_n_decimal_points=default_parameters["round_to_n_decimal_points"],
    #      class_weight=None,
    #      max_depth=default_parameters["max_depth"],
    #      criterion=default_parameters["criterion"],
    #      min_samples_split=default_parameters["min_samples_split"],
    #      min_samples_leaf=default_parameters["min_samples_leaf"],
    #      bootstrap=default_parameters["bootstrap"],
    #      oob_score=default_parameters["oob_score"])

    # logger("CASES").info("========================================================== 6 decimal points")
    #
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, no_estimators=default_parameters["no_estimators"],
    #      predictor_class="scikit",
    #      case_observation_size=default_parameters["case_observation_size"],
    #      prediction_above_one_margin=default_parameters["prediction_above_one_margin"],
    #      round_to_n_decimal_points=6,
    #      class_weight=default_parameters["class_weight"],
    #      max_depth=default_parameters["max_depth"],
    #      criterion=default_parameters["criterion"],
    #      min_samples_split=default_parameters["min_samples_split"],
    #      min_samples_leaf=default_parameters["min_samples_leaf"],
    #      bootstrap=default_parameters["bootstrap"],
    #      oob_score=default_parameters["oob_score"])

    # logger("CASES").info("========================================================== max_depth less")
    #
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, no_estimators=default_parameters["no_estimators"],
    #      predictor_class="scikit",
    #      case_observation_size=default_parameters["case_observation_size"],
    #      prediction_above_one_margin=default_parameters["prediction_above_one_margin"],
    #      round_to_n_decimal_points=default_parameters["round_to_n_decimal_points"],
    #      class_weight=default_parameters["class_weight"],
    #      max_depth=default_parameters["max_depth"] / 2,
    #      criterion=default_parameters["criterion"],
    #      min_samples_split=default_parameters["min_samples_split"],
    #      min_samples_leaf=default_parameters["min_samples_leaf"],
    #      bootstrap=default_parameters["bootstrap"],
    #      oob_score=default_parameters["oob_score"])
    #
    # logger("CASES").info("========================================================== min_samples_split less")
    #
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, no_estimators=default_parameters["no_estimators"],
    #      predictor_class="scikit",
    #      case_observation_size=default_parameters["case_observation_size"],
    #      prediction_above_one_margin=default_parameters["prediction_above_one_margin"],
    #      round_to_n_decimal_points=default_parameters["round_to_n_decimal_points"],
    #      class_weight=default_parameters["class_weight"],
    #      max_depth=default_parameters["max_depth"],
    #      criterion=default_parameters["criterion"],
    #      min_samples_split=int(default_parameters["min_samples_split"] / 2),
    #      min_samples_leaf=default_parameters["min_samples_leaf"],
    #      bootstrap=default_parameters["bootstrap"],
    #      oob_score=default_parameters["oob_score"])
    #
    # logger("CASES").info("========================================================== min split * 10")
    #
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, no_estimators=default_parameters["no_estimators"],
    #      predictor_class="scikit",
    #      case_observation_size=default_parameters["case_observation_size"],
    #      prediction_above_one_margin=default_parameters["prediction_above_one_margin"],
    #      round_to_n_decimal_points=default_parameters["round_to_n_decimal_points"],
    #      class_weight=default_parameters["class_weight"],
    #      max_depth=default_parameters["max_depth"],
    #      criterion=default_parameters["criterion"],
    #      min_samples_split=20,
    #      min_samples_leaf=default_parameters["min_samples_leaf"],
    #      bootstrap=default_parameters["bootstrap"],
    #      oob_score=default_parameters["oob_score"])
    #
    # logger("CASES").info("========================================================== min leaf * 10")
    #
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, no_estimators=default_parameters["no_estimators"],
    #      predictor_class="scikit",
    #      case_observation_size=default_parameters["case_observation_size"],
    #      prediction_above_one_margin=default_parameters["prediction_above_one_margin"],
    #      round_to_n_decimal_points=default_parameters["round_to_n_decimal_points"],
    #      class_weight=default_parameters["class_weight"],
    #      max_depth=default_parameters["max_depth"],
    #      criterion=default_parameters["criterion"],
    #      min_samples_split=default_parameters["min_samples_split"],
    #      min_samples_leaf=10,
    #      bootstrap=default_parameters["bootstrap"],
    #      oob_score=default_parameters["oob_score"])
    #
    # logger("CASES").info("========================================================== min leaf min split max_depth * 5")
    #
    # case(data_handler, luck_average_windows=luck_average_windows,
    #      assessment_average_windows=assessment_average_windows,
    #      pool_name=pool_name, no_estimators=default_parameters["no_estimators"],
    #      predictor_class="scikit",
    #      case_observation_size=default_parameters["case_observation_size"],
    #      prediction_above_one_margin=default_parameters["prediction_above_one_margin"],
    #      round_to_n_decimal_points=default_parameters["round_to_n_decimal_points"],
    #      class_weight=default_parameters["class_weight"],
    #      max_depth=25,
    #      criterion=default_parameters["criterion"],
    #      min_samples_split=10,
    #      min_samples_leaf=5,
    #      bootstrap=default_parameters["bootstrap"],
    #      oob_score=default_parameters["oob_score"])

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


def group_count_x_days(resultsList, x):
    results = []
    group_start = None
    group_days = set()
    group_size = 0
    for row in resultsList:
        day_date = row[0]
        count = row[1]
        if group_start is None:
            group_start = day_date
        if (not day_date in group_days) and (len(group_days) == x):
            results.append([group_start, str(group_size)])
            group_start = day_date
            group_size = count
            group_days.clear()
            group_days.add(day_date)
        else:
            group_days.add(day_date)
            group_size += count
    results.append([group_start, str(group_size)])
    return results


def get_list_key(l):
    return datetime.strptime(l[0], '%m-%d-%Y').timestamp()


if __name__ == "__main__":

    resultsList = block_data.get_all_day_counts()
    resultsList.sort(key=get_list_key)
    # # print day count
    # for row in resultsList:
    #     print("\t".join([str(r) for r in row]))
    per_two_days = group_count_x_days(resultsList, 14)
    per_two_days.sort(key=get_list_key)
    # print(per_two_days)
    for row in per_two_days:
        print("\t".join([str(r) for r in row]))



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

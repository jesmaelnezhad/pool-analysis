from pathlib import Path

from prediction.algorithms import ForcasterPipelinePredictor
from utility import logger
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.classification.compose import ColumnEnsembleClassifier, TimeSeriesForestClassifier
import os
import hashlib
import pathlib


def sktime_case_string_of(observations, observation_identified_flags, label,
                          every_m_observations_for_dimension=None):
    if len(observations) == 0:
        return None

    if len(observations) != len(observation_identified_flags):
        return None

    no_dimensions = len(observations[0])
    if every_m_observations_for_dimension is None:
        every_m_observations_for_dimension = [1 for i in range(0, no_dimensions)]
    elif len(every_m_observations_for_dimension) != no_dimensions:
        logger("MODEL-DATA-PREP").warn("Wrong number of dimensions in every_m_observations_for_dimension parameter.")
        every_m_observations_for_dimension = [1 for i in range(0, no_dimensions)]

    dimension_strings = []
    for d in range(0, no_dimensions):
        every_nth = every_m_observations_for_dimension[d]
        observation_strings = []
        for o_idx, o in enumerate(observations):
            if o_idx % every_nth != 0:
                observation_strings.append('?')
            else:
                if observation_identified_flags[o_idx]:
                    observation_strings.append(str(o[d]))
                else:
                    observation_strings.append('?')
        dimension_strings.append(','.join(observation_strings))
    x_part = ':'.join(dimension_strings)
    return x_part + ':' + str(label)


class SciKitPredictor(ForcasterPipelinePredictor):

    def __init__(self, timestamps, data_points, correct_decision_labels,
                 no_estimators=100,
                 filter_object=None,
                 case_observation_size=24 * 6,
                 every_m_observations_for_dimension=None,
                 class_weight="balanced",
                 max_depth=5,
                 criterion='entropy',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 bootstrap=False,
                 oob_score=False):
        super().__init__(data_points, no_estimators=no_estimators)
        """
        :param correct_decision_labels: parallel list with assessment windows. List of True/False data points label lists
        """
        if len(timestamps) != len(data_points) or len(timestamps) != len(correct_decision_labels):
            logger("MODEL-CREATE").error(
                "Failed to create predictor because of inconsistent length of timestamp/x/y lists")
            return
        self.timestamps = timestamps
        self.correct_decision_labels = correct_decision_labels
        self.filter = filter_object
        self.data_points_filter_results = []
        self.case_observation_size = case_observation_size
        self.every_m_observations_for_dimension = every_m_observations_for_dimension
        self.class_weight = class_weight
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        # The following list will contain one classifier per assessment window
        self.classifiers = []
        self.horizon = 9 * 6

    def init(self):
        if self.filter is not None:
            self.data_points_filter_results = [self.filter.filter(ts) for ts in self.timestamps]
        else:
            self.data_points_filter_results = [True for data_point in self.data_points]

    def create_class_weight_dict(self, true_index):
        if not type(self.class_weight) is dict:
            return self.class_weight
        new_dict = {}
        for l in self.class_weight:
            if l:
                new_dict[true_index] = self.class_weight[l]
            else:
                new_dict[1 - true_index] = self.class_weight[l]

    def prepare_ts_file_name(self,
                             start_index,
                             end_index,
                             case_observation_size,
                             labeling_index,
                             label_window):
        file_full_path = os.getcwd() + "/data/fit_data_w" + str(label_window)
        hasher_object = hashlib.sha256()
        case_last_data_point_index = end_index
        while case_last_data_point_index > start_index + case_observation_size:
            case_data_points = self.data_points[
                               case_last_data_point_index - case_observation_size:case_last_data_point_index]
            case_filter_flags = self.data_points_filter_results[
                                case_last_data_point_index - case_observation_size:case_last_data_point_index]
            case_label = self.correct_decision_labels[case_last_data_point_index - 1][labeling_index]
            hasher_object.update(bytes(str(case_data_points), encoding='utf8'))
            hasher_object.update(bytes(str(case_filter_flags), encoding='utf8'))
            hasher_object.update(bytes(str(case_label), encoding='utf8'))
            case_last_data_point_index -= 1
        hasher_object.update(bytes(str(self.every_m_observations_for_dimension), encoding='utf8'))
        file_full_path += "_" + hasher_object.hexdigest()
        return file_full_path

    def prepare_ts_file(self,
                        start_index,
                        end_index,
                        case_observation_size,
                        labeling_index,
                        label_window):
        file_full_path = self.prepare_ts_file_name(start_index,
                                                   end_index, case_observation_size,
                                                   labeling_index, label_window)
        if Path(file_full_path).is_file():
            logger("MODEL-DATA-PREP").debug("Using existing file: " + file_full_path)
            return file_full_path
        try:
            fit_data_file = open(file_full_path, 'w')
            fit_data_file.write(
                "@problemName fit_data\n@timeStamps false\n@univariate false\n@classLabel true True False\n@data\n")
            no_cases = 0
            case_last_data_point_index = end_index
            while case_last_data_point_index > start_index + case_observation_size:
                case_data_points = self.data_points[
                                   case_last_data_point_index - case_observation_size:case_last_data_point_index]
                case_filter_flags = self.data_points_filter_results[
                                    case_last_data_point_index - case_observation_size:case_last_data_point_index]
                case_label = self.correct_decision_labels[case_last_data_point_index - 1][labeling_index]
                case_str = sktime_case_string_of(case_data_points, case_filter_flags, case_label,
                                                 self.every_m_observations_for_dimension)
                fit_data_file.write(case_str + "\n")
                no_cases += 1
                case_last_data_point_index -= 1
            logger("MODEL-DATA-PREP").debug("No cases written: " + str(no_cases) + " -> " + file_full_path)
            return file_full_path
        finally:
            fit_data_file.close()

    def fit(self, luck_average_windows, assessment_windows, until=None, max_horizon=9 * 6):
        logger("MODEL-FIT").debug(
            "max_horizon: {} / avg windows: {} / assmnt windows: {} / until: {} / total_data_size: {}".format(
                max_horizon,
                str(luck_average_windows),
                str(assessment_windows),
                until,
                len(self.data_points)))
        if until is not None and (until < 0 or until >= len(self.data_points)):
            logger("MODEL-FIT").error("Parameter until is too large for the given data points: {}".format(until))
            return
        self.horizon = max_horizon
        for wi, w in enumerate(assessment_windows):
            if w > self.horizon:
                break
            # prepare data frame for sktime package

            temporary_data_fit_file = self.prepare_ts_file(0, len(self.data_points) if until is None else until,
                                                           self.case_observation_size, wi, w)

            # parse data frames from the temporary fit data file
            X, y = load_from_tsfile_to_dataframe(temporary_data_fit_file, replace_missing_vals_with="-100")
            # which label is the first one?
            true_index = 0
            if y[0] == "false":
                true_index = 1
            new_class_weights = self.create_class_weight_dict(true_index=true_index)
            estimators = []
            for i in range(0, len(luck_average_windows)):
                estimators.append(("TSF{}".format(i), TimeSeriesForestClassifier(
                    n_estimators=int(self.no_estimators),
                    n_jobs=16,
                    max_depth=self.max_depth,
                    class_weight=new_class_weights,
                    criterion=self.criterion,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    oob_score=self.oob_score,
                    bootstrap=self.bootstrap),
                                   [i]))
            c = ColumnEnsembleClassifier(estimators=estimators)
            c.fit(X, y)
            # print(str(c.classes_))
            self.classifiers.append(c)

    def predict(self, luck_average_windows, assessment_windows, from_idx=None):
        logger("MODEL-PREDICT").debug(
            "horizon: {} / avg windows: {} / assmnt windows: {} / from_idx: {} / total_data_size: {}".format(
                self.horizon,
                str(luck_average_windows),
                str(assessment_windows),
                from_idx,
                len(self.data_points)))
        if from_idx is not None and (from_idx < 0 or from_idx >= len(self.data_points)):
            logger("MODEL-PREDICT").error("Parameter until is too large for the given data points: {}".format(from_idx))
            return
        from_idx = len(self.data_points) - 1 if from_idx is None else from_idx
        y_predictions = [[] for i in range(from_idx, len(self.data_points))]
        for wi, w in enumerate(assessment_windows):
            if w > self.horizon:
                break
            # prepare data frame for sktime package
            temporary_data_fit_file = self.prepare_ts_file(from_idx - self.case_observation_size, len(self.data_points),
                                                           self.case_observation_size, wi, w)
            X, y = load_from_tsfile_to_dataframe(temporary_data_fit_file, replace_missing_vals_with="-100")
            y_prediction = self.classifiers[wi].predict(X)
            for pred_point_index, y_point_prediction in enumerate(y_prediction):
                y_predictions[pred_point_index].append(y_point_prediction)

        logger("MODEL-PREDICT").debug("Predictions: {}".format(y_predictions))
        return y_predictions

    def decide(self, current_x, luck_average_windows, assessment_window, horizon_predictions, assessment_windows):
        logger("SKTIME-DECIDE").debug(
            "current_x: {} / avg_windows: {} / assmnt_window: {} / predictions: {} / assmnt_windows: {}".format(
                current_x,
                luck_average_windows,
                assessment_window,
                horizon_predictions,
                assessment_windows
            ))
        if horizon_predictions is None:
            return None
        for wi, w in enumerate(assessment_windows):
            if w > self.horizon:
                break
            if w == assessment_window:
                return horizon_predictions[wi] == 'true'
        return None

import torch
from skits.feature_extraction import AutoregressiveTransformer
from skits.pipeline import ForecasterPipeline
from skits.preprocessing import HorizonTransformer, ReversibleImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FeatureUnion
import numpy as np
from skorch import NeuralNetRegressor
from skorch.callbacks import GradientNormClipping
from xgboost import XGBRegressor

from prediction import TIME_10_MINUTES
from prediction.deep_learning import TemporalConvNet
from utility import logger


class ForcasterPipelinePredictor:
    def __init__(self, data_points, learning_method="linear", aggregator=None,
                 num_lags=7, pred_stride=1, fit_intercept=False,
                 no_estimators=300,
                 no_channels=11):
        """
        :param data_points:
        :param num_lags:
        :param pred_stride:
        :param fit_intercept:
        :param aggregator:
        :param learning_method: linear, booster, deep
        """
        self.data_points = data_points
        self.num_lags = num_lags
        self.pred_stride = pred_stride
        self.fit_intercept = fit_intercept
        self.horizon = 6
        self.aggregator = aggregator
        self.learning_method = learning_method
        self.no_estimators = no_estimators
        self.no_channels = no_channels
        self.pipeline = None

    def get_pipeline(self):
        regressor = None
        if self.learning_method == "linear":
            regressor = MultiOutputRegressor(LinearRegression(fit_intercept=self.fit_intercept),
                                             n_jobs=6)
        elif self.learning_method == "booster":
            regressor = MultiOutputRegressor(XGBRegressor(n_jobs=12,
                                                          n_estimators=self.no_estimators))
        elif self.learning_method == "deep":
            regressor = NeuralNetRegressor(
                module=TemporalConvNet,
                module__num_inputs=1,
                module__num_channels=[2] * self.no_channels,
                module__output_sz=self.horizon,
                module__kernel_size=5,
                module__dropout=0.0,
                max_epochs=60,
                batch_size=256,
                lr=2e-3,
                optimizer=torch.optim.Adam,
                device='cpu',
                iterator_train__shuffle=True,
                callbacks=[GradientNormClipping(gradient_clip_value=1,
                                                gradient_clip_norm_type=2)],
                train_split=None,
            )
        return ForecasterPipeline([
            # Convert the `y` target into a horizon
            ('pre_horizon', HorizonTransformer(horizon=self.horizon)),
            ('pre_reversible_imputer', ReversibleImputer(y_only=True)),
            ('features', FeatureUnion([
                # Generate a week's worth of autoregressive features
                ('ar_features', AutoregressiveTransformer(
                    num_lags=int(self.horizon * self.num_lags), pred_stride=self.pred_stride)),
            ])),
            ('post_feature_imputer', ReversibleImputer()),
            ('regressor', regressor)
        ])


class Aggregator:
    def __init__(self, method, avg_window_idx=None):
        """

        :param method: strength, time_accuracy, or select
        :param avg_window_idx: index of the average column to use as aggregation
        """
        self.method = method
        self.avg_window_idx = avg_window_idx
        self.strength_cache = {}

    def aggregate_lucks(self, data_points, average_windows):
        new_x = []
        for point in data_points:
            if point[0] in self.strength_cache:
                new_point = [point[0], self.strength_cache[point[0]]]
            else:
                if self.method == "select":
                    new_point = [point[0]] + [point[self.avg_window_idx + 1]]
                    self.strength_cache[point[0]] = point[self.avg_window_idx + 1]
                else:
                    transformed_list = None
                    if self.method == "strength":
                        transformed_list = [point[i + 1] * w for i, w in enumerate(average_windows)]
                    elif self.method == "time_accuracy":
                        transformed_list = [point[i + 1] / w for i, w in enumerate(average_windows)]
                    new_point = [point[0]] + [sum(transformed_list) / len(transformed_list)]
                    self.strength_cache[point[0]] = sum(transformed_list) / len(transformed_list)
            new_x.append(new_point)
        return new_x


class StrengthPredictor(ForcasterPipelinePredictor):

    def __init__(self, data_points, learning_method="linear", aggregator=None,
                 num_lags=7, pred_stride=1, fit_intercept=False,
                 no_estimators=300,
                 no_channels=11,
                 success_hardness_factor=1):
        super().__init__(data_points, learning_method, aggregator, num_lags, pred_stride, fit_intercept, no_estimators,
                         no_channels)
        self.success_hardness_factor = success_hardness_factor

    def init(self):
        pass

    def fit(self, luck_average_windows, assessment_windows, until=None, max_horizon=9 * 6):
        x = self.data_points
        if until is not None:
            x = self.data_points[:until]
        self.pred_stride = int(len(assessment_windows) * self.pred_stride)

        self.horizon = max_horizon
        logger("MODEL-FIT").debug(
            "num_lags: {} / pred_stride: {} / fit_intercept: {} / horizon: {}".format(self.num_lags,
                                                                                      self.pred_stride,
                                                                                      self.fit_intercept,
                                                                                      self.horizon))
        strengths = self.aggregator.aggregate_lucks(x, luck_average_windows)
        stengths_serie = [s[1] for s in strengths]
        y = np.array(stengths_serie)
        X = y.reshape(-1, 1).copy()

        self.pipeline = self.get_pipeline()
        if self.learning_method == "deep":
            self.pipeline.fit(X[:-1].astype(np.float32), y[:-1].astype(np.float32))
        else:
            self.pipeline.fit(X[:-1], y[:-1])

    def predict(self, luck_average_windows, assessment_windows, from_idx=None):
        logger("MODEL-FIT").debug(
            "num_lags: {} / pred_stride: {} / fit_intercept: {} / horizon: {}".format(self.num_lags,
                                                                                      self.pred_stride,
                                                                                      self.fit_intercept,
                                                                                      self.horizon))
        x = self.data_points
        strengths = self.aggregator.aggregate_lucks(x, luck_average_windows)
        stengths_serie = [s[1] for s in strengths]
        y = np.array(stengths_serie)
        X = y.reshape(-1, 1).copy()
        result = []
        if self.learning_method == "deep":
            if from_idx is None:
                prediction = self.pipeline.predict(X.astype(np.float32), start_idx=len(X) - 1, to_scale=True)
                predictions = []
                for h in assessment_windows:
                    if h <= self.horizon:
                        predictions.append(prediction[h - 1])
                result.append(predictions)
            else:
                prediction = self.pipeline.predict(X.astype(np.float32), start_idx=from_idx, to_scale=True)
                for p in prediction:
                    predictions = []
                    for h in assessment_windows:
                        if h <= self.horizon:
                            predictions.append(p[h - 1])
                    result.append(predictions)
        else:
            if from_idx is None:
                prediction = self.pipeline.predict(X, start_idx=len(X) - 1)
                predictions = []
                for h in assessment_windows:
                    if h <= self.horizon:
                        predictions.append(prediction[h - 1])
                result.append(predictions)
            else:
                prediction = self.pipeline.predict(X, start_idx=from_idx)
                for p in prediction:
                    predictions = []
                    for h in assessment_windows:
                        if h <= self.horizon:
                            predictions.append(p[h - 1])
                    result.append(predictions)
        return result

    def decide(self, current_x, luck_average_windows, assessment_window, horizon_predictions, assessment_windows):
        horizon_prediction = None
        for i, aw in enumerate(assessment_windows):
            if assessment_window == aw:
                horizon_prediction = horizon_predictions[i]
                break
        if horizon_prediction is None:
            return None
        strength = self.aggregator.aggregate_lucks([current_x, ], luck_average_windows)
        return horizon_prediction > strength[0][1] * self.success_hardness_factor


def find_index_of_last_timestamp_before(data_points, timestamp):
    index = len(data_points) - 1
    while index >= 0 and data_points[index][0] > timestamp:
        index -= 1
    return index


class StepPredictor(ForcasterPipelinePredictor):

    def __init__(self, data_points, learning_method="linear", aggregator=None,
                 num_lags=7, pred_stride=1, fit_intercept=False,
                 no_estimators=300,
                 no_channels=11,
                 filter_object=None,
                 too_late_to_predict_time_threshold=1.5 * TIME_10_MINUTES,
                 positive_decision_occurrence_count_threshold=2):
        super().__init__(data_points, learning_method, aggregator, num_lags, pred_stride, fit_intercept, no_estimators,
                         no_channels)
        self.filter = filter_object
        self.data_points_filtered = []
        self.too_late_to_predict_time_threshold = too_late_to_predict_time_threshold
        self.positive_decision_occurrence_count_threshold = positive_decision_occurrence_count_threshold
        self.prediction_failed_in_fit = False

    def init(self):
        self.data_points_filtered = [data_point for data_point in self.data_points if self.filter.filter(data_point[0])]
        print(str(self.data_points_filtered))

    def fit(self, luck_average_windows, assessment_windows, until=None, max_horizon=4):
        x = self.data_points_filtered
        if until is not None:
            until_filtered = find_index_of_last_timestamp_before(x, self.data_points[until][0])
            if until_filtered < 0:
                self.prediction_failed_in_fit = True
                logger("MODEL-FIT").warn("Prediction failed in fit phase")
                return
            x = self.data_points_filtered[:until_filtered]
        self.pred_stride = int(len(assessment_windows) * self.pred_stride)
        self.horizon = max_horizon
        logger("MODEL-FIT").debug(
            "num_lags: {} / pred_stride: {} / fit_intercept: {} / horizon: {}".format(self.num_lags,
                                                                                      self.pred_stride,
                                                                                      self.fit_intercept,
                                                                                      self.horizon))
        occurrence_times = [data_point[0] for data_point in x]
        y = np.array(occurrence_times)
        X = y.reshape(-1, 1).copy()

        self.pipeline = self.get_pipeline()
        if self.learning_method == "deep":
            self.pipeline.fit(X[:-1].astype(np.float32), y[:-1].astype(np.float32))
        else:
            self.pipeline.fit(X[:-1], y[:-1])

    def predict(self, luck_average_windows, assessment_windows, from_idx=None):
        logger("MODEL-FIT").debug(
            "num_lags: {} / pred_stride: {} / fit_intercept: {} / horizon: {}".format(self.num_lags,
                                                                                      self.pred_stride,
                                                                                      self.fit_intercept,
                                                                                      self.horizon))
        from_idx = len(self.data_points) - 1 if from_idx is None else from_idx

        if self.prediction_failed_in_fit:
            return [None for i in range(from_idx, len(self.data_points))]

        from_idx_on_occurrences = find_index_of_last_timestamp_before(self.data_points_filtered,
                                                                      self.data_points[from_idx][0])
        if from_idx_on_occurrences < 0:
            logger("MODEL-PREDICT").warn("No block occurrence is found before the given points")
            return [None for i in range(from_idx, len(self.data_points))]

        # Find prediction on filtered data
        occurrence_timestamps = [data_point[0] for data_point in self.data_points_filtered]
        y = np.array(occurrence_timestamps)
        X = y.reshape(-1, 1).copy()

        to_scale = False
        if self.learning_method == "deep":
            X = X.astype(np.float32)
            to_scale = True
        future_points_prediction = self.pipeline.predict(X, start_idx=from_idx_on_occurrences, to_scale=to_scale)

        result = []
        # For each requested point, check if there is any close occurrence point to use for prediction
        for i in range(from_idx, len(self.data_points)):
            data_point = self.data_points[i]
            last_filtered_index = find_index_of_last_timestamp_before(self.data_points_filtered, data_point[0])
            if last_filtered_index < 0 or (data_point[0] -
                                           self.data_points_filtered[last_filtered_index][
                                               0]) > self.too_late_to_predict_time_threshold:
                result.append(None)
            else:
                data_point_last_prediction = future_points_prediction[last_filtered_index - from_idx_on_occurrences]
                prediction_age = data_point[0] - self.data_points_filtered[last_filtered_index][0]
                result.append([p - prediction_age for p in data_point_last_prediction])
        return result

    def decide(self, current_x, luck_average_windows, assessment_window, horizon_predictions, assessment_windows):
        logger("STEP-PREDICTOR").debug(
            "current_x: {} / avg_windows: {} / assmnt_window: {} / predictions: {} / assmnt_windows: {}".format(
                current_x,
                luck_average_windows,
                assessment_window,
                horizon_predictions,
                assessment_windows
            ))
        if horizon_predictions is None:
            return None
        window_length = TIME_10_MINUTES * assessment_window
        occurrences_count_in_window = 0
        for p in horizon_predictions:
            if 0 < p - current_x[0] <= window_length:
                occurrences_count_in_window += 1
        return occurrences_count_in_window >= self.positive_decision_occurrence_count_threshold

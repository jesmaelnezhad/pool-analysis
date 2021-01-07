import numpy as np
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sktime.classification.compose import ColumnEnsembleClassifier, TimeSeriesForestClassifier
from sktime.datasets import load_airline, load_arrow_head
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.load_data import load_from_tsfile_to_dataframe

from db import block_data
from main import prepare_pools, prepare_average_assessment_windows, prepare_average_luck_windows
from prediction import predictor

if __name__ == "__main__":
    data_2d_list = [[10, -1, 2], [10, +1, 3], [10, -1, 4], [10, +1, 5], [10, -1, 6], [10, +1, 7], [10, -1, 8],
                    [10, +1, 9]]
    X, y = load_from_tsfile_to_dataframe('/home/jamshid/PycharmProjects/pool-analysis/prediction/test_pandas_data.ts')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classifier = ColumnEnsembleClassifier(estimators=[
        ("TSF1", TimeSeriesForestClassifier(n_estimators=100), [1]),
    ])
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(str(X_train))
    print(str(y_pred))
    print(str(accuracy_score(y_test, y_pred)))
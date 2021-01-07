import numpy as np
import pandas as pd
from sktime.utils.load_data import from_long_to_nested
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sktime.classification.compose import ColumnEnsembleClassifier, TimeSeriesForestClassifier
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.utils.load_data import from_long_to_nested

def generate_long_table(ts, num_cases=1):
    series_len = len(ts)
    num_dims = len(ts[0])

    rows_per_case = series_len * num_dims
    total_rows = num_cases * series_len * num_dims

    case_ids = np.empty(total_rows, dtype=np.int)
    idxs = np.empty(total_rows, dtype=np.int)
    dims = np.empty(total_rows, dtype=np.int)
    vals = np.random.rand(total_rows)

    for i in range(total_rows):
        case_ids[i] = int(i / rows_per_case)
        rem = i % rows_per_case
        dims[i] = int(rem / series_len)
        idxs[i] = rem % series_len
        vals[i] = ts[idxs[i]][dims[i]]

    df = pd.DataFrame()
    df["case_id"] = pd.Series(case_ids)
    df["dim_id"] = pd.Series(dims)
    df["reading_id"] = pd.Series(idxs)
    df["value"] = pd.Series(vals)
    return df


case1 = ([
    (23534575, [10, -1, 2]),
    (23545575, [10, +1, 3]),
    (23564575, [10, -1, 4]),
    (23564575, [10, +1, 5]),
    (23564575, [10, -1, 6])
], 'A')
case2 = ([
    (23534575, [10, -1, 2]),
    (23545575, [10, +1, 3]),
    (23564575, [10, -1, 4]),
    (23564575, [10, +1, 5]),
    (23564575, [10, -1, 6])
], 'B')
case3 = ([
    (23534575, [10, -1, 2]),
    (23545575, [10, +1, 3]),
    (23564575, [10, -1, 4]),
    (23564575, [10, +1, 5]),
    (23564575, [10, -1, 6])
], 'A')

data = [case1, case2, case3]

# data -> our function -> (X_nested, y)

X = generate_long_table(ts)
X.head()

X_nested = from_long_to_nested(X)
X_nested.head()
y = np.array(['a'])  # , 'b', 'a', 'b', 'a', 'b', 'a', 'b'])

print(X_nested)

X_train, X_test, y_train, y_test = train_test_split(X_nested, y)
print(X.head())
classifier = ColumnEnsembleClassifier(estimators=[
    ("TSF1", TimeSeriesForestClassifier(n_estimators=100), [1]),
    ("TSF2", TimeSeriesForestClassifier(n_estimators=100), [2]),
])
classifier.fit(X_train, y_train)

# Use the test portion of data for prediction so we can understand how accurate our model was learned
y_pred = classifier.predict(X_test)
# Use the native `accuracy_score` method to calculate the accuracy based on the test outcomes and the predicted outcomes
print("Accuracy score is: " + str(accuracy_score(y_test, y_pred)))


def generate_example_long_table(num_cases=50, series_len=20, num_dims=2):
    rows_per_case = series_len * num_dims
    total_rows = num_cases * series_len * num_dims

    case_ids = np.empty(total_rows, dtype=np.int)
    idxs = np.empty(total_rows, dtype=np.int)
    dims = np.empty(total_rows, dtype=np.int)
    vals = np.random.rand(total_rows)
    for i in range(total_rows):
        case_ids[i] = int(i / rows_per_case)
        rem = i % rows_per_case
        dims[i] = int(rem / series_len)
        idxs[i] = rem % series_len

    df = pd.DataFrame()
    df["case_id"] = pd.Series(case_ids)
    df["dim_id"] = pd.Series(dims)
    df["reading_id"] = pd.Series(idxs)
    df["value"] = pd.Series(vals)
    return df


X = generate_example_long_table(num_cases=50, series_len=20, num_dims=4)
X.head()
print(len(X))



X_nested = from_long_to_nested(X)
X_nested.head()
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FeatureUnion

from skits.feature_extraction import AutoregressiveTransformer
from skits.pipeline import ForecasterPipeline
from skits.preprocessing import ReversibleImputer, HorizonTransformer
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# lin_pipeline = ForecasterPipeline([
#     # Convert the `y` target into a horizon
#     ('pre_horizon', HorizonTransformer(horizon=2)),
#     ('pre_reversible_imputer', ReversibleImputer(y_only=True)),
#     ('features', FeatureUnion([
#         # Generate a week's worth of autoregressive features
#         ('ar_features', AutoregressiveTransformer(num_lags=2)),
#     ])),
#     ('post_feature_imputer', ReversibleImputer()),
#     ('regressor', MultiOutputRegressor(LinearRegression(fit_intercept=False),
#                                        n_jobs=6))
# ])
#
# X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
# X2 = np.array([[4, 4], [5, 5]])
# y = np.array([0, 1, 2, 3])
# lin_pipeline = lin_pipeline.fit(X.reshape(-1, 1), y)
# prediction = lin_pipeline.predict(X2.reshape(-1, 1))
# print(str(prediction))

y = np.arange(10, dtype=np.float32)
ht = HorizonTransformer(horizon=3)
y_horizon = ht.fit_transform(y.reshape(-1, 1))
print(str(y_horizon))


# a = [[0,1,1], [0,2,1], [2,1,2]]
# a2 = np.array(a)
# print(str(a2))
# X = a2.reshape(-1,1)
# print(str(X[:1]))

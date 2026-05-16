import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from config import RANDOM_STATE

from prepocessing.feature_engineering import make_new_features

def make_submission(X, y, best_params, numeric_features, nominal_features, ordinal_features):
    test_df = pd.read_csv("data/test.csv")

    test_id = test_df["Id"]

    X_test, _, _, _ = make_new_features(test_df)

    categorical_features = nominal_features + ordinal_features

    for val in categorical_features:
        X[val] = X[val].fillna("Unknown").astype(str)
        X_test[val] = X_test[val].fillna("Unknown").astype(str)



    model = CatBoostRegressor(**best_params, random_state=RANDOM_STATE, verbose=100)

    model.fit(X, y, cat_features=categorical_features)

    preds_log = model.predict(X_test)

    preds = np.expm1(preds_log)

    submission = pd.DataFrame({
        "Id": test_id,
        "SalePrice": preds
    })

    submission.to_csv('data/submission.csv', index=False)

    print("submission.csv saved!")
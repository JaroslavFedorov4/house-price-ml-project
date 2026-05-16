import pandas as pd 
import numpy as np

from config import TARGET, NUMERIC_FEATURES, ORDINAL_FEATURES, NOMINAL_FEATURES
from prepocessing.feature_engineering import make_new_features

def load_data(path):
    df = pd.read_csv(path)

    X, numeric_features, nominal_features, ordinal_features = make_new_features(df)
    y = np.log1p(df[TARGET])

    return X, y, numeric_features, nominal_features, ordinal_features
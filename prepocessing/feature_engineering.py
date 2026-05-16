import numpy as np
import pandas as pd

from config import TARGET, NUMERIC_FEATURES, ORDINAL_FEATURES, NOMINAL_FEATURES


def make_new_features(df):
    df = df.copy()
    nominal_features = NOMINAL_FEATURES.copy()
    ordinal_features = ORDINAL_FEATURES.copy()
    numeric_features = NUMERIC_FEATURES.copy()

    df["QualityArea"] = (df["OverallQual"] * df["GrLivArea"])

    df["TotalArea"] = (df["GrLivArea"] + df["TotalBsmtSF"])

    df["QualityGarage"] = (df["OverallQual"] * df["GarageCars"])

    numeric_features.extend(["QualityArea", "TotalArea", "QualityGarage"])

    numeric_features = [
        col for col in numeric_features
        if col in df.columns
    ]

    ordinal_features = [
        col for col in ordinal_features
        if col in df.columns
    ]

    nominal_features = [
        col for col in nominal_features
        if col in df.columns
    ]

    features = numeric_features + nominal_features + ordinal_features

    return df[features], numeric_features, nominal_features, ordinal_features
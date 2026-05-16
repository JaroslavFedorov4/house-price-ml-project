from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import NUMERIC_FEATURES, ORDINAL_FEATURES, NOMINAL_FEATURES

def prep(numeric_features, nominal_features, ordinal_features):
    normalize_numeric = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    normalize_nominal = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    normalize_ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', normalize_numeric , numeric_features),
        ('nominal', normalize_nominal, nominal_features),
        ('ordinal', normalize_ordinal, ordinal_features)
    ])

    return preprocessor

def prep_NN(numeric_features, nominal_features, ordinal_features):
    normalize_numeric = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    normalize_nominal = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    normalize_ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor_num = ColumnTransformer([
        ('num', normalize_numeric, numeric_features)
    ])

    preprocessor_nominal = ColumnTransformer([
        ('nominal', normalize_nominal, nominal_features)
    ])

    preprocessor_ordinal = ColumnTransformer([
        ('ordinal', normalize_ordinal, ordinal_features)
    ])

    return preprocessor_num, preprocessor_nominal, preprocessor_ordinal
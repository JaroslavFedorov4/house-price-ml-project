from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import torch

TARGET = "SalePrice"

ORDINAL_FEATURES = ['BsmtQual', 'HeatingQC', 'KitchenQual', 'ExterCond', 'GarageFinish']
NOMINAL_FEATURES = ['Neighborhood', 'MSZoning', 'Condition1', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'MasVnrType', 'Foundation', 'CentralAir']
NUMERIC_FEATURES = ['OverallQual', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'GrLivArea', 'TotRmsAbvGrd']


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

RANDOM_STATE = 42
N_SPLITS = 3
SCORING = "neg_mean_squared_error"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

models = {
    "BaseLineModelLinearRegression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet(),
    "KNN": KNeighborsRegressor(algorithm="brute", n_jobs=1),
    "RandomForestRegressor": RandomForestRegressor(random_state=RANDOM_STATE),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "XGBRegressor": XGBRegressor(random_state=RANDOM_STATE),
    "LGBMRegressor": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, force_col_wise=True, verbosity=-1),
    "Catboost": CatBoostRegressor(random_state=RANDOM_STATE, verbose=100, loss_function="RMSE")
}

param_grids = {
    "BaseLineModelLinearRegression": {},
    "Lasso": {
        'model__alpha': [0.0001, 0.0005, 0.001, 0.005]
    },
    "Ridge": {
        'model__alpha': [0.1, 1, 5]
    },
    "ElasticNet": {
        'model__alpha': [0.0005, 0.001, 0.005],
        'model__l1_ratio': [0.2, 0.5, 0.8]
    },
    "KNN": {
        'model__n_neighbors': [3],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    },
    "RandomForestRegressor": {
        'model__n_estimators': [64, 100],
        'model__max_features': [2, 3, 4],
        'model__max_depth': [3, 5, 7, 9],
        'model__bootstrap': [True, False]
    },
    "DecisionTreeRegressor": {
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [3, 4, 5],
        'model__min_samples_leaf': [1, 2, 3, 4, 5],
        'model__criterion': ['squared_error', 'friedman_mse']
    },
    "XGBRegressor": {
        'model__n_estimators': [100],
        'model__learning_rate': [0.05],
        'model__max_depth': [3],
        'model__subsample': [0.8],
        'model__colsample_bytree': [0.8],
        'model__reg_alpha': [0.001],
        'model__reg_lambda': [1]
    },
    "Catboost": {
        "iterations": [100, 200],
        "learning_rate": [0.05, 0.1],
        "depth": [3, 4],
        "early_stopping_rounds": [50, 100],
        "l2_leaf_reg": [3, 5]
    },
    "LGBMRegressor": {
        "model__n_estimators": [150],
        "model__learning_rate": [0.05],
        "model__num_leaves": [15],
        "model__max_depth": [3],
        'model__min_child_samples': [10],
        'model__reg_alpha': [0.001],
        'model__reg_lambda': [5],
        "model__verbosity": [-1]
    }
}
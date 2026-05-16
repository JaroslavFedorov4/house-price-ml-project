import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from config import RANDOM_STATE, N_SPLITS, SCORING, models, param_grids
from prepocessing.preprocess import prep
from catboost import Pool

def grid_search_in_model(model_name, model, X, y, numeric_features, nominal_features, ordinal_features):
    pipeline = Pipeline([
        ("preprocessor", prep(numeric_features, nominal_features, ordinal_features)),
        ("model", model)
    ])

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        scoring=SCORING,
        cv=cv,
        verbose=1,
        n_jobs=1
    )

    grid.fit(X, y)

    best_score = np.sqrt(-grid.best_score_)

    return {"model": model_name, "best_score": best_score, "best_params": grid.best_params_, "best_estimator": grid.best_estimator_}

def grid_search_in_model_catboost(model, X, y, nominal_features, ordinal_features):
    categorical_features = nominal_features + ordinal_features

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    best_score = float("inf")
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grids["Catboost"]):
        fold_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()

            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()

            X_train[categorical_features] = X_train[categorical_features].fillna("Unknown").astype(str)
            X_val[categorical_features] = X_val[categorical_features].fillna("Unknown").astype(str)

            train_pool = Pool(
                data=X_train,
                label=y_train,
                cat_features=categorical_features
            )

            val_pool = Pool(
                data=X_val,
                label=y_val,
                cat_features=categorical_features
            )

            current_model = clone(model)
            current_model.set_params(**params)

            current_model.fit(
                train_pool,
                eval_set = val_pool,
                use_best_model=True,
                verbose = 0
            )

            pred = current_model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, pred))
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)

        if mean_score < best_score:
            best_score = mean_score
            best_params = params
            best_model = clone(model)
            best_model.set_params(**params)
    
    X_full = X.copy()

    X_full[categorical_features] = X_full[categorical_features].fillna("Unknown").astype(str)

    train_pool = Pool(
        data=X_full, 
        label=y,
        cat_features=categorical_features
    )

    best_model = clone(model)

    best_model.set_params(**best_params)

    best_model.fit(
        train_pool,
        verbose = 0
    )
    
    return {
        "model": "Catboost",
        "best_score": best_score,
        "best_params": best_params,
        "best_estimator": best_model
    }

def grid_all_models(X, y, numeric_features, nominal_features, ordinal_features):
    results = []

    for model_name, model in models.items():
        print(f'\nTrainig model {model_name}')


        if model_name == "Catboost":
            result = grid_search_in_model_catboost(model, X, y, nominal_features, ordinal_features)
        else:
            result = grid_search_in_model(
                model_name=model_name,
                model=model,
                X=X, 
                y=y,
                numeric_features=numeric_features,
                nominal_features=nominal_features,
                ordinal_features=ordinal_features
            )

        results.append(result)

    result_df = pd.DataFrame([
        {
            "model": item["model"],
            "best_score": item["best_score"],
            "best_params": item["best_params"]
        }
        for item in results
    ])

    result_df = result_df.sort_values(
        by='best_score',
        ascending=True
    )

    return result_df, results
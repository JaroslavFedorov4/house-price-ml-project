import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from catboost import Pool

def average_pred(full_results, X, y, categorical_features):
    all_preds = []
    X_cat = X.copy()
    for item in full_results:
        model_name = item["model"]
        model = item["best_estimator"]
        if model_name == "Catboost":
            X_cat[categorical_features] = X_cat[categorical_features].astype(str)
            pred = model.predict(X_cat)
            all_preds.append(pred)
        else:
            pred = model.predict(X)
            all_preds.append(pred)

    pred_avg = np.mean(all_preds, axis=0)

    rmse = np.sqrt(mean_squared_error(y, pred_avg))

    return rmse

def stacking_acc(full_results, X_train, y_train, X_val, y_val, categorical_features, n_splits, random_state):
    n_samples = len(X_train)
    n_models = len(full_results)

    meta_X_train = np.zeros((n_samples, n_models))

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for model_idx, item in enumerate(full_results):
        model_name = item["model"]
        model_base = item["best_estimator"]

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx].copy()
            X_fold_val = X_train.iloc[val_idx].copy()

            y_fold_train = y_train.iloc[train_idx].copy()

            model = clone(model_base)

            if model_name == "Catboost":
                X_fold_train[categorical_features] = X_fold_train[categorical_features].fillna("Unknown").astype(str)
                X_fold_val[categorical_features] = X_fold_val[categorical_features].fillna("Unknown").astype(str)

                train_pool = Pool(
                    data=X_fold_train,
                    label=y_fold_train,
                    cat_features=categorical_features
                )

                model.fit(train_pool, verbose=0)

                pred = model.predict(X_fold_val)

            else:
                model.fit(X_fold_train, y_fold_train)
                pred = model.predict(X_fold_val)

            meta_X_train[val_idx, model_idx] = pred

    meta_model = Ridge(alpha=0.1)

    meta_model.fit(meta_X_train, y_train)

    meta_X_val = np.zeros((len(X_val), n_models))

    for model_idx, item in enumerate(full_results):
        model_name = item["model"]
        model = item["best_estimator"]

        if model_name == "Catboost":
            X_val_cat = X_val.copy()
            X_val_cat[categorical_features] = X_val_cat[categorical_features].fillna("Unknown").astype(str)

            pred = model.predict(X_val_cat)

        else:
            pred = model.predict(X_val)

        meta_X_val[:, model_idx] = pred

    final_pred = meta_model.predict(meta_X_val)

    rmse = np.sqrt(mean_squared_error(y_val, final_pred))

    return rmse
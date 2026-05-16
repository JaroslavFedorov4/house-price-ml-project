from utils.load_data import load_data
from models.train import grid_all_models
from models.train_nn import train_nn
from models.ensemble import average_pred, stacking_acc
from sklearn.model_selection import train_test_split
from config import N_SPLITS, RANDOM_STATE
from utils.submit import make_submission

def main():
    X, y, numeric_features, nominal_features, ordinal_features = load_data("data/train.csv")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    results_df, full_results = grid_all_models(X_train, y_train, numeric_features, nominal_features, ordinal_features)

    nn_result = train_nn(X_train, X_val, y_train, y_val, numeric_features, nominal_features, ordinal_features)

    nn_row = {
        "model": nn_result["model"],
        "best_score": nn_result["best_score"],
        "best_params": nn_result["best_params"]
    }

    results_df.loc[len(results_df)] = nn_row

    results_df = results_df.sort_values("best_score")

    catboost_params = next(
        item for item in full_results
        if item["model"] == "Catboost"
    )

    categorical_features = nominal_features + ordinal_features

    avg_rmse = average_pred(full_results, X_val, y_val, categorical_features)

    stack_rmse = stacking_acc(full_results, X_train, y_train, X_val, y_val, categorical_features, n_splits=N_SPLITS, random_state=RANDOM_STATE)

    make_submission(X, y, catboost_params["best_params"], numeric_features, nominal_features, ordinal_features)

    print("\nRESULTS:")
    print(results_df)
    print("\nAVERAGE RMSE:")
    print(avg_rmse)
    print("\nSTACKING RMSE:")
    print(stack_rmse)

if __name__ == "__main__":
    main()

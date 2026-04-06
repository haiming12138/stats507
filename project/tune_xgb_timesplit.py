import argparse
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DEFAULT_NUM_COLS = [
    "AddressScore",
    "PhoneScore",
    "MissingAddressScore",
    "MissingPhoneScore",
]

DEFAULT_CAT_COLS = [
    "State",
    "Partner",
    "DebtLevel",
    "PublisherZoneName",
    "PublisherCampaignName",
    "AdvertiserCampaignName",
    "CampaignDesign",
    "PageLength",
    "AdGroup_New",
]

TARGET_MAP = {
    "y_closed": lambda df: (df["CallStatus"] == "closed").astype(int),
    "y_ep_only": lambda df: (df["CallStatus"] == "good").astype(int),
    "y_bad": lambda df: (df["CallStatus"] == "bad").astype(int),
}


def prepare_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    if target not in TARGET_MAP:
        raise ValueError(f"Unsupported target '{target}'. Use one of {list(TARGET_MAP)}")

    data = df.copy()
    data["LeadCreated"] = pd.to_datetime(data["LeadCreated"], errors="coerce")
    data = data.sort_values("LeadCreated").reset_index(drop=True)
    y = TARGET_MAP[target](data)

    num_cols = [c for c in DEFAULT_NUM_COLS if c in data.columns]
    cat_cols = [c for c in DEFAULT_CAT_COLS if c in data.columns]

    X = data[num_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")

    return X, y, num_cols, cat_cols


def make_pipeline(
    num_cols: List[str],
    cat_cols: List[str],
    model_params: Dict,
    n_jobs: int,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=n_jobs,
        **model_params,
    )

    return Pipeline(steps=[("prep", preprocess), ("model", model)])


def evaluate_params(
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    params: Dict,
    n_splits: int,
    n_jobs: int,
) -> Dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pos = float(y_train.sum())
        neg = float((1 - y_train).sum())
        fold_params = dict(params)
        fold_params["scale_pos_weight"] = (neg / pos) if pos > 0 else 1.0

        pipe = make_pipeline(num_cols, cat_cols, fold_params, n_jobs=n_jobs)
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]

        X_train_trans = prep.fit_transform(X_train, y_train)
        X_test_trans = prep.transform(X_test)

        model.fit(
            X_train_trans,
            y_train,
            eval_set=[(X_test_trans, y_test)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_test_trans)[:, 1]

        row = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "pos_rate_test": float(y_test.mean()),
            "log_loss": log_loss(y_test, y_prob, labels=[0, 1]),
        }

        try:
            row["roc_auc"] = roc_auc_score(y_test, y_prob)
        except ValueError:
            row["roc_auc"] = np.nan

        try:
            row["pr_auc"] = average_precision_score(y_test, y_prob)
        except ValueError:
            row["pr_auc"] = np.nan

        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    return {
        "params": params,
        "fold_metrics": fold_df,
        "mean_roc_auc": float(fold_df["roc_auc"].mean(skipna=True)),
        "mean_pr_auc": float(fold_df["pr_auc"].mean(skipna=True)),
        "mean_log_loss": float(fold_df["log_loss"].mean(skipna=True)),
    }


def run_tuning(
    data: pd.DataFrame,
    target: str,
    n_splits: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict:
    X, y, num_cols, cat_cols = prepare_data(data, target=target)

    param_grid = {
        "n_estimators": [600, 900, 1200],
        "learning_rate": np.geomspace(0.01, 0.3, 30),
        "max_depth": [3, 4, 5, 10],
        "min_child_weight": [3, 5, 8],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": np.geomspace(0.13, 50, 20),
        "reg_alpha": np.geomspace(0.05, 33, 20),
        "early_stopping_rounds": [25]
    }

    sampled_params = list(
        ParameterSampler(
            param_distributions=param_grid,
            n_iter=n_iter,
            random_state=random_state,
        )
    )

    results = []
    for i, params in enumerate(sampled_params, start=1):
        out = evaluate_params(
            X,
            y,
            num_cols,
            cat_cols,
            params,
            n_splits=n_splits,
            n_jobs=n_jobs,
        )
        results.append(out)
        print(
            f"[{i:04d}/{len(sampled_params):04d}] ROC_AUC={out['mean_roc_auc']:.5f} "
            f"PR_AUC={out['mean_pr_auc']:.5f} "
            f"LogLoss={out['mean_log_loss']:.5f} "
            f"params={params}"
        )

    leaderboard = pd.DataFrame(
        [
            {
                **r["params"],
                "mean_roc_auc": r["mean_roc_auc"],
                "mean_pr_auc": r["mean_pr_auc"],
                "mean_log_loss": r["mean_log_loss"],
            }
            for r in results
        ]
    ).sort_values(["mean_roc_auc", "mean_pr_auc"], ascending=[False, False])

    best_idx = int(leaderboard.index[0])
    best_result = results[best_idx]
    best_params = best_result["params"]
    best_folds = best_result["fold_metrics"]

    return {
        "target": target,
        "n_splits": n_splits,
        "n_iter": n_iter,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "best_params": best_params,
        "best_fold_metrics": best_folds,
        "leaderboard": leaderboard.reset_index(drop=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time-aware hyperparameter tuning for XGBoost classification."
    )
    parser.add_argument("--input", required=True, help="Input data path (.csv or .parquet).")
    parser.add_argument(
        "--target",
        default="y_closed",
        choices=["y_closed", "y_ep_only", "y_bad"],
        help="Target to model.",
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of TimeSeriesSplit folds.")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of random parameter combinations to try.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for parameter sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top parameter rows to print from leaderboard.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="CPU cores for XGBoost. Use -1 to use all available cores.",
    )
    args = parser.parse_args()

    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    elif args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        raise ValueError("Only .csv and .parquet are supported for --input")

    output = run_tuning(
        df,
        target=args.target,
        n_splits=args.splits,
        n_iter=args.n_iter,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    with open(f'{args.target}_best_param.pkl', 'wb') as f:
        pickle.dump(output, f)

    print("\nBest params:")
    print(output["best_params"])
    print("\nBest params fold-level metrics:")
    print(output["best_fold_metrics"].round(5))
    print("\nLeaderboard (top rows):")
    print(output["leaderboard"].head(args.top_k).round(5))


if __name__ == "__main__":
    main()

# python tune_xgb_timesplit.py --input data.csv --target y_closed --n-iter 1000
# python tune_xgb_timesplit.py --input data.csv --target y_ep_only --n-iter 1000
# python tune_xgb_timesplit.py --input data.csv --target y_bad

import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna

from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from optuna.integration import LightGBMPruningCallback

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "featured" / "train_features.parquet"
TEST_PATH = BASE_DIR / "data" / "featured" / "test_features.parquet"

SUBMISSION_PATH = BASE_DIR / "submissions" / "lambdamart_submission.csv"
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

print("Loading data...")
train_df = pl.read_parquet(TRAIN_PATH)
test_df = pl.read_parquet(TEST_PATH)

# =========================================================
# RELEVANCE LABEL (LambdaMART target)
# =========================================================

train_df = train_df.with_columns(
    (
        pl.col("booking_bool") * 5 +
        pl.col("click_bool") * (1 - pl.col("booking_bool"))
    ).alias("relevance")
)

# =========================================================
# DROP NON-FEATURES
# =========================================================

drop_cols = [
    "click_bool", "booking_bool", "gross_bookings_usd",
    "relevance", "srch_id", "prop_id",
    "position", "date_time", "checkin_datetime"
]

feature_cols = [c for c in train_df.columns if c not in drop_cols]

# =========================================================
# CATEGORICAL FEATURES (IMPORTANT FOR LIGHTGBM)
# =========================================================
# LightGBM handles categoricals ONLY if:
# - dtype is pandas 'category'
# - or integer-encoded categorical columns
# =========================================================

categorical_cols = [
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "srch_destination_id"
]

# convert after pandas conversion later

# =========================================================
# TRAIN / VALIDATION SPLIT (GROUP SAFE)
# =========================================================

queries = train_df["srch_id"].unique().to_numpy()

gkf = GroupKFold(n_splits=5)

train_idx, val_idx = next(gkf.split(queries, groups=queries))

train_queries = queries[train_idx]
val_queries = queries[val_idx]

train_split = train_df.filter(pl.col("srch_id").is_in(train_queries)).sort("srch_id")
val_split   = train_df.filter(pl.col("srch_id").is_in(val_queries)).sort("srch_id")

# =========================================================
# FEATURE MATRICES
# =========================================================

X_train = train_split.select(feature_cols).to_pandas()
X_val   = val_split.select(feature_cols).to_pandas()

y_train = train_split["relevance"].to_numpy()
y_val   = val_split["relevance"].to_numpy()

# categorical conversion
for c in categorical_cols:
    if c in X_train.columns:
        X_train[c] = X_train[c].astype("category")
        X_val[c]   = X_val[c].astype("category")

# =========================================================
# GROUPS (CRITICAL ORDERING!)
# =========================================================

train_group = (
    train_split.group_by("srch_id").len()
    .sort("srch_id")["len"].to_list()
)

val_group = (
    val_split.group_by("srch_id").len()
    .sort("srch_id")["len"].to_list()
)

# =========================================================
# OPTUNA OBJECTIVE
# =========================================================

def objective(trial):

    params = {
        "objective": "lambdarank",
        "label_gain": [0, 1, 0, 0, 0, 5],
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "lambdarank_truncation_level": 5, 

        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),

        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),

        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),

        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),

        "verbosity": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
    val_data   = lgb.Dataset(X_val, label=y_val, group=val_group, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0), LightGBMPruningCallback(trial, "ndcg@5")]
    )

    trial.set_user_attr("best_iteration", model.best_iteration)

    preds = model.predict(X_val)

    df = pd.DataFrame({
        "srch_id": val_split["srch_id"].to_numpy(),
        "y": y_val,
        "p": preds
    })

    ndcgs = []
    for q, g in df.groupby("srch_id"):
        ndcgs.append(
            ndcg_score([g["y"].values], [g["p"].values], k=5)
        )

    return np.mean(ndcgs)

# =========================================================
# RUN OPTUNA
# =========================================================

print("Running Optuna...")
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50))
study.optimize(objective, n_trials=100, n_jobs=1)

print("Best params:", study.best_params)

# =========================================================
# TRAIN FINAL MODEL
# =========================================================

best_params = study.best_params
best_params.update({
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "verbosity": -1,
    "seed": 42
})

full_train = train_df.sort("srch_id")

X_full = full_train.select(feature_cols).to_pandas()
y_full = full_train["relevance"].to_numpy()

for c in categorical_cols:
    if c in X_full.columns:
        X_full[c] = X_full[c].astype("category")

full_group = (
    full_train.group_by("srch_id").len()
    .sort("srch_id")["len"].to_list()
)

full_dataset = lgb.Dataset(X_full, label=y_full, group=full_group)

final_model = lgb.train(
    best_params,
    full_dataset,
    num_boost_round=study.best_trial.user_attrs["best_iteration"]
)

# =========================================================
# TEST PREDICTIONS
# =========================================================

test_sorted = test_df.sort("srch_id")
X_test = test_sorted.select(feature_cols).to_pandas()

for c in categorical_cols:
    if c in X_test.columns:
        X_test[c] = X_test[c].astype("category")

test_pred = final_model.predict(X_test)

submission = (
    test_sorted
    .select(["srch_id", "prop_id"])
    .with_columns(pl.Series("prediction", test_pred))
    .sort(["srch_id", "prediction"], descending=[False, True])
    .select(["srch_id", "prop_id"])
)

submission.write_csv(SUBMISSION_PATH)

print("Saved:", SUBMISSION_PATH)
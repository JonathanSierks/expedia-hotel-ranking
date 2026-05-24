import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
import json

from pathlib import Path
from sklearn.metrics import ndcg_score
from optuna.integration import LightGBMPruningCallback

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH      = BASE_DIR / "data" / "featured" / "train_features.parquet"
VAL_PATH        = BASE_DIR / "data" / "featured" / "val_features.parquet"
TEST_PATH       = BASE_DIR / "data" / "featured" / "test_features.parquet"
SUBMISSION_PATH = BASE_DIR / "submissions" / "lambdamart_submission.csv"
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# CONFIG
# Set to True if feature_engineering.py was run with
# INCLUDE_CLICK_BOOKING_FEATURES = True
# =========================================================
USE_OOF_RATES = True   # set False if features were built without click/booking

# =========================================================
# LOAD DATA
# =========================================================

print("Loading data...")
train_split = pl.read_parquet(TRAIN_PATH)
val_split   = pl.read_parquet(VAL_PATH)
test_df     = pl.read_parquet(TEST_PATH)

# =========================================================
# COLUMNS
# =========================================================

# Never used as model features
ALWAYS_DROP = [
    "click_bool", "booking_bool", "gross_bookings_usd",
    "relevance", "srch_id", "prop_id",
    "position", "date_time", "checkin_datetime",
    "prop_booking_count", "prop_click_count",
]

CATEGORICAL_COLS = [
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "srch_destination_id",
    "prop_starrating",
    "prop_brand_bool",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "search_month",
    "search_day_of_week",
    "search_hour",
    "checkin_month",
    "checkin_day_of_week",
    "promotion_flag",
    "random_bool",
    "family_trip_flag",
    "cheapest_hotel_flag",
    "visitor_history_star_missing",
    "visitor_history_price_missing",
    "missing_star_rating_flag",
    "review_score_zero_flag",
    "review_score_missing_flag",
    "missing_historical_price_flag",
    "affinity_score_missing_flag",
    "distance_missing_flag",
    "location_score2_missing_flag",
    # dest_search_count is an integer count — treat as numeric,
    # NOT categorical. Frequency features are kept as numeric.
]

# =========================================================
# HELPERS
# =========================================================

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in ALWAYS_DROP
            and not c.endswith("_booking_count")
            and not c.endswith("_click_count")]

def to_pandas_with_cats(df, feature_cols):
    pdf = df.select(feature_cols).to_pandas()
    for c in CATEGORICAL_COLS:
        if c in pdf.columns:
            pdf[c] = pdf[c].fillna(-1).astype("int32").astype("category")
    return pdf

def make_group(df):
    return df.group_by("srch_id").len().sort("srch_id")["len"].to_list()

def compute_ndcg_per_query(preds, labels, srch_ids):
    df = pd.DataFrame({"srch_id": srch_ids, "y": labels, "p": preds})
    return float(np.mean([
        ndcg_score([g["y"].values], [g["p"].values], k=5)
        for _, g in df.groupby("srch_id")
    ]))

# =========================================================
# FIXED DATA PREP
# =========================================================

y_train = train_split["relevance"].to_numpy()
y_val   = val_split["relevance"].to_numpy()

train_group = make_group(train_split)
val_group   = make_group(val_split)

feature_cols = get_feature_cols(train_split)
X_train = to_pandas_with_cats(train_split, feature_cols)
X_val   = to_pandas_with_cats(val_split,   feature_cols)

print(f"Feature count: {len(feature_cols)}")
print(f"Features: {feature_cols[:10]}...")

# free_raw_data=False is required when reusing datasets across Optuna
# trials with varying min_data_in_leaf — see feature_pre_filter note below.
train_data = lgb.Dataset(X_train, label=y_train, group=train_group, free_raw_data=False)
val_data   = lgb.Dataset(X_val,   label=y_val,   group=val_group,   free_raw_data=False,
                         reference=train_data)

# =========================================================
# SPEED SETUP
# =========================================================

import os as _os

N_CORES = min(_os.cpu_count() or 4, 32)
print(f"Detected {_os.cpu_count()} logical cores → using {N_CORES} LightGBM threads")

# =========================================================
# OPTUNA OBJECTIVE
#
# Speed changes for compute server:
#  - num_threads = N_CORES (was 4) — biggest single speedup
#  - GBDT max rounds 5000 → 3000 (best solutions converge ~1000)
#  - DART_FIXED_ROUNDS 1000 → 600 (no pruning, so each DART
#    trial runs the full count; halving saves significant time)
#  - n_trials 300 → 150 (TPE already has signal from prior runs;
#    at ~3 min/trial with full cores this is ~7.5 hours)
# =========================================================

DART_FIXED_ROUNDS = 600

def objective(trial):

    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

    params = {
        "objective":   "lambdarank",
        "metric":      "ndcg",
        "ndcg_eval_at": [5],
        "label_gain":  [0, 1, 0, 0, 0, 5],
        "lambdarank_truncation_level": 10,
        "verbosity":   -1,
        "seed":        42,
        "num_threads": N_CORES,
        "feature_pre_filter": False,
        "boosting_type": boosting_type,

        # ---- Tree complexity ----
        "num_leaves":       trial.suggest_int("num_leaves", 20, 80),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 300, 5000, log=True),

        # ---- Column / row sampling ----
        "feature_fraction":        trial.suggest_float("feature_fraction", 0.25, 0.75),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.35, 0.85),
        "bagging_fraction":        trial.suggest_float("bagging_fraction", 0.4, 0.9),
        "bagging_freq":            trial.suggest_int("bagging_freq", 1, 5),

        # ---- Regularisation ----
        "lambda_l1":         trial.suggest_float("lambda_l1", 0.05, 100.0, log=True),
        "lambda_l2":         trial.suggest_float("lambda_l2", 0.05, 100.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
        "path_smooth":       trial.suggest_float("path_smooth", 0.0, 5.0),

        # ---- Learning rate ----
        # Upper bound raised to 0.12 to cover previous best results
        # (best params used lr=0.087 which was outside the old 0.08 cap)
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
    }

    if boosting_type == "dart":
        params["drop_rate"]    = trial.suggest_float("drop_rate", 0.05, 0.4)
        params["skip_drop"]    = trial.suggest_float("skip_drop", 0.3, 0.7)
        params["uniform_drop"] = trial.suggest_categorical("uniform_drop", [True, False])

    if boosting_type == "dart":
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            num_boost_round=DART_FIXED_ROUNDS,
            callbacks=[lgb.log_evaluation(0)],
        )
        train_ndcg = model.best_score["train"]["ndcg@5"]
        val_ndcg   = model.best_score["val"]["ndcg@5"]
        best_it    = DART_FIXED_ROUNDS
    else:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            num_boost_round=3000,
            callbacks=[
                lgb.early_stopping(150, verbose=False),
                lgb.log_evaluation(0),
                LightGBMPruningCallback(trial, "ndcg@5", valid_name="val"),
            ],
        )
        train_ndcg = model.best_score["train"]["ndcg@5"]
        val_ndcg   = model.best_score["val"]["ndcg@5"]
        best_it    = model.best_iteration

    gap = train_ndcg - val_ndcg

    trial.set_user_attr("best_iteration",  best_it)
    trial.set_user_attr("train_ndcg",      train_ndcg)
    trial.set_user_attr("val_ndcg",        val_ndcg)
    trial.set_user_attr("gap",             gap)
    trial.set_user_attr("boosting_type",   boosting_type)

    print(
        f"  Trial {trial.number:>3} | "
        f"[{boosting_type}] "
        f"iter={best_it:>4} | "
        f"train={train_ndcg:.5f} | val={val_ndcg:.5f} | "
        f"gap={gap:+.5f}"
    )

    return val_ndcg


# =========================================================
# RUN OPTUNA
# =========================================================

print("\nRunning Optuna...")

study = optuna.create_study(
    direction="maximize",
    # Raise n_warmup_steps to 120 to give low-LR GBDT trials
    # enough rounds before the pruner can kill them
    pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=120),
)

# Seed 1: previous all-time best (val=0.42247), low LR variant
study.enqueue_trial({
    "boosting_type": "gbdt",
    "num_leaves": 59, "max_depth": 6, "min_data_in_leaf": 1428,
    "feature_fraction": 0.306, "feature_fraction_bynode": 0.797,
    "bagging_fraction": 0.721, "bagging_freq": 3,
    "lambda_l1": 50.0, "lambda_l2": 84.0,
    "min_gain_to_split": 0.427, "path_smooth": 3.9,
    "learning_rate": 0.03,   # lower LR version of previous best
})
# Seed 2: most recent best (val=0.42237) — exact params as-is
study.enqueue_trial({
    "boosting_type": "gbdt",
    "num_leaves": 60, "max_depth": 6, "min_data_in_leaf": 2247,
    "feature_fraction": 0.3185281640058442, "feature_fraction_bynode": 0.7667289552245905,
    "bagging_fraction": 0.6924774194013738, "bagging_freq": 3,
    "lambda_l1": 35.579234885721114, "lambda_l2": 0.2085100585459029,
    "min_gain_to_split": 0.12387771198593384, "path_smooth": 4.064936589657529,
    "learning_rate": 0.0866232365440887,
})
# Seed 3: most recent best with lower LR (let it run longer)
study.enqueue_trial({
    "boosting_type": "gbdt",
    "num_leaves": 60, "max_depth": 6, "min_data_in_leaf": 2247,
    "feature_fraction": 0.3185281640058442, "feature_fraction_bynode": 0.7667289552245905,
    "bagging_fraction": 0.6924774194013738, "bagging_freq": 3,
    "lambda_l1": 35.579234885721114, "lambda_l2": 0.2085100585459029,
    "min_gain_to_split": 0.12387771198593384, "path_smooth": 4.064936589657529,
    "learning_rate": 0.02,   # ~4x more rounds at same quality
})
# Seed 4: GBDT with aggressive regularisation + very low LR
study.enqueue_trial({
    "boosting_type": "gbdt",
    "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 2000,
    "feature_fraction": 0.4, "feature_fraction_bynode": 0.6,
    "bagging_fraction": 0.6, "bagging_freq": 5,
    "lambda_l1": 10.0, "lambda_l2": 10.0,
    "min_gain_to_split": 0.2, "path_smooth": 2.0,
    "learning_rate": 0.01,
})
# Seed 5: DART — params near recent best, moderate drop rate
study.enqueue_trial({
    "boosting_type": "dart",
    "num_leaves": 60, "max_depth": 6, "min_data_in_leaf": 2247,
    "feature_fraction": 0.32, "feature_fraction_bynode": 0.77,
    "bagging_fraction": 0.69, "bagging_freq": 3,
    "lambda_l1": 35.0, "lambda_l2": 0.2,
    "min_gain_to_split": 0.12, "path_smooth": 4.0,
    "learning_rate": 0.05,
    "drop_rate": 0.1, "skip_drop": 0.5, "uniform_drop": False,
})
# Seed 6: DART with higher drop rate (stronger regularisation)
study.enqueue_trial({
    "boosting_type": "dart",
    "num_leaves": 40, "max_depth": 5, "min_data_in_leaf": 1500,
    "feature_fraction": 0.4, "feature_fraction_bynode": 0.5,
    "bagging_fraction": 0.6, "bagging_freq": 4,
    "lambda_l1": 20.0, "lambda_l2": 20.0,
    "min_gain_to_split": 0.3, "path_smooth": 2.0,
    "learning_rate": 0.03,
    "drop_rate": 0.2, "skip_drop": 0.4, "uniform_drop": True,
})

study.optimize(objective, n_trials=150, n_jobs=1)

# =========================================================
# POST-SEARCH DIAGNOSTICS
# =========================================================

print("\n" + "="*70)
print(f"Best val NDCG@5:  {study.best_value:.5f}")
print(f"Best params:      {study.best_params}")
best_iter = study.best_trial.user_attrs["best_iteration"]
best_gap  = study.best_trial.user_attrs["gap"]
print(f"Best iteration:   {best_iter}")
print(f"Train/val gap:    {best_gap:+.5f}")

print("\nTop 10 trials:")
rows = []
for t in study.trials:
    if t.value is not None:
        rows.append({
            "trial":      t.number,
            "val_ndcg":   t.value,
            "train_ndcg": t.user_attrs.get("train_ndcg", float("nan")),
            "gap":        t.user_attrs.get("gap", float("nan")),
            "iter":       t.user_attrs.get("best_iteration", -1),
            "boosting":   t.user_attrs.get("boosting_type", "?"),
        })
top10 = sorted(rows, key=lambda x: x["val_ndcg"], reverse=True)[:10]
for r in top10:
    print(
        f"  [{r['trial']:>3}] val={r['val_ndcg']:.5f}  "
        f"train={r['train_ndcg']:.5f}  gap={r['gap']:+.5f}  "
        f"iter={r['iter']}  [{r['boosting']}]"
    )
print("="*70)

# =========================================================
# SAVE TOP-10 PARAMS for use by predict.py
# =========================================================

# Collect full trial info for top-10 so predict.py can
# train a separate final model for each and save 10 submissions
top10_full = []
for t in study.trials:
    if t.value is None:
        continue
    top10_full.append({
        "trial":         t.number,
        "val_ndcg":      t.value,
        "train_ndcg":    t.user_attrs.get("train_ndcg", float("nan")),
        "gap":           t.user_attrs.get("gap", float("nan")),
        "best_iteration": t.user_attrs.get("best_iteration", -1),
        "boosting_type": t.user_attrs.get("boosting_type", "gbdt"),
        "params":        t.params,
    })

top10_full = sorted(top10_full, key=lambda x: x["val_ndcg"], reverse=True)[:10]

RESULTS_PATH = BASE_DIR / "submissions" / "best_params.json"
with open(RESULTS_PATH, "w") as f:
    json.dump(top10_full, f, indent=2)

print(f"\nTop-10 trial params saved to: {RESULTS_PATH}")
print("Run predict.py to train 10 final models and generate 10 submission files.")
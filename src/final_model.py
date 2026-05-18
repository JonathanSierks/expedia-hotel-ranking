# best model so far
# """
# predict.py
# ==========
# Trains the final LambdaMART model on the combined train+val split
# using the best hyperparameters found by train.py, then generates
# the Kaggle submission file.

# Usage:
#     python src/predict.py

# Prerequisites:
#     1. feature_engineering.py has been run (parquet files exist)
#     2. train.py has been run (best_params.json exists)

# You can also skip train.py and paste params manually — see
# MANUAL_PARAMS below.
# """

# import json
# import polars as pl
# import lightgbm as lgb
# import numpy as np
# import pandas as pd

# from pathlib import Path

# # =========================================================
# # PATHS
# # =========================================================

# BASE_DIR = Path(__file__).resolve().parent.parent

# TRAIN_PATH      = BASE_DIR / "data" / "featured" / "train_features.parquet"
# VAL_PATH        = BASE_DIR / "data" / "featured" / "val_features.parquet"
# TEST_PATH       = BASE_DIR / "data" / "featured" / "test_features.parquet"
# PARAMS_PATH     = BASE_DIR / "submissions" / "best_params.json"
# MODEL_PATH      = BASE_DIR / "submissions" / "final_model.txt"
# SUBMISSION_PATH = BASE_DIR / "submissions" / "lambdamart_submission.csv"
# SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

# # =========================================================
# # MANUAL PARAMS
# # If you want to skip train.py entirely, set USE_MANUAL=True
# # and paste the best params + iteration here.
# # =========================================================

# USE_MANUAL = True

# MANUAL_PARAMS = {
#     "num_leaves": 59,
#     "max_depth": 6,
#     "min_data_in_leaf": 1428,
#     "feature_fraction": 0.30589237048741985,
#     "feature_fraction_bynode": 0.7971801065165558,
#     "bagging_fraction": 0.721464375576001,
#     "bagging_freq": 3,
#     "lambda_l1": 50.159132370457534,
#     "lambda_l2": 83.94168652660339,
#     "min_gain_to_split": 0.4267237708735313,
#     "path_smooth": 3.90449002716333,
#     "learning_rate": 0.09087975075406851,
# }
# MANUAL_BEST_ITERATION = 1273

# # =========================================================
# # COLUMN LISTS  (must match train.py)
# # =========================================================

# ALWAYS_DROP = [
#     "click_bool", "booking_bool", "gross_bookings_usd",
#     "relevance", "srch_id", "prop_id",
#     "position", "date_time", "checkin_datetime",
#     "prop_booking_count", "prop_click_count",
# ]

# CATEGORICAL_COLS = [
#     "site_id",
#     "visitor_location_country_id",
#     "prop_country_id",
#     "srch_destination_id",
#     "prop_starrating",
#     "prop_brand_bool",
#     "srch_adults_count",
#     "srch_children_count",
#     "srch_room_count",
#     "srch_saturday_night_bool",
#     "search_month",
#     "search_day_of_week",
#     "search_hour",
#     "checkin_month",
#     "checkin_day_of_week",
#     "promotion_flag",
#     "random_bool",
#     "family_trip_flag",
#     "cheapest_hotel_flag",
#     "visitor_history_star_missing",
#     "visitor_history_price_missing",
#     "missing_star_rating_flag",
#     "review_score_zero_flag",
#     "review_score_missing_flag",
#     "missing_historical_price_flag",
#     "affinity_score_missing_flag",
#     "distance_missing_flag",
#     "location_score2_missing_flag",
# ]

# # =========================================================
# # HELPERS
# # =========================================================

# def get_feature_cols(df):
#     return [c for c in df.columns
#             if c not in ALWAYS_DROP
#             and not c.endswith("_booking_count")
#             and not c.endswith("_click_count")]

# def to_pandas_with_cats(df, feature_cols):
#     pdf = df.select(feature_cols).to_pandas()
#     for c in CATEGORICAL_COLS:
#         if c in pdf.columns:
#             pdf[c] = pdf[c].fillna(-1).astype("int32").astype("category")
#     return pdf

# def make_group(df):
#     return df.group_by("srch_id").len().sort("srch_id")["len"].to_list()

# def align_schemas(train_df, val_df):
#     """
#     Cast columns to matching dtypes before concat.

#     The OOF target encoding in feature_engineering.py writes
#     train columns as Float32 (from np.float32 arrays) but val
#     columns come from a Polars join which infers Float64.
#     Polars concat is strict about schema equality, so we cast
#     every column in val to match train's dtype.
#     """
#     casts = []
#     for col_name, train_dtype in zip(train_df.columns, train_df.dtypes):
#         val_dtype = val_df.schema.get(col_name)
#         if val_dtype is not None and val_dtype != train_dtype:
#             casts.append(pl.col(col_name).cast(train_dtype))
#     if casts:
#         val_df = val_df.with_columns(casts)
#     return val_df

# # =========================================================
# # LOAD PARAMS
# # =========================================================

# if USE_MANUAL:
#     best_params   = MANUAL_PARAMS
#     best_iter     = MANUAL_BEST_ITERATION
#     print("Using manually specified hyperparameters.")
# else:
#     if not PARAMS_PATH.exists():
#         raise FileNotFoundError(
#             f"No params file found at {PARAMS_PATH}.\n"
#             "Run train.py first, or set USE_MANUAL=True and fill in MANUAL_PARAMS."
#         )
#     with open(PARAMS_PATH) as f:
#         results = json.load(f)
#     best_params = results["best_params"]
#     best_iter   = results["best_iteration"]
#     print(f"Loaded params from {PARAMS_PATH}")
#     print(f"  Best val NDCG@5 : {results['best_val_ndcg']:.5f}")
#     print(f"  Train/val gap   : {results['best_gap']:+.5f}")

# print(f"\nBest params: {best_params}")
# print(f"Best iteration: {best_iter}")

# # =========================================================
# # LOAD DATA
# # =========================================================

# print("\nLoading parquet files...")
# train_split = pl.read_parquet(TRAIN_PATH)
# val_split   = pl.read_parquet(VAL_PATH)
# test_df     = pl.read_parquet(TEST_PATH)

# print(f"  train_split : {len(train_split):,} rows")
# print(f"  val_split   : {len(val_split):,} rows")
# print(f"  test_df     : {len(test_df):,} rows")

# # =========================================================
# # COMBINE TRAIN + VAL
# #
# # The Float32/Float64 schema mismatch fix:
# #   - train_split has prop_booking_rate_oof / prop_ctr_oof
# #     as Float32 (written by np.full(..., dtype=np.float32))
# #   - val_split has those columns as Float64 (from a Polars
# #     join, which defaults to Float64 for float literals)
# #   - pl.concat is strict: both frames must have identical
# #     dtypes per column.
# #   Fix: cast val_split columns to match train_split dtypes
# #   before concatenating.
# # =========================================================

# print("\nAligning schemas (fixing Float32/Float64 mismatch)...")
# val_split = align_schemas(train_split, val_split)

# train_combined = pl.concat([train_split, val_split]).sort("srch_id")
# print(f"  Combined    : {len(train_combined):,} rows | "
#       f"{train_combined['srch_id'].n_unique():,} queries")

# # =========================================================
# # PREPARE FEATURES
# # =========================================================

# feature_cols = get_feature_cols(train_combined)
# print(f"\nFeature count: {len(feature_cols)}")

# X_full     = to_pandas_with_cats(train_combined, feature_cols)
# y_full     = train_combined["relevance"].to_numpy()
# full_group = make_group(train_combined)

# # =========================================================
# # FINAL MODEL
# # =========================================================

# final_params = dict(best_params)
# final_params.update({
#     "objective":              "lambdarank",
#     "metric":                 "ndcg",
#     "ndcg_eval_at":           [5],
#     "label_gain":             [0, 1, 0, 0, 0, 5],
#     "lambdarank_truncation_level": 10,
#     "verbosity":              -1,
#     "seed":                   42,
#     "num_threads":            4,
#     # Required when min_data_in_leaf varies — see train.py comment
#     "feature_pre_filter":     False,
# })

# print(f"\nTraining final model on {len(train_combined):,} rows "
#       f"for {best_iter} rounds...")

# final_model = lgb.train(
#     final_params,
#     lgb.Dataset(X_full, label=y_full, group=full_group),
#     num_boost_round=best_iter,
#     callbacks=[lgb.log_evaluation(100)],
# )

# # Save model so you can reload it later without retraining
# final_model.save_model(str(MODEL_PATH))
# print(f"Model saved to: {MODEL_PATH}")

# # =========================================================
# # FEATURE IMPORTANCE
# # =========================================================

# importance = pd.DataFrame({
#     "feature":    final_model.feature_name(),
#     "importance": final_model.feature_importance(importance_type="gain"),
# }).sort_values("importance", ascending=False)

# print(f"\nTop 30 features by gain ({len(importance)} total):")
# print(importance.head(30).to_string(index=False))

# # =========================================================
# # SUBMISSION
# # =========================================================

# print("\nGenerating predictions on test set...")
# test_sorted = test_df.sort("srch_id")
# X_test      = to_pandas_with_cats(test_sorted, feature_cols)
# test_pred   = final_model.predict(X_test)

# submission = (
#     test_sorted.select(["srch_id", "prop_id"])
#     .with_columns(pl.Series("prediction", test_pred))
#     .sort(["srch_id", "prediction"], descending=[False, True])
#     .select(["srch_id", "prop_id"])
# )

# submission.write_csv(SUBMISSION_PATH)
# print(f"\nSubmission saved : {SUBMISSION_PATH}")
# print(f"Rows             : {len(submission):,}")
# print(f"Unique queries   : {submission['srch_id'].n_unique():,}")

# second best model so far
# """
# predict.py
# ==========
# Trains the final LambdaMART model on the combined train+val split
# using the best hyperparameters found by train.py, then generates
# the Kaggle submission file.

# Usage:
#     python src/predict.py

# Prerequisites:
#     1. feature_engineering.py has been run (parquet files exist)
#     2. train.py has been run (best_params.json exists)

# You can also skip train.py and paste params manually — see
# MANUAL_PARAMS below.
# """

# import json
# import polars as pl
# import lightgbm as lgb
# import numpy as np
# import pandas as pd

# from pathlib import Path

# # =========================================================
# # PATHS
# # =========================================================

# BASE_DIR = Path(__file__).resolve().parent.parent

# TRAIN_PATH      = BASE_DIR / "data" / "featured" / "train_features.parquet"
# VAL_PATH        = BASE_DIR / "data" / "featured" / "val_features.parquet"
# TEST_PATH       = BASE_DIR / "data" / "featured" / "test_features.parquet"
# PARAMS_PATH     = BASE_DIR / "submissions" / "best_params.json"
# MODEL_PATH      = BASE_DIR / "submissions" / "final_model.txt"
# SUBMISSION_PATH = BASE_DIR / "submissions" / "lambdamart_submission.csv"
# SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

# # =========================================================
# # MANUAL PARAMS
# # If you want to skip train.py entirely, set USE_MANUAL=True
# # and paste the best params + iteration here.
# # =========================================================

# USE_MANUAL = False

# MANUAL_PARAMS = {
#     "num_leaves": 59,
#     "max_depth": 6,
#     "min_data_in_leaf": 1428,
#     "feature_fraction": 0.30589237048741985,
#     "feature_fraction_bynode": 0.7971801065165558,
#     "bagging_fraction": 0.721464375576001,
#     "bagging_freq": 3,
#     "lambda_l1": 50.159132370457534,
#     "lambda_l2": 83.94168652660339,
#     "min_gain_to_split": 0.4267237708735313,
#     "path_smooth": 3.90449002716333,
#     "learning_rate": 0.09087975075406851,
# }
# MANUAL_BEST_ITERATION = 1273

# # =========================================================
# # COLUMN LISTS  (must match train.py)
# # =========================================================

# ALWAYS_DROP = [
#     "click_bool", "booking_bool", "gross_bookings_usd",
#     "relevance", "srch_id", "prop_id",
#     "position", "date_time", "checkin_datetime",
#     "prop_booking_count", "prop_click_count",
# ]

# CATEGORICAL_COLS = [
#     "site_id",
#     "visitor_location_country_id",
#     "prop_country_id",
#     "srch_destination_id",
#     "prop_starrating",
#     "prop_brand_bool",
#     "srch_adults_count",
#     "srch_children_count",
#     "srch_room_count",
#     "srch_saturday_night_bool",
#     "search_month",
#     "search_day_of_week",
#     "search_hour",
#     "checkin_month",
#     "checkin_day_of_week",
#     "promotion_flag",
#     "random_bool",
#     "family_trip_flag",
#     "cheapest_hotel_flag",
#     "visitor_history_star_missing",
#     "visitor_history_price_missing",
#     "missing_star_rating_flag",
#     "review_score_zero_flag",
#     "review_score_missing_flag",
#     "missing_historical_price_flag",
#     "affinity_score_missing_flag",
#     "distance_missing_flag",
#     "location_score2_missing_flag",
#     # Frequency features (_freq columns) are kept as numeric — not listed here.
# ]

# # =========================================================
# # HELPERS
# # =========================================================

# def get_feature_cols(df):
#     return [c for c in df.columns
#             if c not in ALWAYS_DROP
#             and not c.endswith("_booking_count")
#             and not c.endswith("_click_count")]

# def to_pandas_with_cats(df, feature_cols):
#     pdf = df.select(feature_cols).to_pandas()
#     for c in CATEGORICAL_COLS:
#         if c in pdf.columns:
#             pdf[c] = pdf[c].fillna(-1).astype("int32").astype("category")
#     return pdf

# def make_group(df):
#     return df.group_by("srch_id").len().sort("srch_id")["len"].to_list()

# def align_schemas(train_df, val_df):
#     """
#     Cast columns to matching dtypes before concat.

#     The OOF target encoding in feature_engineering.py writes
#     train columns as Float32 (from np.float32 arrays) but val
#     columns come from a Polars join which infers Float64.
#     Polars concat is strict about schema equality, so we cast
#     every column in val to match train's dtype.
#     """
#     casts = []
#     for col_name, train_dtype in zip(train_df.columns, train_df.dtypes):
#         val_dtype = val_df.schema.get(col_name)
#         if val_dtype is not None and val_dtype != train_dtype:
#             casts.append(pl.col(col_name).cast(train_dtype))
#     if casts:
#         val_df = val_df.with_columns(casts)
#     return val_df

# # =========================================================
# # LOAD PARAMS
# # =========================================================

# if USE_MANUAL:
#     best_params   = MANUAL_PARAMS
#     best_iter     = MANUAL_BEST_ITERATION
#     print("Using manually specified hyperparameters.")
# else:
#     if not PARAMS_PATH.exists():
#         raise FileNotFoundError(
#             f"No params file found at {PARAMS_PATH}.\n"
#             "Run train.py first, or set USE_MANUAL=True and fill in MANUAL_PARAMS."
#         )
#     with open(PARAMS_PATH) as f:
#         results = json.load(f)
#     best_params = results["best_params"]
#     best_iter   = results["best_iteration"]
#     print(f"Loaded params from {PARAMS_PATH}")
#     print(f"  Best val NDCG@5 : {results['best_val_ndcg']:.5f}")
#     print(f"  Train/val gap   : {results['best_gap']:+.5f}")

# print(f"\nBest params: {best_params}")
# print(f"Best iteration: {best_iter}")

# # =========================================================
# # LOAD DATA
# # =========================================================

# print("\nLoading parquet files...")
# train_split = pl.read_parquet(TRAIN_PATH)
# val_split   = pl.read_parquet(VAL_PATH)
# test_df     = pl.read_parquet(TEST_PATH)

# print(f"  train_split : {len(train_split):,} rows")
# print(f"  val_split   : {len(val_split):,} rows")
# print(f"  test_df     : {len(test_df):,} rows")

# # =========================================================
# # COMBINE TRAIN + VAL
# #
# # The Float32/Float64 schema mismatch fix:
# #   - train_split has prop_booking_rate_oof / prop_ctr_oof
# #     as Float32 (written by np.full(..., dtype=np.float32))
# #   - val_split has those columns as Float64 (from a Polars
# #     join, which defaults to Float64 for float literals)
# #   - pl.concat is strict: both frames must have identical
# #     dtypes per column.
# #   Fix: cast val_split columns to match train_split dtypes
# #   before concatenating.
# # =========================================================

# print("\nAligning schemas (fixing Float32/Float64 mismatch)...")
# val_split = align_schemas(train_split, val_split)

# train_combined = pl.concat([train_split, val_split]).sort("srch_id")
# print(f"  Combined    : {len(train_combined):,} rows | "
#       f"{train_combined['srch_id'].n_unique():,} queries")

# # =========================================================
# # PREPARE FEATURES
# # =========================================================

# feature_cols = get_feature_cols(train_combined)
# print(f"\nFeature count: {len(feature_cols)}")

# X_full     = to_pandas_with_cats(train_combined, feature_cols)
# y_full     = train_combined["relevance"].to_numpy()
# full_group = make_group(train_combined)

# # =========================================================
# # FINAL MODEL
# # =========================================================

# final_params = dict(best_params)
# final_params.update({
#     "objective":              "lambdarank",
#     "metric":                 "ndcg",
#     "ndcg_eval_at":           [5],
#     "label_gain":             [0, 1, 0, 0, 0, 5],
#     "lambdarank_truncation_level": 10,
#     "verbosity":              -1,
#     "seed":                   42,
#     "num_threads":            4,
#     # Required when min_data_in_leaf varies — see train.py comment
#     "feature_pre_filter":     False,
# })

# print(f"\nTraining final model on {len(train_combined):,} rows "
#       f"for {best_iter} rounds...")

# final_model = lgb.train(
#     final_params,
#     lgb.Dataset(X_full, label=y_full, group=full_group),
#     num_boost_round=best_iter,
#     callbacks=[lgb.log_evaluation(100)],
# )

# # Save model so you can reload it later without retraining
# final_model.save_model(str(MODEL_PATH))
# print(f"Model saved to: {MODEL_PATH}")

# # =========================================================
# # FEATURE IMPORTANCE
# # =========================================================

# importance = pd.DataFrame({
#     "feature":    final_model.feature_name(),
#     "importance": final_model.feature_importance(importance_type="gain"),
# }).sort_values("importance", ascending=False)

# print(f"\nTop 30 features by gain ({len(importance)} total):")
# print(importance.head(30).to_string(index=False))

# # =========================================================
# # SUBMISSION
# # =========================================================

# print("\nGenerating predictions on test set...")
# test_sorted = test_df.sort("srch_id")
# X_test      = to_pandas_with_cats(test_sorted, feature_cols)
# test_pred   = final_model.predict(X_test)

# submission = (
#     test_sorted.select(["srch_id", "prop_id"])
#     .with_columns(pl.Series("prediction", test_pred))
#     .sort(["srch_id", "prediction"], descending=[False, True])
#     .select(["srch_id", "prop_id"])
# )

# submission.write_csv(SUBMISSION_PATH)
# print(f"\nSubmission saved : {SUBMISSION_PATH}")
# print(f"Rows             : {len(submission):,}")
# print(f"Unique queries   : {submission['srch_id'].n_unique():,}")

"""
final_model.py
==========
Trains the final LambdaMART model on the combined train+val split
using the best hyperparameters found by train.py, then generates
the Kaggle submission file.

Usage:
    python src/predict.py

Prerequisites:
    1. feature_engineering.py has been run (parquet files exist)
    2. train.py has been run (best_params.json exists)

You can also skip train.py and paste params manually — see
MANUAL_PARAMS below.
"""

import json
import os
import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd

from pathlib import Path

N_CORES = min(os.cpu_count() or 4, 32)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH      = BASE_DIR / "data" / "featured" / "train_features.parquet"
VAL_PATH        = BASE_DIR / "data" / "featured" / "val_features.parquet"
TEST_PATH       = BASE_DIR / "data" / "featured" / "test_features.parquet"
PARAMS_PATH     = BASE_DIR / "submissions" / "best_params.json"
MODEL_PATH      = BASE_DIR / "submissions" / "final_model.txt"
SUBMISSION_PATH = BASE_DIR / "submissions" / "lambdamart_submission.csv"
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# MANUAL PARAMS
# If you want to skip train.py entirely, set USE_MANUAL=True
# and paste the best params + iteration here.
# =========================================================

USE_MANUAL = False

MANUAL_PARAMS = {
    "num_leaves": 59,
    "max_depth": 6,
    "min_data_in_leaf": 1428,
    "feature_fraction": 0.30589237048741985,
    "feature_fraction_bynode": 0.7971801065165558,
    "bagging_fraction": 0.721464375576001,
    "bagging_freq": 3,
    "lambda_l1": 50.159132370457534,
    "lambda_l2": 83.94168652660339,
    "min_gain_to_split": 0.4267237708735313,
    "path_smooth": 3.90449002716333,
    "learning_rate": 0.09087975075406851,
}
MANUAL_BEST_ITERATION = 1273

# =========================================================
# COLUMN LISTS  (must match train.py)
# =========================================================

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
    # Frequency features (_freq columns) are kept as numeric — not listed here.
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

def align_schemas(train_df, val_df):
    """
    Cast columns to matching dtypes before concat.

    The OOF target encoding in feature_engineering.py writes
    train columns as Float32 (from np.float32 arrays) but val
    columns come from a Polars join which infers Float64.
    Polars concat is strict about schema equality, so we cast
    every column in val to match train's dtype.
    """
    casts = []
    for col_name, train_dtype in zip(train_df.columns, train_df.dtypes):
        val_dtype = val_df.schema.get(col_name)
        if val_dtype is not None and val_dtype != train_dtype:
            casts.append(pl.col(col_name).cast(train_dtype))
    if casts:
        val_df = val_df.with_columns(casts)
    return val_df

# =========================================================
# LOAD TOP-10 PARAMS
# =========================================================

if USE_MANUAL:
    # Wrap single manual config in a list so the loop below
    # works identically whether manual or loaded from file
    top10_trials = [{
        "trial":          -1,
        "val_ndcg":       0.0,
        "best_iteration": MANUAL_BEST_ITERATION,
        "boosting_type":  MANUAL_PARAMS.get("boosting_type", "gbdt"),
        "params":         MANUAL_PARAMS,
    }]
    print("Using manually specified hyperparameters (1 model).")
else:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"No params file found at {PARAMS_PATH}.\n"
            "Run train.py first, or set USE_MANUAL=True and fill in MANUAL_PARAMS."
        )
    with open(PARAMS_PATH) as f:
        top10_trials = json.load(f)

    print(f"Loaded {len(top10_trials)} trial configs from {PARAMS_PATH}")
    for i, t in enumerate(top10_trials):
        print(f"  #{i+1}  trial={t['trial']}  "
              f"val={t['val_ndcg']:.5f}  "
              f"iter={t['best_iteration']}  "
              f"[{t.get('boosting_type', 'gbdt')}]")

# =========================================================
# LOAD DATA  (once, reused for all 10 models)
# =========================================================

print("\nLoading parquet files...")
train_split = pl.read_parquet(TRAIN_PATH)
val_split   = pl.read_parquet(VAL_PATH)
test_df     = pl.read_parquet(TEST_PATH)

print(f"  train_split : {len(train_split):,} rows")
print(f"  val_split   : {len(val_split):,} rows")
print(f"  test_df     : {len(test_df):,} rows")

# =========================================================
# COMBINE TRAIN + VAL
# =========================================================

print("\nAligning schemas (fixing Float32/Float64 mismatch)...")
val_split = align_schemas(train_split, val_split)

train_combined = pl.concat([train_split, val_split]).sort("srch_id")
print(f"  Combined    : {len(train_combined):,} rows | "
      f"{train_combined['srch_id'].n_unique():,} queries")

# =========================================================
# PREPARE FEATURES  (once, reused for all 10 models)
# =========================================================

feature_cols = get_feature_cols(train_combined)
print(f"\nFeature count: {len(feature_cols)}")

X_full     = to_pandas_with_cats(train_combined, feature_cols)
y_full     = train_combined["relevance"].to_numpy()
full_group = make_group(train_combined)

test_sorted = test_df.sort("srch_id")
X_test      = to_pandas_with_cats(test_sorted, feature_cols)

# =========================================================
# TRAIN TOP-10 MODELS + GENERATE SUBMISSIONS
# =========================================================

print(f"\n{'='*70}")
print(f"Training {len(top10_trials)} models and generating submissions...")
print(f"{'='*70}")

for rank, trial_info in enumerate(top10_trials, start=1):

    trial_num    = trial_info["trial"]
    val_ndcg     = trial_info["val_ndcg"]
    best_iter    = trial_info["best_iteration"]
    boosting     = trial_info.get("boosting_type", "gbdt")
    trial_params = trial_info["params"]

    print(f"\n[{rank}/10] trial={trial_num}  val={val_ndcg:.5f}  "
          f"iter={best_iter}  [{boosting}]")

    # Build full param dict
    final_params = dict(trial_params)
    final_params.update({
        "objective":               "lambdarank",
        "metric":                  "ndcg",
        "ndcg_eval_at":            [5],
        "label_gain":              [0, 1, 0, 0, 0, 5],
        "lambdarank_truncation_level": 10,
        "verbosity":               -1,
        "seed":                    42,
        "num_threads":             N_CORES,
        "feature_pre_filter":      False,
    })
    # Ensure boosting_type is set (may live in params or top-level)
    if "boosting_type" not in final_params:
        final_params["boosting_type"] = boosting

    print(f"  Training on {len(train_combined):,} rows for {best_iter} rounds...")
    model = lgb.train(
        final_params,
        lgb.Dataset(X_full, label=y_full, group=full_group),
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(200)],
    )

    # Save model
    model_file = BASE_DIR / "submissions" / f"model_rank{rank:02d}_trial{trial_num}.txt"
    model.save_model(str(model_file))
    print(f"  Model saved: {model_file.name}")

    # Feature importance for rank-1 model only (avoid wall of text)
    if rank == 1:
        importance = pd.DataFrame({
            "feature":    model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        print(f"\n  Top 20 features by gain ({len(importance)} total):")
        print(importance.head(20).to_string(index=False))

    # Generate submission
    test_pred  = model.predict(X_test)
    submission = (
        test_sorted.select(["srch_id", "prop_id"])
        .with_columns(pl.Series("prediction", test_pred))
        .sort(["srch_id", "prediction"], descending=[False, True])
        .select(["srch_id", "prop_id"])
    )

    sub_file = (BASE_DIR / "submissions" /
                f"submission_rank{rank:02d}_trial{trial_num}_val{val_ndcg:.5f}.csv")
    submission.write_csv(sub_file)
    print(f"  Submission: {sub_file.name}  "
          f"({len(submission):,} rows, "
          f"{submission['srch_id'].n_unique():,} queries)")

print(f"\n{'='*70}")
print(f"Done. {len(top10_trials)} submission files saved to:")
print(f"  {BASE_DIR / 'submissions'}/")
print(f"{'='*70}")
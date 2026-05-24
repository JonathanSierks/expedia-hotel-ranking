# """
# fairness_analysis.py
# ====================
# Detects and mitigates performance disparity between family and
# non-family travellers in the Expedia hotel ranking model.

# Steps
# -----
# 1. Load already-engineered parquet files (train + val splits).
#    No re-running of feature engineering is needed: the OOF target
#    encoding in feature_engineering.py was already computed using
#    only train_split rows when assigning val_split rates, so there
#    is no leakage in the existing parquet files.

# 2. Train a baseline LambdaMART model on train_split using the
#    best hyperparameters found by Optuna (1219 rounds, no early
#    stopping).

# 3. Evaluate NDCG@5 overall, for family searches, and for
#    non-family searches on the validation split.

# 4. Re-weight the training data to upweight family trip rows
#    (underrepresented group) using inverse-frequency weights.

# 5. Train the same model architecture on the re-weighted data
#    and evaluate again with the same three metrics.

# 6. Print a comparison table and discuss the fairness-accuracy
#    trade-off.
# """

# import os
# import numpy as np
# import pandas as pd
# import polars as pl
# import lightgbm as lgb

# from pathlib import Path
# from sklearn.metrics import ndcg_score

# # =========================================================
# # PATHS
# # =========================================================

# BASE_DIR  = Path(__file__).resolve().parent.parent
# TRAIN_PATH = BASE_DIR / "data" / "featured" / "train_features.parquet"
# VAL_PATH   = BASE_DIR / "data" / "featured" / "val_features.parquet"

# # =========================================================
# # BEST HYPERPARAMETERS  (from Optuna run, trial 5)
# # best_iteration = 1219 → train for exactly this many rounds
# # =========================================================

# BEST_ITERATION = 1219

# BASE_PARAMS = {
#     "objective":                "lambdarank",
#     "metric":                   "ndcg",
#     "ndcg_eval_at":             [5],
#     "label_gain":               [0, 1, 0, 0, 0, 5],
#     "lambdarank_truncation_level": 10,
#     "verbosity":                -1,
#     "seed":                     42,
#     "num_threads":              min(os.cpu_count() or 4, 32),
#     "feature_pre_filter":       False,
#     "boosting_type":            "gbdt",
#     "num_leaves":               59,
#     "max_depth":                6,
#     "min_data_in_leaf":         1428,
#     "feature_fraction":         0.30589237048741985,
#     "feature_fraction_bynode":  0.7971801065165558,
#     "bagging_fraction":         0.721464375576001,
#     "bagging_freq":             3,
#     "lambda_l1":                50.159132370457534,
#     "lambda_l2":                83.94168652660339,
#     "min_gain_to_split":        0.4267237708735313,
#     "path_smooth":              3.90449002716333,
#     "learning_rate":            0.09087975075406851,
# }

# # =========================================================
# # COLUMN LISTS  (must match feature_engineering.py output)
# # =========================================================

# ALWAYS_DROP = [
#     "click_bool", "booking_bool", "gross_bookings_usd",
#     "relevance", "srch_id", "prop_id",
#     "position", "date_time", "checkin_datetime",
#     "prop_booking_count", "prop_click_count",
# ]

# CATEGORICAL_COLS = [
#     "site_id", "visitor_location_country_id", "prop_country_id",
#     "srch_destination_id", "prop_starrating", "prop_brand_bool",
#     "srch_adults_count", "srch_children_count", "srch_room_count",
#     "srch_saturday_night_bool", "search_month", "search_day_of_week",
#     "search_hour", "checkin_month", "checkin_day_of_week",
#     "promotion_flag", "random_bool", "family_trip_flag",
#     "cheapest_hotel_flag", "visitor_history_star_missing",
#     "visitor_history_price_missing", "missing_star_rating_flag",
#     "review_score_zero_flag", "review_score_missing_flag",
#     "missing_historical_price_flag", "affinity_score_missing_flag",
#     "distance_missing_flag", "location_score2_missing_flag",
# ]

# # =========================================================
# # HELPERS
# # =========================================================

# def get_feature_cols(df: pl.DataFrame) -> list[str]:
#     return [
#         c for c in df.columns
#         if c not in ALWAYS_DROP
#         and not c.endswith("_booking_count")
#         and not c.endswith("_click_count")
#     ]


# def to_pandas_with_cats(df: pl.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
#     pdf = df.select(feature_cols).to_pandas()
#     for c in CATEGORICAL_COLS:
#         if c in pdf.columns:
#             pdf[c] = pdf[c].fillna(-1).astype("int32").astype("category")
#     return pdf


# def make_group(df: pl.DataFrame) -> list[int]:
#     return df.group_by("srch_id").len().sort("srch_id")["len"].to_list()


# def align_schemas(train_df: pl.DataFrame, val_df: pl.DataFrame) -> pl.DataFrame:
#     """Cast val columns to match train dtypes (fixes Float32/Float64 mismatch)."""
#     casts = []
#     for col_name, train_dtype in zip(train_df.columns, train_df.dtypes):
#         val_dtype = val_df.schema.get(col_name)
#         if val_dtype is not None and val_dtype != train_dtype:
#             casts.append(pl.col(col_name).cast(train_dtype))
#     if casts:
#         val_df = val_df.with_columns(casts)
#     return val_df


# def compute_ndcg_at5(preds: np.ndarray,
#                      labels: np.ndarray,
#                      srch_ids: np.ndarray) -> float:
#     """Mean NDCG@5 across all queries that have at least one relevant item."""
#     df = pd.DataFrame({"srch_id": srch_ids, "y": labels, "p": preds})
#     scores = []
#     for _, g in df.groupby("srch_id"):
#         if g["y"].sum() > 0:
#             scores.append(
#                 ndcg_score([g["y"].values], [g["p"].values], k=5)
#             )
#     return float(np.mean(scores)) if scores else float("nan")


# def evaluate_fairness(preds: np.ndarray,
#                       val_df_pd: pd.DataFrame,
#                       label_col: str = "relevance",
#                       group_col: str = "srch_children_count") -> dict:
#     """
#     Compute NDCG@5 for the full val set, family travellers
#     (srch_children_count > 0), and non-family travellers.
#     """
#     srch_ids = val_df_pd["srch_id"].values
#     labels   = val_df_pd[label_col].values
#     children = val_df_pd[group_col].values

#     family_mask    = children > 0
#     nonfamily_mask = children == 0

#     ndcg_overall   = compute_ndcg_at5(preds, labels, srch_ids)
#     ndcg_family    = compute_ndcg_at5(preds[family_mask],
#                                        labels[family_mask],
#                                        srch_ids[family_mask])
#     ndcg_nonfamily = compute_ndcg_at5(preds[nonfamily_mask],
#                                        labels[nonfamily_mask],
#                                        srch_ids[nonfamily_mask])

#     n_family_rows    = int(family_mask.sum())
#     n_nonfamily_rows = int(nonfamily_mask.sum())
#     n_family_queries = int(
#         pd.Series(srch_ids[family_mask]).nunique()
#     )
#     n_nonfamily_queries = int(
#         pd.Series(srch_ids[nonfamily_mask]).nunique()
#     )

#     return {
#         "ndcg_overall":         ndcg_overall,
#         "ndcg_family":          ndcg_family,
#         "ndcg_nonfamily":       ndcg_nonfamily,
#         "gap (nonfam - fam)":   ndcg_nonfamily - ndcg_family,
#         "n_family_rows":        n_family_rows,
#         "n_nonfamily_rows":     n_nonfamily_rows,
#         "n_family_queries":     n_family_queries,
#         "n_nonfamily_queries":  n_nonfamily_queries,
#     }


# # =========================================================
# # LOAD DATA
# # =========================================================

# print("Loading parquet files...")
# train_split = pl.read_parquet(TRAIN_PATH)
# val_split   = pl.read_parquet(VAL_PATH)

# print(f"  train_split : {len(train_split):,} rows | "
#       f"{train_split['srch_id'].n_unique():,} queries")
# print(f"  val_split   : {len(val_split):,} rows   | "
#       f"{val_split['srch_id'].n_unique():,} queries")

# # Fix Float32/Float64 schema mismatch from OOF encoding
# val_split = align_schemas(train_split, val_split)

# # =========================================================
# # FAMILY vs NON-FAMILY SPLIT STATS
# # =========================================================

# n_total        = len(train_split)
# n_family       = int((train_split["srch_children_count"] > 0).sum())
# n_nonfamily    = n_total - n_family
# pct_family     = 100 * n_family / n_total

# print(f"\nTraining set group breakdown:")
# print(f"  Family trips    : {n_family:>10,}  ({pct_family:.1f}%)")
# print(f"  Non-family trips: {n_nonfamily:>10,}  ({100 - pct_family:.1f}%)")

# # Booking rates per group (to check for base-rate differences)
# fam_rows  = train_split.filter(pl.col("srch_children_count") > 0)
# nfam_rows = train_split.filter(pl.col("srch_children_count") == 0)

# print(f"\nBooking rates in training set:")
# print(f"  Family booking rate    : "
#       f"{float(fam_rows['booking_bool'].mean()):.4f}")
# print(f"  Non-family booking rate: "
#       f"{float(nfam_rows['booking_bool'].mean()):.4f}")

# # =========================================================
# # PREPARE FEATURES  (done once, reused for both models)
# # =========================================================

# feature_cols = get_feature_cols(train_split)
# print(f"\nFeature count: {len(feature_cols)}")

# X_train = to_pandas_with_cats(train_split, feature_cols)
# X_val   = to_pandas_with_cats(val_split,   feature_cols)

# y_train = train_split["relevance"].to_numpy()
# y_val   = val_split["relevance"].to_numpy()

# train_group = make_group(train_split)
# val_group   = make_group(val_split)

# # Keep val as pandas for fairness evaluation (need srch_id, children)
# val_df_pd = val_split.select(
#     ["srch_id", "relevance", "srch_children_count"]
# ).to_pandas()

# # =========================================================
# # MODEL 1: BASELINE  (no re-weighting)
# # =========================================================

# print("\n" + "="*60)
# print("BASELINE MODEL (no re-weighting)")
# print("="*60)

# baseline_dataset = lgb.Dataset(
#     X_train, label=y_train, group=train_group,
#     free_raw_data=False
# )

# print(f"Training for {BEST_ITERATION} rounds...")
# baseline_model = lgb.train(
#     BASE_PARAMS,
#     baseline_dataset,
#     num_boost_round=BEST_ITERATION,
#     callbacks=[lgb.log_evaluation(200)],
# )

# baseline_preds  = baseline_model.predict(X_val)
# baseline_result = evaluate_fairness(baseline_preds, val_df_pd)

# print("\nBaseline evaluation results:")
# for k, v in baseline_result.items():
#     if isinstance(v, float):
#         print(f"  {k:<25}: {v:.5f}")
#     else:
#         print(f"  {k:<25}: {v:,}")

# # =========================================================
# # RE-WEIGHTING
# #
# # Inverse-frequency weights for two groups:
# #
# #   w_family    = n_total / (2 × n_family)
# #   w_nonfamily = n_total / (2 × n_nonfamily)
# #
# # The factor of 2 (= number of groups) normalises the weights
# # so that their sum equals n_total, keeping gradient magnitudes
# # comparable to the unweighted case.
# #
# # Effect: each family trip row contributes more to the lambda
# # gradients during training, forcing the model to pay more
# # attention to errors on the underrepresented group.
# # =========================================================

# w_family    = n_total / (2 * n_family)
# w_nonfamily = n_total / (2 * n_nonfamily)

# children_train = train_split["srch_children_count"].to_numpy()
# sample_weights = np.where(children_train > 0, w_family, w_nonfamily)

# print(f"\nRe-weighting:")
# print(f"  w_family    = {n_total} / (2 × {n_family}) = {w_family:.4f}")
# print(f"  w_nonfamily = {n_total} / (2 × {n_nonfamily}) = {w_nonfamily:.4f}")
# print(f"  Total weight sum: {sample_weights.sum():.0f}  "
#       f"(expected: {n_total:,})")

# # =========================================================
# # MODEL 2: RE-WEIGHTED
# # =========================================================

# print("\n" + "="*60)
# print("RE-WEIGHTED MODEL")
# print("="*60)

# weighted_dataset = lgb.Dataset(
#     X_train, label=y_train, group=train_group,
#     weight=sample_weights,
#     free_raw_data=False
# )

# print(f"Training for {BEST_ITERATION} rounds...")
# weighted_model = lgb.train(
#     BASE_PARAMS,
#     weighted_dataset,
#     num_boost_round=BEST_ITERATION,
#     callbacks=[lgb.log_evaluation(200)],
# )

# weighted_preds  = weighted_model.predict(X_val)
# weighted_result = evaluate_fairness(weighted_preds, val_df_pd)

# print("\nRe-weighted evaluation results:")
# for k, v in weighted_result.items():
#     if isinstance(v, float):
#         print(f"  {k:<25}: {v:.5f}")
#     else:
#         print(f"  {k:<25}: {v:,}")

# # =========================================================
# # COMPARISON TABLE
# # =========================================================

# print("\n" + "="*60)
# print("FAIRNESS COMPARISON")
# print("="*60)

# metrics = [
#     "ndcg_overall",
#     "ndcg_family",
#     "ndcg_nonfamily",
#     "gap (nonfam - fam)",
# ]

# print(f"\n{'Metric':<25} {'Baseline':>12} {'Re-weighted':>12} {'Change':>10}")
# print("-" * 62)
# for m in metrics:
#     b = baseline_result[m]
#     w = weighted_result[m]
#     change = w - b
#     sign = "+" if change >= 0 else ""
#     print(f"{m:<25} {b:>12.5f} {w:>12.5f} {sign}{change:>9.5f}")

# print("\nGroup sizes (validation set):")
# print(f"  Family queries    : "
#       f"{baseline_result['n_family_queries']:,}  "
#       f"({baseline_result['n_family_rows']:,} rows)")
# print(f"  Non-family queries: "
#       f"{baseline_result['n_nonfamily_queries']:,}  "
#       f"({baseline_result['n_nonfamily_rows']:,} rows)")

# print("\nInterpretation:")
# gap_before = baseline_result["gap (nonfam - fam)"]
# gap_after  = weighted_result["gap (nonfam - fam)"]
# gap_change = gap_after - gap_before

# print(f"  Performance gap before mitigation: {gap_before:+.5f}")
# print(f"  Performance gap after  mitigation: {gap_after:+.5f}")
# print(f"  Gap reduction: {gap_change:+.5f} "
#       f"({'improved' if gap_change < 0 else 'worsened'})")

# overall_change = weighted_result["ndcg_overall"] - baseline_result["ndcg_overall"]
# print(f"  Overall NDCG@5 change: {overall_change:+.5f} "
#       f"(fairness-accuracy trade-off)")


"""
fairness_analysis.py
====================
Detects and mitigates performance disparity between family and
non-family travellers in the Expedia hotel ranking model.

Steps
-----
1. Load already-engineered parquet files (train + val splits).
   No re-running of feature engineering is needed: the OOF target
   encoding in feature_engineering.py was already computed using
   only train_split rows when assigning val_split rates, so there
   is no leakage in the existing parquet files.

2. Train a baseline LambdaMART model on train_split using the
   best hyperparameters found by Optuna (1219 rounds, no early
   stopping).

3. Evaluate NDCG@5 overall, for family searches, and for
   non-family searches on the validation split.

4. Re-weight the training data to upweight family trip rows
   (underrepresented group) using inverse-frequency weights.

5. Train the same model architecture on the re-weighted data
   and evaluate again with the same three metrics.

6. Print a comparison table and discuss the fairness-accuracy
   trade-off.
"""

import os
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

from pathlib import Path
from sklearn.metrics import ndcg_score

# =========================================================
# PATHS
# =========================================================

BASE_DIR  = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "featured" / "train_features.parquet"
VAL_PATH   = BASE_DIR / "data" / "featured" / "val_features.parquet"

# =========================================================
# BEST HYPERPARAMETERS  (from Optuna run, trial 5)
# best_iteration = 1219 → train for exactly this many rounds
# =========================================================

BEST_ITERATION = 1219

BASE_PARAMS = {
    "objective":                "lambdarank",
    "metric":                   "ndcg",
    "ndcg_eval_at":             [5],
    "label_gain":               [0, 1, 0, 0, 0, 5],
    "lambdarank_truncation_level": 10,
    "verbosity":                -1,
    "seed":                     42,
    "num_threads":              min(os.cpu_count() or 4, 32),
    "feature_pre_filter":       False,
    "boosting_type":            "gbdt",
    "num_leaves":               59,
    "max_depth":                6,
    "min_data_in_leaf":         1428,
    "feature_fraction":         0.30589237048741985,
    "feature_fraction_bynode":  0.7971801065165558,
    "bagging_fraction":         0.721464375576001,
    "bagging_freq":             3,
    "lambda_l1":                50.159132370457534,
    "lambda_l2":                83.94168652660339,
    "min_gain_to_split":        0.4267237708735313,
    "path_smooth":              3.90449002716333,
    "learning_rate":            0.09087975075406851,
}

# =========================================================
# COLUMN LISTS  (must match feature_engineering.py output)
# =========================================================

ALWAYS_DROP = [
    "click_bool", "booking_bool", "gross_bookings_usd",
    "relevance", "srch_id", "prop_id",
    "position", "date_time", "checkin_datetime",
    "prop_booking_count", "prop_click_count",
]

CATEGORICAL_COLS = [
    "site_id", "visitor_location_country_id", "prop_country_id",
    "srch_destination_id", "prop_starrating", "prop_brand_bool",
    "srch_adults_count", "srch_children_count", "srch_room_count",
    "srch_saturday_night_bool", "search_month", "search_day_of_week",
    "search_hour", "checkin_month", "checkin_day_of_week",
    "promotion_flag", "random_bool", "family_trip_flag",
    "cheapest_hotel_flag", "visitor_history_star_missing",
    "visitor_history_price_missing", "missing_star_rating_flag",
    "review_score_zero_flag", "review_score_missing_flag",
    "missing_historical_price_flag", "affinity_score_missing_flag",
    "distance_missing_flag", "location_score2_missing_flag",
]

# =========================================================
# HELPERS
# =========================================================

def get_feature_cols(df: pl.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in ALWAYS_DROP
        and not c.endswith("_booking_count")
        and not c.endswith("_click_count")
    ]


def to_pandas_with_cats(df: pl.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    pdf = df.select(feature_cols).to_pandas()
    for c in CATEGORICAL_COLS:
        if c in pdf.columns:
            pdf[c] = pdf[c].fillna(-1).astype("int32").astype("category")
    return pdf


def make_group(df: pl.DataFrame) -> list[int]:
    return df.group_by("srch_id").len().sort("srch_id")["len"].to_list()


def align_schemas(train_df: pl.DataFrame, val_df: pl.DataFrame) -> pl.DataFrame:
    """Cast val columns to match train dtypes (fixes Float32/Float64 mismatch)."""
    casts = []
    for col_name, train_dtype in zip(train_df.columns, train_df.dtypes):
        val_dtype = val_df.schema.get(col_name)
        if val_dtype is not None and val_dtype != train_dtype:
            casts.append(pl.col(col_name).cast(train_dtype))
    if casts:
        val_df = val_df.with_columns(casts)
    return val_df


def compute_ndcg_at5(preds: np.ndarray,
                     labels: np.ndarray,
                     srch_ids: np.ndarray) -> float:
    """Mean NDCG@5 across all queries that have at least one relevant item."""
    df = pd.DataFrame({"srch_id": srch_ids, "y": labels, "p": preds})
    scores = []
    for _, g in df.groupby("srch_id"):
        if g["y"].sum() > 0:
            scores.append(
                ndcg_score([g["y"].values], [g["p"].values], k=5)
            )
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_fairness(preds: np.ndarray,
                      val_df_pd: pd.DataFrame,
                      label_col: str = "relevance",
                      group_col: str = "srch_children_count") -> dict:
    """
    Compute NDCG@5 for the full val set, family travellers
    (srch_children_count > 0), and non-family travellers.
    """
    srch_ids = val_df_pd["srch_id"].values
    labels   = val_df_pd[label_col].values
    children = val_df_pd[group_col].values

    family_mask    = children > 0
    nonfamily_mask = children == 0

    ndcg_overall   = compute_ndcg_at5(preds, labels, srch_ids)
    ndcg_family    = compute_ndcg_at5(preds[family_mask],
                                       labels[family_mask],
                                       srch_ids[family_mask])
    ndcg_nonfamily = compute_ndcg_at5(preds[nonfamily_mask],
                                       labels[nonfamily_mask],
                                       srch_ids[nonfamily_mask])

    n_family_rows    = int(family_mask.sum())
    n_nonfamily_rows = int(nonfamily_mask.sum())
    n_family_queries = int(
        pd.Series(srch_ids[family_mask]).nunique()
    )
    n_nonfamily_queries = int(
        pd.Series(srch_ids[nonfamily_mask]).nunique()
    )

    return {
        "ndcg_overall":         ndcg_overall,
        "ndcg_family":          ndcg_family,
        "ndcg_nonfamily":       ndcg_nonfamily,
        "gap (nonfam - fam)":   ndcg_nonfamily - ndcg_family,
        "n_family_rows":        n_family_rows,
        "n_nonfamily_rows":     n_nonfamily_rows,
        "n_family_queries":     n_family_queries,
        "n_nonfamily_queries":  n_nonfamily_queries,
    }


# =========================================================
# LOAD DATA
# =========================================================

print("Loading parquet files...")
train_split = pl.read_parquet(TRAIN_PATH)
val_split   = pl.read_parquet(VAL_PATH)

print(f"  train_split : {len(train_split):,} rows | "
      f"{train_split['srch_id'].n_unique():,} queries")
print(f"  val_split   : {len(val_split):,} rows   | "
      f"{val_split['srch_id'].n_unique():,} queries")

# Fix Float32/Float64 schema mismatch from OOF encoding
val_split = align_schemas(train_split, val_split)

# =========================================================
# FAMILY vs NON-FAMILY SPLIT STATS
# =========================================================

n_total        = len(train_split)
n_family       = int((train_split["srch_children_count"] > 0).sum())
n_nonfamily    = n_total - n_family
pct_family     = 100 * n_family / n_total

print(f"\nTraining set group breakdown:")
print(f"  Family trips    : {n_family:>10,}  ({pct_family:.1f}%)")
print(f"  Non-family trips: {n_nonfamily:>10,}  ({100 - pct_family:.1f}%)")

# Booking rates per group (to check for base-rate differences)
fam_rows  = train_split.filter(pl.col("srch_children_count") > 0)
nfam_rows = train_split.filter(pl.col("srch_children_count") == 0)

print(f"\nBooking rates in training set:")
print(f"  Family booking rate    : "
      f"{float(fam_rows['booking_bool'].mean()):.4f}")
print(f"  Non-family booking rate: "
      f"{float(nfam_rows['booking_bool'].mean()):.4f}")

# =========================================================
# PREPARE FEATURES  (done once, reused for both models)
# =========================================================

feature_cols = get_feature_cols(train_split)
print(f"\nFeature count: {len(feature_cols)}")

X_train = to_pandas_with_cats(train_split, feature_cols)
X_val   = to_pandas_with_cats(val_split,   feature_cols)

y_train = train_split["relevance"].to_numpy()
y_val   = val_split["relevance"].to_numpy()

train_group = make_group(train_split)
val_group   = make_group(val_split)

# Keep val as pandas for fairness evaluation (need srch_id, children)
val_df_pd = val_split.select(
    ["srch_id", "relevance", "srch_children_count"]
).to_pandas()

# =========================================================
# MODEL 1: BASELINE  (no re-weighting)
# =========================================================

print("\n" + "="*60)
print("BASELINE MODEL (no re-weighting)")
print("="*60)

baseline_dataset = lgb.Dataset(
    X_train, label=y_train, group=train_group,
    free_raw_data=False
)

print(f"Training for {BEST_ITERATION} rounds...")
baseline_model = lgb.train(
    BASE_PARAMS,
    baseline_dataset,
    num_boost_round=BEST_ITERATION,
    callbacks=[lgb.log_evaluation(200)],
)

baseline_preds  = baseline_model.predict(X_val)
baseline_result = evaluate_fairness(baseline_preds, val_df_pd)

print("\nBaseline evaluation results:")
for k, v in baseline_result.items():
    if isinstance(v, float):
        print(f"  {k:<25}: {v:.5f}")
    else:
        print(f"  {k:<25}: {v:,}")

# =========================================================
# RE-WEIGHTING
#
# Inverse-frequency weights for two groups:
#
#   w_family    = n_total / (2 × n_family)
#   w_nonfamily = n_total / (2 × n_nonfamily)
#
# The factor of 2 (= number of groups) normalises the weights
# so that their sum equals n_total, keeping gradient magnitudes
# comparable to the unweighted case.
#
# Effect: each family trip row contributes more to the lambda
# gradients during training, forcing the model to pay more
# attention to errors on the underrepresented group.
# =========================================================

# =========================================================
# RE-WEIGHTING
#
# The bias detection showed the model performs BETTER on
# family trips (NDCG@5 = 0.433) than non-family trips
# (0.418). The underperforming group is non-family.
#
# Standard inverse-frequency weights give the LARGER weight
# to the SMALLER group — which would upweight families and
# make things worse. Instead we deliberately invert the
# assignment: non-family rows get the larger weight so the
# model pays more attention to its errors there.
#
# Concretely we swap: non-family gets w = n_total / (2 * n_nonfamily)
# which is SMALLER than w_family = n_total / (2 * n_family)
# in the standard formula, so we flip the assignment:
#
#   weight[nonfamily] = n_total / (2 * n_family)      ← larger weight
#   weight[family]    = n_total / (2 * n_nonfamily)   ← smaller weight
#
# The total weight sum still equals n_total, keeping
# gradient magnitudes comparable to the unweighted case.
# =========================================================

# Standard inverse-frequency weights
w_inv_family    = n_total / (2 * n_family)      # larger  (minority group)
w_inv_nonfamily = n_total / (2 * n_nonfamily)   # smaller (majority group)

# Assign the LARGER weight to non-family (underperforming group)
# and the SMALLER weight to family (overperforming group)
w_for_nonfamily = w_inv_family     # larger
w_for_family    = w_inv_nonfamily  # smaller

children_train = train_split["srch_children_count"].to_numpy()
sample_weights = np.where(
    children_train > 0,
    w_for_family,      # family rows get the smaller weight
    w_for_nonfamily,   # non-family rows get the larger weight
)

print(f"\nRe-weighting (upweighting non-family / underperforming group):")
print(f"  w_nonfamily = {w_for_nonfamily:.4f}  (larger  — underperforming group)")
print(f"  w_family    = {w_for_family:.4f}  (smaller — overperforming group)")
print(f"  Total weight sum: {sample_weights.sum():.0f}  "
      f"(expected: {n_total:,})")

# =========================================================
# MODEL 2: RE-WEIGHTED
# =========================================================

print("\n" + "="*60)
print("RE-WEIGHTED MODEL")
print("="*60)

weighted_dataset = lgb.Dataset(
    X_train, label=y_train, group=train_group,
    weight=sample_weights,
    free_raw_data=False
)

print(f"Training for {BEST_ITERATION} rounds...")
weighted_model = lgb.train(
    BASE_PARAMS,
    weighted_dataset,
    num_boost_round=BEST_ITERATION,
    callbacks=[lgb.log_evaluation(200)],
)

weighted_preds  = weighted_model.predict(X_val)
weighted_result = evaluate_fairness(weighted_preds, val_df_pd)

print("\nRe-weighted evaluation results:")
for k, v in weighted_result.items():
    if isinstance(v, float):
        print(f"  {k:<25}: {v:.5f}")
    else:
        print(f"  {k:<25}: {v:,}")

# =========================================================
# COMPARISON TABLE
# =========================================================

print("\n" + "="*60)
print("FAIRNESS COMPARISON")
print("="*60)

metrics = [
    "ndcg_overall",
    "ndcg_family",
    "ndcg_nonfamily",
    "gap (nonfam - fam)",
]

print(f"\n{'Metric':<25} {'Baseline':>12} {'Re-weighted':>12} {'Change':>10}")
print("-" * 62)
for m in metrics:
    b = baseline_result[m]
    w = weighted_result[m]
    change = w - b
    sign = "+" if change >= 0 else ""
    print(f"{m:<25} {b:>12.5f} {w:>12.5f} {sign}{change:>9.5f}")

print("\nGroup sizes (validation set):")
print(f"  Family queries    : "
      f"{baseline_result['n_family_queries']:,}  "
      f"({baseline_result['n_family_rows']:,} rows)")
print(f"  Non-family queries: "
      f"{baseline_result['n_nonfamily_queries']:,}  "
      f"({baseline_result['n_nonfamily_rows']:,} rows)")

print("\nInterpretation:")
gap_before = baseline_result["gap (nonfam - fam)"]
gap_after  = weighted_result["gap (nonfam - fam)"]
gap_change = gap_after - gap_before

print(f"  Baseline gap (nonfamily - family): {gap_before:+.5f}")
print(f"  Mitigated gap (nonfamily - family): {gap_after:+.5f}")
print(f"  Gap change: {gap_change:+.5f}")
if gap_after > gap_before:
    print("  → Gap reduced: non-family performance improved relative to family")
else:
    print("  → Gap increased: mitigation did not help non-family performance")

overall_change = weighted_result["ndcg_overall"] - baseline_result["ndcg_overall"]
print(f"  Overall NDCG@5 change: {overall_change:+.5f} "
      f"(fairness-accuracy trade-off)")
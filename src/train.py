# import polars as pl
# import lightgbm as lgb
# import numpy as np
# import pandas as pd
# import optuna

# from pathlib import Path
# from sklearn.metrics import ndcg_score
# from optuna.integration import LightGBMPruningCallback

# optuna.logging.set_verbosity(optuna.logging.WARNING)

# # =========================================================
# # PATHS
# # =========================================================

# BASE_DIR = Path(__file__).resolve().parent.parent

# TRAIN_PATH      = BASE_DIR / "data" / "featured" / "train_features.parquet"
# VAL_PATH        = BASE_DIR / "data" / "featured" / "val_features.parquet"
# TEST_PATH       = BASE_DIR / "data" / "featured" / "test_features.parquet"
# SUBMISSION_PATH = BASE_DIR / "submissions" / "lambdamart_submission.csv"
# SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

# # =========================================================
# # CONFIG
# # Set to True if feature_engineering.py was run with
# # INCLUDE_CLICK_BOOKING_FEATURES = True
# # =========================================================
# USE_SMOOTHED_RATES = False

# # =========================================================
# # LOAD DATA
# # =========================================================

# print("Loading data...")
# train_split = pl.read_parquet(TRAIN_PATH)
# val_split   = pl.read_parquet(VAL_PATH)
# test_df     = pl.read_parquet(TEST_PATH)

# # =========================================================
# # COLUMNS
# # =========================================================

# # Never used as model features
# ALWAYS_DROP = [
#     "click_bool", "booking_bool", "gross_bookings_usd",
#     "relevance", "srch_id", "prop_id",
#     "position", "date_time", "checkin_datetime",
#     # Raw counts — only used to compute smoothed rates
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
# # SMOOTHING (only used when USE_SMOOTHED_RATES=True)
# # =========================================================

# def apply_smoothing(df, m_book, m_click, global_booking_rate, global_ctr):
#     return df.with_columns([
#         (
#             (pl.col("prop_booking_count").fill_null(0) + m_book * global_booking_rate) /
#             (pl.col("prop_impressions").fill_null(0)   + m_book)
#         ).alias("prop_booking_rate_smoothed"),
#         (
#             (pl.col("prop_click_count").fill_null(0) + m_click * global_ctr) /
#             (pl.col("prop_impressions").fill_null(0)  + m_click)
#         ).alias("prop_ctr_smoothed"),
#     ])

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

# def compute_ndcg_per_query(preds, labels, srch_ids):
#     df = pd.DataFrame({"srch_id": srch_ids, "y": labels, "p": preds})
#     return float(np.mean([
#         ndcg_score([g["y"].values], [g["p"].values], k=5)
#         for _, g in df.groupby("srch_id")
#     ]))

# # =========================================================
# # GLOBAL RATES
# # =========================================================

# GLOBAL_BOOKING_RATE = float(train_split["booking_bool"].mean())
# GLOBAL_CTR          = float(train_split["click_bool"].mean())
# print(f"Global booking rate: {GLOBAL_BOOKING_RATE:.4f}  |  Global CTR: {GLOBAL_CTR:.4f}")

# # =========================================================
# # FIXED DATA PREP (outside Optuna — only smoothing changes)
# # =========================================================

# y_train = train_split["relevance"].to_numpy()
# y_val   = val_split["relevance"].to_numpy()

# train_group = make_group(train_split)
# val_group   = make_group(val_split)

# # =========================================================
# # OPTUNA OBJECTIVE
# # =========================================================

# def objective(trial):

#     # --- Smoothing (only when click/booking counts are present) ---
#     if USE_SMOOTHED_RATES:
#         m_book  = trial.suggest_float("m_book",  10.0, 1000.0, log=True)
#         m_click = trial.suggest_float("m_click", 10.0, 1000.0, log=True)
#         tr = apply_smoothing(train_split, m_book, m_click, GLOBAL_BOOKING_RATE, GLOBAL_CTR)
#         va = apply_smoothing(val_split,   m_book, m_click, GLOBAL_BOOKING_RATE, GLOBAL_CTR)
#     else:
#         tr = train_split
#         va = val_split

#     feature_cols = get_feature_cols(tr)
#     X_train = to_pandas_with_cats(tr, feature_cols)
#     X_val   = to_pandas_with_cats(va, feature_cols)

#     # --- LightGBM params ---
#     params = {
#         "objective":   "lambdarank",
#         "metric":      "ndcg",
#         "ndcg_eval_at": [5],
#         "label_gain":  [0, 1, 0, 0, 0, 5],
#         "lambdarank_truncation_level": 5,
#         "verbosity":   -1,
#         "seed":        42,

#         # Tree complexity — keep shallow to control overfitting
#         "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
#         "max_depth":        trial.suggest_int("max_depth", 3, 7),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 3000, log=True),

#         # Column / row sampling — primary regularisation levers
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.8),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.9),
#         "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),

#         # L1 / L2
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 50.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 50.0, log=True),

#         # Learning rate — lower is safer, early stopping handles rounds
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

#         # Extra-trees mode: use random splits instead of best splits.
#         # Much stronger regularisation at the cost of some accuracy.
#         # Very useful when you see large train/val gaps.
#         "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
#     }

#     train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
#     val_data   = lgb.Dataset(X_val,   label=y_val,   group=val_group, reference=train_data)

#     model = lgb.train(
#         params,
#         train_data,
#         valid_sets=[train_data, val_data],
#         valid_names=["train", "val"],
#         num_boost_round=1000,
#         callbacks=[
#             lgb.early_stopping(60, verbose=False),
#             lgb.log_evaluation(0),
#             LightGBMPruningCallback(trial, "ndcg@5", valid_name="val"),
#         ],
#     )

#     train_ndcg = model.best_score["train"]["ndcg@5"]
#     val_ndcg   = model.best_score["val"]["ndcg@5"]
#     gap        = train_ndcg - val_ndcg

#     trial.set_user_attr("best_iteration", model.best_iteration)
#     trial.set_user_attr("train_ndcg", train_ndcg)
#     trial.set_user_attr("val_ndcg",   val_ndcg)
#     trial.set_user_attr("gap",        gap)

#     print(
#         f"  Trial {trial.number:>3} | "
#         f"iter={model.best_iteration:>4} | "
#         f"train={train_ndcg:.5f} | val={val_ndcg:.5f} | "
#         f"gap={gap:+.5f} | "
#         f"extra_trees={params['extra_trees']}"
#     )

#     return val_ndcg


# # =========================================================
# # RUN OPTUNA
# # =========================================================

# print("\nRunning Optuna...")

# study = optuna.create_study(
#     direction="maximize",
#     pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=60),
# )

# # Seed with a known safe starting point to anchor the search
# study.enqueue_trial({
#     "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 500,
#     "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 5,
#     "lambda_l1": 0.5, "lambda_l2": 0.5, "learning_rate": 0.05,
#     "extra_trees": False,
# })

# study.optimize(objective, n_trials=100, n_jobs=1)

# # --- Post-search diagnostics ---
# print("\n" + "="*70)
# print(f"Best val NDCG@5:  {study.best_value:.5f}")
# print(f"Best params:      {study.best_params}")
# best_iter = study.best_trial.user_attrs["best_iteration"]
# best_gap  = study.best_trial.user_attrs["gap"]
# print(f"Best iteration:   {best_iter}")
# print(f"Train/val gap:    {best_gap:+.5f}")

# # Show top-10 trials sorted by val NDCG with their gap
# print("\nTop 10 trials:")
# rows = []
# for t in study.trials:
#     if t.value is not None:
#         rows.append({
#             "trial": t.number,
#             "val_ndcg": t.value,
#             "train_ndcg": t.user_attrs.get("train_ndcg", float("nan")),
#             "gap": t.user_attrs.get("gap", float("nan")),
#             "iter": t.user_attrs.get("best_iteration", -1),
#         })
# top10 = sorted(rows, key=lambda x: x["val_ndcg"], reverse=True)[:10]
# for r in top10:
#     print(f"  [{r['trial']:>3}] val={r['val_ndcg']:.5f}  train={r['train_ndcg']:.5f}  gap={r['gap']:+.5f}  iter={r['iter']}")

# print("="*70)

# # =========================================================
# # FINAL MODEL — train on full data (train + val combined)
# # =========================================================

# print("\nBuilding final training dataset (train + val)...")

# train_combined = pl.concat([train_split, val_split]).sort("srch_id")

# if USE_SMOOTHED_RATES:
#     # Recompute prop counts from the full combined training data
#     prop_counts_combined = train_combined.group_by("prop_id").agg([
#         pl.col("booking_bool").sum().cast(pl.Int32).alias("prop_booking_count"),
#         pl.col("click_bool").sum().cast(pl.Int32).alias("prop_click_count"),
#         pl.len().cast(pl.Int32).alias("prop_impressions"),
#     ])
#     # Drop old counts and join fresh ones
#     drop_cols = [c for c in ["prop_booking_count", "prop_click_count", "prop_impressions",
#                               "prop_booking_rate_smoothed", "prop_ctr_smoothed"]
#                  if c in train_combined.columns]
#     train_combined = train_combined.drop(drop_cols).join(prop_counts_combined, on="prop_id", how="left")

#     drop_cols_test = [c for c in ["prop_booking_count", "prop_click_count", "prop_impressions",
#                                    "prop_booking_rate_smoothed", "prop_ctr_smoothed"]
#                       if c in test_df.columns]
#     test_final = test_df.drop(drop_cols_test).join(prop_counts_combined, on="prop_id", how="left")

#     global_booking_rate_full = float(train_combined["booking_bool"].mean())
#     global_ctr_full          = float(train_combined["click_bool"].mean())
#     best_m_book  = study.best_params["m_book"]
#     best_m_click = study.best_params["m_click"]

#     train_combined = apply_smoothing(train_combined, best_m_book, best_m_click,
#                                      global_booking_rate_full, global_ctr_full)
#     test_final     = apply_smoothing(test_final, best_m_book, best_m_click,
#                                      global_booking_rate_full, global_ctr_full)
# else:
#     test_final = test_df

# feature_cols_final = get_feature_cols(train_combined)

# X_full  = to_pandas_with_cats(train_combined, feature_cols_final)
# y_full  = train_combined["relevance"].to_numpy()
# full_group = make_group(train_combined)

# # Build final params from best trial (exclude smoothing keys, add fixed settings)
# exclude_keys = {"m_book", "m_click"}
# final_params = {k: v for k, v in study.best_params.items() if k not in exclude_keys}
# final_params.update({
#     "objective":   "lambdarank",
#     "metric":      "ndcg",
#     "ndcg_eval_at": [5],
#     "label_gain":  [0, 1, 0, 0, 0, 5],
#     "lambdarank_truncation_level": 5,
#     "verbosity":   -1,
#     "seed":        42,
# })

# print(f"Training final model for {best_iter} rounds...")
# final_model = lgb.train(final_params, lgb.Dataset(X_full, label=y_full, group=full_group),
#                         num_boost_round=best_iter)

# # =========================================================
# # FEATURE IMPORTANCE
# # =========================================================

# importance = pd.DataFrame({
#     "feature":    final_model.feature_name(),
#     "importance": final_model.feature_importance(importance_type="gain"),
# }).sort_values("importance", ascending=False)

# print(f"\nTop 20 features (gain), {len(importance)} total:")
# print(importance.head(20).to_string(index=False))

# # =========================================================
# # SUBMISSION
# # =========================================================

# test_sorted = test_final.sort("srch_id")
# X_test = to_pandas_with_cats(test_sorted, feature_cols_final)
# test_pred = final_model.predict(X_test)

# submission = (
#     test_sorted.select(["srch_id", "prop_id"])
#     .with_columns(pl.Series("prediction", test_pred))
#     .sort(["srch_id", "prediction"], descending=[False, True])
#     .select(["srch_id", "prop_id"])
# )
# submission.write_csv(SUBMISSION_PATH)
# print(f"\nSubmission saved: {SUBMISSION_PATH}")
# print(f"Rows: {len(submission):,}  |  Unique queries: {submission['srch_id'].n_unique():,}")

import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna

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
#
# NOTE: smoothing is now FIXED in feature_engineering.py
# (OOF encoding), so there are no m_book/m_click params here.
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
    # Raw counts — consumed during OOF encoding in feature_engineering.py
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
# OPTUNA OBJECTIVE
#
# Overfitting fix strategy:
#  1. Tighter num_leaves / max_depth ranges (shallower trees)
#  2. Higher min_data_in_leaf floor (each leaf must have data)
#  3. Aggressive feature/bagging fraction
#  4. Stronger L1/L2 regularisation
#  5. Lower learning rate range
#  6. min_gain_to_split > 0 (prune useless splits)
#  7. path_smooth > 0 (smooths leaf values — strong regulariser)
#  8. We DO NOT tune extra_trees here because OOF features
#     are now much cleaner; extra_trees was previously helping
#     mainly by degrading memorisation of leaky features.
# =========================================================

def objective(trial):

    params = {
        "objective":   "lambdarank",
        "metric":      "ndcg",
        "ndcg_eval_at": [5],
        # Relevance weights: 0→0, 1→1, 5→5 (click=1, book=5)
        "label_gain":  [0, 1, 0, 0, 0, 5],
        "lambdarank_truncation_level": 10,   # consider top-10 for gradient
        "verbosity":   -1,
        "seed":        42,
        "num_threads": 4,

        # -------------------------------------------------------
        # FIX: "Reducing min_data_in_leaf with feature_pre_filter=true"
        #
        # LightGBM pre-filters features based on the FIRST trial's
        # min_data_in_leaf and caches the result on the Dataset object.
        # Later trials with a SMALLER value can't recover those features.
        # Setting False disables caching so min_data_in_leaf can vary
        # freely. Requires free_raw_data=False on the Dataset (set above).
        # -------------------------------------------------------
        "feature_pre_filter": False,

        # ---- Tree complexity (kept shallow) ----
        "num_leaves":       trial.suggest_int("num_leaves", 15, 63),
        "max_depth":        trial.suggest_int("max_depth", 3, 6),
        # High floor: each leaf must represent many searches
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 500, 5000, log=True),

        # ---- Column/row sampling ----
        "feature_fraction":         trial.suggest_float("feature_fraction", 0.3, 0.7),
        "feature_fraction_bynode":  trial.suggest_float("feature_fraction_bynode", 0.4, 0.8),
        "bagging_fraction":         trial.suggest_float("bagging_fraction", 0.4, 0.8),
        "bagging_freq":             trial.suggest_int("bagging_freq", 1, 5),

        # ---- Regularisation ----
        "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 100.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 100.0, log=True),
        # Minimum information gain to make a split
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
        # Smooths leaf output values — strong regulariser for ranking
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 5.0),

        # ---- Learning rate ----
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(80, verbose=False),
            lgb.log_evaluation(0),
            LightGBMPruningCallback(trial, "ndcg@5", valid_name="val"),
        ],
    )

    train_ndcg = model.best_score["train"]["ndcg@5"]
    val_ndcg   = model.best_score["val"]["ndcg@5"]
    gap        = train_ndcg - val_ndcg

    trial.set_user_attr("best_iteration", model.best_iteration)
    trial.set_user_attr("train_ndcg", train_ndcg)
    trial.set_user_attr("val_ndcg",   val_ndcg)
    trial.set_user_attr("gap",        gap)

    print(
        f"  Trial {trial.number:>3} | "
        f"iter={model.best_iteration:>4} | "
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
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=80),
)

# Seed with a known well-regularised starting point
study.enqueue_trial({
    "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 1000,
    "feature_fraction": 0.5, "feature_fraction_bynode": 0.6,
    "bagging_fraction": 0.6, "bagging_freq": 5,
    "lambda_l1": 1.0, "lambda_l2": 1.0,
    "min_gain_to_split": 0.1, "path_smooth": 1.0,
    "learning_rate": 0.05,
})
# Second seed: more aggressive regularisation
study.enqueue_trial({
    "num_leaves": 24, "max_depth": 4, "min_data_in_leaf": 2000,
    "feature_fraction": 0.4, "feature_fraction_bynode": 0.5,
    "bagging_fraction": 0.5, "bagging_freq": 3,
    "lambda_l1": 5.0, "lambda_l2": 5.0,
    "min_gain_to_split": 0.5, "path_smooth": 2.0,
    "learning_rate": 0.03,
})

study.optimize(objective, n_trials=100, n_jobs=1)

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
        })
top10 = sorted(rows, key=lambda x: x["val_ndcg"], reverse=True)[:10]
for r in top10:
    print(
        f"  [{r['trial']:>3}] val={r['val_ndcg']:.5f}  "
        f"train={r['train_ndcg']:.5f}  gap={r['gap']:+.5f}  iter={r['iter']}"
    )
print("="*70)

# =========================================================
# FINAL MODEL — train on full data (train + val combined)
# =========================================================

print("\nBuilding final training dataset (train + val)...")
train_combined = pl.concat([train_split, val_split]).sort("srch_id")

feature_cols_final = get_feature_cols(train_combined)
X_full     = to_pandas_with_cats(train_combined, feature_cols_final)
y_full     = train_combined["relevance"].to_numpy()
full_group = make_group(train_combined)

final_params = dict(study.best_params)
final_params.update({
    "objective":   "lambdarank",
    "metric":      "ndcg",
    "ndcg_eval_at": [5],
    "label_gain":  [0, 1, 0, 0, 0, 5],
    "lambdarank_truncation_level": 10,
    "verbosity":   -1,
    "seed":        42,
    "num_threads": 4,
    "feature_pre_filter": False,
})

print(f"Training final model for {best_iter} rounds...")
final_model = lgb.train(
    final_params,
    lgb.Dataset(X_full, label=y_full, group=full_group),
    num_boost_round=best_iter,
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "feature":    final_model.feature_name(),
    "importance": final_model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)

print(f"\nTop 30 features (gain), {len(importance)} total:")
print(importance.head(30).to_string(index=False))

# =========================================================
# SUBMISSION
# =========================================================

test_sorted = test_df.sort("srch_id")
X_test      = to_pandas_with_cats(test_sorted, feature_cols_final)
test_pred   = final_model.predict(X_test)

submission = (
    test_sorted.select(["srch_id", "prop_id"])
    .with_columns(pl.Series("prediction", test_pred))
    .sort(["srch_id", "prediction"], descending=[False, True])
    .select(["srch_id", "prop_id"])
)
submission.write_csv(SUBMISSION_PATH)
print(f"\nSubmission saved: {SUBMISSION_PATH}")
print(f"Rows: {len(submission):,}  |  Unique queries: {submission['srch_id'].n_unique():,}")
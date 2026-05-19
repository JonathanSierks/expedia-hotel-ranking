#best method so far
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(100)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_FEATURED_DIR  = BASE_DIR / "data" / "featured"
DATA_FEATURED_DIR.mkdir(parents=True, exist_ok=True)

DATA_PROCESSED_TRAIN_PATH = DATA_PROCESSED_DIR / "training_set_VU_DM.parquet"
DATA_PROCESSED_TEST_PATH  = DATA_PROCESSED_DIR / "test_set_VU_DM.parquet"

# =========================================================
# CONFIG
# Set INCLUDE_CLICK_BOOKING_FEATURES = True to include OOF
# smoothed target-encoded booking/click rates per prop_id.
# These are computed with out-of-fold encoding to prevent
# leakage, which was the cause of the large train/val gap.
# =========================================================
INCLUDE_CLICK_BOOKING_FEATURES = True

# Smoothing strength for target encoding (higher = more
# regularisation, pulls estimates toward the global mean)
M_BOOK  = 100   # fixed, no longer a hyperparameter
M_CLICK = 50

# =========================================================
# LOAD
# =========================================================

print("Loading data...")
train_df = pl.read_parquet(DATA_PROCESSED_TRAIN_PATH)
test_df  = pl.read_parquet(DATA_PROCESSED_TEST_PATH)

# =========================================================
# CLIP OUTLIERS  (train quantiles reused for test)
# =========================================================

def clip_outliers(df, price_low=None, price_high=None,
                  book_high=None, stay_high=None, fit=False):
    if fit:
        price_low, price_high = df.select([
            pl.col("price_usd").quantile(0.001).alias("l"),
            pl.col("price_usd").quantile(0.999).alias("h"),
        ]).row(0)
        book_high = df.select(pl.col("srch_booking_window").quantile(0.999)).item()
        stay_high = df.select(pl.col("srch_length_of_stay").quantile(0.999)).item()
    return df.with_columns([
        pl.col("price_usd").clip(price_low, price_high),
        pl.col("srch_booking_window").clip(0, book_high),
        pl.col("srch_length_of_stay").clip(1, stay_high),
    ]), price_low, price_high, book_high, stay_high

train_df, p_lo, p_hi, bk_hi, st_hi = clip_outliers(train_df, fit=True)
test_df, *_ = clip_outliers(test_df, p_lo, p_hi, bk_hi, st_hi)

# =========================================================
# BASE FEATURE ENGINEERING  (no labels used)
# =========================================================

def engineer_features(df):

    df = df.with_columns(pl.col("price_usd").log1p().alias("log_price_usd"))

    # ----------------------------------------------------------
    # Query-level aggregates — gives "within-search context"
    # (Recommended by winners: compare hotels within a query)
    # ----------------------------------------------------------
    query_stats = df.group_by("srch_id").agg([
        pl.col("price_usd").mean().alias("query_price_mean"),
        pl.col("price_usd").std().alias("query_price_std"),
        pl.col("price_usd").min().alias("query_price_min"),
        pl.col("price_usd").max().alias("query_price_max"),
        pl.col("log_price_usd").mean().alias("query_log_price_mean"),
        pl.col("log_price_usd").std().alias("query_log_price_std"),
        pl.col("prop_starrating").mean().alias("query_star_mean"),
        pl.col("prop_review_score").mean().alias("query_review_mean"),
        pl.col("prop_review_score").min().alias("query_review_min"),
        pl.col("prop_review_score").max().alias("query_review_max"),
        pl.col("prop_location_score1").mean().alias("query_location_mean1"),
        pl.col("prop_location_score1").max().alias("query_location_max1"),
        pl.col("prop_location_score2").mean().alias("query_location_mean2"),
        pl.len().alias("query_hotel_count"),
    ])
    df = df.join(query_stats, on="srch_id", how="left")

    # ----------------------------------------------------------
    # Within-query relative features
    # ----------------------------------------------------------
    df = df.with_columns([
        (pl.col("price_usd") - pl.col("query_price_mean")).alias("price_diff_from_query_mean"),
        ((pl.col("price_usd") - pl.col("query_price_mean")) / (pl.col("query_price_std") + 1e-6)).alias("price_zscore"),
        (pl.col("log_price_usd") - pl.col("query_log_price_mean")).alias("log_price_diff_from_mean"),
        ((pl.col("log_price_usd") - pl.col("query_log_price_mean")) / (pl.col("query_log_price_std") + 1e-6)).alias("log_price_zscore"),
        # Percentile ranks within query (listwise features — key insight from 5th place paper)
        (pl.col("price_usd").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("price_pct_rank"),
        (pl.col("prop_starrating").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("star_pct_rank"),
        (pl.col("prop_review_score").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("review_pct_rank"),
        (pl.col("prop_location_score1").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("location_pct_rank1"),
        (pl.col("prop_location_score2").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("location_pct_rank2"),
        (pl.col("price_usd") == pl.col("price_usd").min().over("srch_id")).cast(pl.Int8).alias("cheapest_hotel_flag"),
        (pl.col("prop_starrating") - pl.col("query_star_mean")).alias("star_diff_from_mean"),
        (pl.col("prop_review_score") - pl.col("query_review_mean")).alias("review_diff_from_mean"),
        (pl.col("prop_location_score1") - pl.col("query_location_mean1")).alias("location_diff_from_mean1"),
        (pl.col("prop_location_score2") - pl.col("query_location_mean2")).alias("location_diff_from_mean2"),
    ])

    # ----------------------------------------------------------
    # Composite features from 5th-place paper (Liu et al. 2013)
    # ump, price_diff, starrating_diff, per_fee, total_fee,
    # score1d2, score2ma
    # ----------------------------------------------------------
    df = df.with_columns([
        # ump = exp(prop_log_historical_price) - price_usd
        # measures how much cheaper/pricier the hotel is vs its own history
        (pl.col("prop_log_historical_price").exp() - pl.col("price_usd")).alias("ump"),
        # price_diff = visitor_hist_adr_usd - price_usd
        # measures alignment of current price with visitor's usual spend
        (pl.col("visitor_hist_adr_usd") - pl.col("price_usd")).alias("price_diff"),
        # starrating_diff = visitor_hist_starrating - prop_starrating
        (pl.col("visitor_hist_starrating") - pl.col("prop_starrating")).alias("starrating_diff"),
        # per_fee = total price / number of guests
        (
            pl.col("price_usd") * pl.col("srch_room_count") /
            (pl.col("srch_adults_count") + pl.col("srch_children_count") + 1e-6)
        ).alias("per_fee"),
        # total_fee = total spend for the stay
        (pl.col("price_usd") * pl.col("srch_room_count")).alias("total_fee"),
        # score1d2 = location_score2 / (location_score1 + epsilon)
        (
            (pl.col("prop_location_score2").fill_null(0) + 0.0001) /
            (pl.col("prop_location_score1").fill_null(0) + 0.0001)
        ).alias("score1d2"),
        # score2ma = prop_location_score2 * srch_query_affinity_score
        (
            pl.col("prop_location_score2").fill_null(0) *
            pl.col("srch_query_affinity_score").fill_null(0)
        ).alias("score2ma"),
        # count_window composite (room count × booking window)
        (
            pl.col("srch_room_count") * pl.col("srch_booking_window").max() +
            pl.col("srch_booking_window")
        ).alias("count_window"),
    ])

    # ----------------------------------------------------------
    # Visitor alignment
    # ----------------------------------------------------------
    df = df.with_columns([
        (pl.col("prop_starrating") - pl.col("visitor_hist_starrating")).abs().alias("star_rating_alignment"),
        (pl.col("price_usd") - pl.col("visitor_hist_adr_usd")).abs().alias("price_alignment"),
    ])

    # ----------------------------------------------------------
    # Competitor features
    # ----------------------------------------------------------
    comp_rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    comp_diff_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]
    comp_inv_cols  = [f"comp{i}_inv" for i in range(1, 9)]
    df = df.with_columns([
        sum([(pl.col(c) == -1).cast(pl.Int8).fill_null(0) for c in comp_rate_cols]).alias("num_comp_cheaper"),
        sum([(pl.col(c) ==  1).cast(pl.Int8).fill_null(0) for c in comp_rate_cols]).alias("num_comp_more_expensive"),
        sum([(pl.col(c) ==  1).cast(pl.Int8).fill_null(0) for c in comp_inv_cols ]).alias("competitor_availability_pressure"),
        pl.mean_horizontal(comp_diff_cols).alias("avg_comp_price_diff"),
    ])

    # ----------------------------------------------------------
    # Travel party features
    # ----------------------------------------------------------
    df = df.with_columns([
        (pl.col("srch_children_count") > 0).cast(pl.Int8).alias("family_trip_flag"),
        (pl.col("srch_adults_count") + pl.col("srch_children_count")).alias("group_travel_size"),
        ((pl.col("srch_adults_count") + pl.col("srch_children_count")) / pl.col("srch_room_count")).alias("guests_per_room"),
    ])

    # ----------------------------------------------------------
    # Temporal features
    # ----------------------------------------------------------
    df = df.with_columns([
        pl.col("date_time").dt.month().cast(pl.Int8).alias(""),
        pl.col("date_time").dt.weekday().cast(pl.Int8).alias("search_day_of_week"),
        pl.col("date_time").dt.hour().cast(pl.Int8).alias("search_hour"),
    ])
    df = df.with_columns(
        (pl.col("date_time") + pl.col("srch_booking_window") * pl.duration(days=1)).alias("checkin_datetime")
    )
    df = df.with_columns([
        pl.col("checkin_datetime").dt.month().cast(pl.Int8).alias("checkin_month"),
        pl.col("checkin_datetime").dt.weekday().cast(pl.Int8).alias("checkin_day_of_week"),
    ])

    # ----------------------------------------------------------
    # Promotion interaction
    # ----------------------------------------------------------
    df = df.with_columns([
        (pl.col("promotion_flag") * pl.col("price_diff_from_query_mean")).alias("promotion_price_interaction"),
    ])

    # ----------------------------------------------------------
    # Missingness flags (missing = negative signal per 3rd-place winner)
    # ----------------------------------------------------------
    df = df.with_columns([
        pl.col("visitor_hist_starrating").is_null().cast(pl.Int8).alias("visitor_history_star_missing"),
        pl.col("visitor_hist_adr_usd").is_null().cast(pl.Int8).alias("visitor_history_price_missing"),
        (pl.col("prop_starrating") == 0).cast(pl.Int8).alias("missing_star_rating_flag"),
        (pl.col("prop_review_score") == 0).cast(pl.Int8).alias("review_score_zero_flag"),
        pl.col("prop_review_score").is_null().cast(pl.Int8).alias("review_score_missing_flag"),
        (pl.col("prop_log_historical_price") == 0).cast(pl.Int8).alias("missing_historical_price_flag"),
        pl.col("srch_query_affinity_score").is_null().cast(pl.Int8).alias("affinity_score_missing_flag"),
        pl.col("orig_destination_distance").is_null().cast(pl.Int8).alias("distance_missing_flag"),
        pl.col("prop_location_score2").is_null().cast(pl.Int8).alias("location_score2_missing_flag"),
    ])

    # ----------------------------------------------------------
    # Relevance label (training data only)
    # ----------------------------------------------------------
    if "booking_bool" in df.columns and "click_bool" in df.columns:
        df = df.with_columns(
            (pl.col("booking_bool") * 5 + pl.col("click_bool") * (1 - pl.col("booking_bool"))).alias("relevance")
        )

    return df


# =========================================================
# PROP-LEVEL NUMERIC AGGREGATES  (no labels — leakage-safe)
#
# KEY INSIGHT from 1st-place winner (Jahrer):
# Computing mean/std/median of ALL numeric features per
# prop_id gives the model a "description" of each hotel
# across all its appearances. This was the single biggest
# improvement (0.51 → 0.53 NDCG).
#
# These are computed from train+test combined so the model
# sees the same hotel context at test time.
# =========================================================

NUMERIC_COLS_FOR_PROP_AGG = [
    "price_usd",
    "log_price_usd",
    "prop_location_score1",
    "prop_location_score2",
    "prop_review_score",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "ump",
    "per_fee",
    "total_fee",
    "score2ma",
]


def compute_prop_numeric_aggs(df):
    """
    Compute mean, std, median of key numeric features per prop_id.
    Called ONCE on the combined train+test frame so the hotel
    profile is the same for training and inference.
    """
    agg_exprs = []
    for col in NUMERIC_COLS_FOR_PROP_AGG:
        if col in df.columns:
            agg_exprs += [
                pl.col(col).mean().alias(f"prop_{col}_mean"),
                pl.col(col).std().alias(f"prop_{col}_std"),
                pl.col(col).median().alias(f"prop_{col}_median"),
            ]
    return df.group_by("prop_id").agg(agg_exprs)


# =========================================================
# POSITION STATS  (non-random sessions only — leakage-safe)
#
# Expedia's own ranking is a strong prior for what gets
# booked. Average/best position per hotel in non-random
# sessions captures this signal without using click/booking
# labels.
# =========================================================

def compute_prop_position_stats(train_df):
    """Per-prop position stats from non-random sessions only."""
    return (
        train_df
        .filter(pl.col("random_bool") == 0)
        .group_by("prop_id")
        .agg([
            pl.col("position").mean().alias("prop_avg_position"),
            pl.col("position").min().alias("prop_best_position"),
            pl.col("position").std().alias("prop_position_std"),
            pl.len().alias("prop_nonrandom_appearances"),
        ])
    )


# =========================================================
# ESTIMATED POSITION  (from 4th-place Igor et al.)
#
# For each (destination, hotel) pair, compute the hotel's
# average rank position in that destination (non-random
# sessions), then take 1/mean_position.
#
# Why 1/position?
#   Position 1 is best. 1/1=1.0, 1/5=0.2, 1/20=0.05.
#   Converts "lower=better" into "higher=better" so the
#   feature is monotonically aligned with booking likelihood.
#
# Why (destination, prop) instead of just prop?
#   A hotel might rank #1 in Amsterdam but #15 for all of
#   Netherlands searches. The destination context captures
#   how competitive this hotel is in the exact market the
#   user is searching — which is what drives booking.
#
# Why non-random sessions only?
#   When random_bool=1, Expedia shuffled the order randomly.
#   Those positions are noise; we want Expedia's actual
#   algorithmic opinion of the hotel in that destination.
# =========================================================

def compute_estimated_position(train_df):
    """
    Returns DataFrame with (srch_destination_id, prop_id,
    estimated_position) where estimated_position = 1/mean(position)
    from non-random training sessions.
    """
    return (
        train_df
        .filter(pl.col("random_bool") == 0)
        .group_by(["srch_destination_id", "prop_id"])
        .agg(
            pl.col("position").mean().alias("_mean_pos")
        )
        .with_columns(
            (1.0 / pl.col("_mean_pos")).alias("estimated_position")
        )
        .drop("_mean_pos")
    )


# =========================================================
# OUT-OF-FOLD TARGET ENCODING  (prevents train/val leakage)
#
# Root cause of the large train/val gap when using
# click/booking features: the model saw booking rates
# computed from the SAME rows it trained on, so rates were
# memorised rather than generalised.
#
# Fix: use 5-fold OOF encoding.
#   - For each fold, compute rates from the other 4 folds
#     and assign them to this fold.
#   - Val split and test always get rates from the full
#     training set (with smoothing toward the global mean).
#
# This is equivalent to leave-one-out target encoding but
# at query granularity, which is the correct unit of
# independence here.
# =========================================================

def smoothed_rate(count, impressions, global_rate, m):
    """Bayesian smoothing: (count + m*global) / (impressions + m)."""
    return (count + m * global_rate) / (impressions + m)


def compute_oof_target_encoding(train_df, n_folds=5, m_book=100, m_click=50):
    """
    Returns train_df with OOF smoothed booking/click rate columns:
        prop_booking_rate_smoothed
        prop_ctr_smoothed

    Steps:
      1. Split srch_ids into n_folds groups.
      2. For each fold, compute per-prop rates from the *other* folds.
      3. Assign those rates to rows in the *held-out* fold.
      4. Global rates are used for smoothing and for props not seen
         in the training folds.
    """
    global_booking_rate = float(train_df["booking_bool"].mean())
    global_ctr          = float(train_df["click_bool"].mean())

    queries   = train_df["srch_id"].unique().to_numpy()
    np.random.seed(42)
    np.random.shuffle(queries)
    folds     = np.array_split(queries, n_folds)

    # Pre-allocate output arrays (indexed by row position)
    n_rows       = len(train_df)
    book_rates   = np.full(n_rows, global_booking_rate, dtype=np.float32)
    click_rates  = np.full(n_rows, global_ctr,          dtype=np.float32)

    # Build a query→row-index mapping (fast lookup)
    srch_ids_np = train_df["srch_id"].to_numpy()

    for fold_idx, held_out_queries in enumerate(folds):
        held_out_set = set(held_out_queries.tolist())
        train_mask   = ~np.isin(srch_ids_np, list(held_out_set))
        held_mask    = np.isin(srch_ids_np, list(held_out_set))

        # Compute rates from the complementary folds
        oof_train = train_df.filter(train_mask)
        prop_stats = oof_train.group_by("prop_id").agg([
            pl.col("booking_bool").sum().alias("book_cnt"),
            pl.col("click_bool").sum().alias("click_cnt"),
            pl.len().alias("impressions"),
        ])

        prop_ids   = prop_stats["prop_id"].to_numpy()
        book_cnt   = prop_stats["book_cnt"].to_numpy()
        click_cnt  = prop_stats["click_cnt"].to_numpy()
        impr       = prop_stats["impressions"].to_numpy()

        book_rate_map  = dict(zip(
            prop_ids,
            smoothed_rate(book_cnt,  impr, global_booking_rate, m_book)
        ))
        click_rate_map = dict(zip(
            prop_ids,
            smoothed_rate(click_cnt, impr, global_ctr,          m_click)
        ))

        held_prop_ids  = train_df.filter(held_mask)["prop_id"].to_numpy()
        held_row_idx   = np.where(held_mask)[0]

        book_rates[held_row_idx]  = np.array(
            [book_rate_map.get(pid, global_booking_rate) for pid in held_prop_ids],
            dtype=np.float32
        )
        click_rates[held_row_idx] = np.array(
            [click_rate_map.get(pid, global_ctr) for pid in held_prop_ids],
            dtype=np.float32
        )

        print(f"  OOF fold {fold_idx+1}/{n_folds} done "
              f"({held_mask.sum():,} rows encoded)")

    train_df = train_df.with_columns([
        pl.Series("prop_booking_rate_oof", book_rates),
        pl.Series("prop_ctr_oof",          click_rates),
    ])
    return train_df, global_booking_rate, global_ctr


def compute_test_target_encoding(source_df, target_df, m_book, m_click,
                                  global_booking_rate, global_ctr):
    """
    Compute smoothed rates from source_df (full training set)
    and join onto target_df (val or test).
    """
    prop_stats = source_df.group_by("prop_id").agg([
        pl.col("booking_bool").sum().alias("book_cnt"),
        pl.col("click_bool").sum().alias("click_cnt"),
        pl.len().alias("impressions"),
    ]).with_columns([
        (
            (pl.col("book_cnt") + m_book * global_booking_rate) /
            (pl.col("impressions") + m_book)
        ).alias("prop_booking_rate_oof"),
        (
            (pl.col("click_cnt") + m_click * global_ctr) /
            (pl.col("impressions") + m_click)
        ).alias("prop_ctr_oof"),
    ]).select(["prop_id", "prop_booking_rate_oof", "prop_ctr_oof"])

    return target_df.join(prop_stats, on="prop_id", how="left").with_columns([
        pl.col("prop_booking_rate_oof").fill_null(global_booking_rate),
        pl.col("prop_ctr_oof").fill_null(global_ctr),
    ])


# =========================================================
# MAIN PIPELINE
# =========================================================

print("Engineering base features...")
train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

# ----------------------------------------------------------
# Train / val split  (group-safe, srch_id level)
# ----------------------------------------------------------
queries    = train_df["srch_id"].unique().to_numpy()
gkf        = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(queries, groups=queries))

train_split_df = train_df.filter(pl.col("srch_id").is_in(queries[train_idx])).sort("srch_id")
val_split_df   = train_df.filter(pl.col("srch_id").is_in(queries[val_idx])).sort("srch_id")

print(f"Train: {len(train_split_df):,} rows | {train_split_df['srch_id'].n_unique():,} queries")
print(f"Val:   {len(val_split_df):,} rows   | {val_split_df['srch_id'].n_unique():,} queries")

# ----------------------------------------------------------
# Prop-level NUMERIC aggregates — computed on train+test
# combined so the hotel profile is consistent at test time.
# (Winner insight: biggest single improvement for LambdaMART)
# ----------------------------------------------------------
print("Computing prop-level numeric aggregates (train+test combined)...")

# We need to combine train and test to compute a "global" hotel profile.
# test_df doesn't have booking/click cols, but the numeric cols we
# aggregate over are all present in both.
# is this ok to do? no test set leakage?
common_cols = [c for c in train_df.columns if c in test_df.columns]
combined_for_prop_agg = pl.concat([
    train_df.select(common_cols),
    test_df.select(common_cols),
])

prop_numeric_aggs = compute_prop_numeric_aggs(w)

train_split_df = train_split_df.join(prop_numeric_aggs, on="prop_id", how="left")
val_split_df   = val_split_df.join(prop_numeric_aggs,   on="prop_id", how="left")
test_df        = test_df.join(prop_numeric_aggs,         on="prop_id", how="left")

# ----------------------------------------------------------
# Price relative to hotel's own typical price
# (complements the per-prop aggs above)
# ----------------------------------------------------------
def add_price_vs_prop(df):
    return df.with_columns([
        (pl.col("price_usd") - pl.col("prop_price_usd_mean")).alias("price_vs_prop_mean"),
        ((pl.col("price_usd") - pl.col("prop_price_usd_mean")) / (pl.col("prop_price_usd_std") + 1e-6)).alias("price_vs_prop_zscore"),
    ])

train_split_df = add_price_vs_prop(train_split_df)
val_split_df   = add_price_vs_prop(val_split_df)
test_df        = add_price_vs_prop(test_df)

# ----------------------------------------------------------
# Position stats  (leakage-safe: uses Expedia rank, not labels)
# Computed from train_split_df only (no leakage into val)
# ----------------------------------------------------------
print("Computing position stats (non-random sessions)...")
prop_pos_train = compute_prop_position_stats(train_split_df)
prop_pos_full  = compute_prop_position_stats(train_df)

train_split_df = train_split_df.join(prop_pos_train, on="prop_id", how="left")
val_split_df   = val_split_df.join(prop_pos_train,   on="prop_id", how="left")
test_df        = test_df.join(prop_pos_full,          on="prop_id", how="left")


def add_position_rank(df):
    if "prop_avg_position" in df.columns:
        return df.with_columns([
            (pl.col("prop_avg_position").rank("ordinal").over("srch_id") /
             pl.col("query_hotel_count")).alias("prop_avg_position_pct_rank"),
        ])
    return df

train_split_df = add_position_rank(train_split_df)
val_split_df   = add_position_rank(val_split_df)
test_df        = add_position_rank(test_df)

# ----------------------------------------------------------
# Estimated position per (destination, prop) — 4th place idea
# Leakage-safe: uses Expedia's own position, not click/book labels.
# Computed from train_split_df only so val rows never leak.
# ----------------------------------------------------------
print("Computing estimated position (destination × prop)...")
est_pos_train = compute_estimated_position(train_split_df)
est_pos_full  = compute_estimated_position(train_df)

train_split_df = train_split_df.join(
    est_pos_train, on=["srch_destination_id", "prop_id"], how="left"
)
val_split_df = val_split_df.join(
    est_pos_train, on=["srch_destination_id", "prop_id"], how="left"
)
test_df = test_df.join(
    est_pos_full, on=["srch_destination_id", "prop_id"], how="left"
)
# Hotels never seen in (destination, prop) pairs get the global
# median — they're unknown quantities, not necessarily bad.


def fill_est_pos(df):
    med = df["estimated_position"].median()
    return df.with_columns(
        pl.col("estimated_position").fill_null(med if med is not None else 0.05)
    )

train_split_df = fill_est_pos(train_split_df)
val_split_df   = fill_est_pos(val_split_df)
test_df        = fill_est_pos(test_df)

# ----------------------------------------------------------
# OOF Target Encoding  (click/booking rates — leak-free)
# ----------------------------------------------------------
if INCLUDE_CLICK_BOOKING_FEATURES:
    print("\nComputing OOF target encoding for click/booking rates...")
    print("(5-fold, smoothed — eliminates train/val leakage)")

    train_split_df, global_booking_rate, global_ctr = compute_oof_target_encoding(
        train_split_df, n_folds=5, m_book=M_BOOK, m_click=M_CLICK
    )

    # Val and test: rates from the full training split (smoothed)
    val_split_df = compute_test_target_encoding(
        train_split_df, val_split_df,
        M_BOOK, M_CLICK, global_booking_rate, global_ctr
    )
    test_df = compute_test_target_encoding(
        train_df,   # full training data for test
        test_df,
        M_BOOK, M_CLICK, global_booking_rate, global_ctr
    )
    print(f"Global booking rate: {global_booking_rate:.4f}  |  Global CTR: {global_ctr:.4f}")
else:
    print("Skipping click/booking target encoding (INCLUDE_CLICK_BOOKING_FEATURES=False).")

# =========================================================
# MISSING VALUE REPORT
# =========================================================

def missing_values(df, name):
    mc = df.null_count().transpose(include_header=True).rename(
        {"column": "variable", "column_0": "missing count"}
    )
    mf = mc.with_columns((pl.col("missing count") / len(df)).alias("missing fraction"))
    print(f"\nMissing values — {name}:")
    print(mf.sort("missing fraction", descending=True).filter(pl.col("missing fraction") > 0))

missing_values(train_split_df, "train split")
missing_values(val_split_df,   "val split")
missing_values(test_df,        "test")

# =========================================================
# SAVE
# =========================================================

print("\nSaving parquet files...")
train_split_df.write_parquet(DATA_FEATURED_DIR / "train_features.parquet")
val_split_df.write_parquet(DATA_FEATURED_DIR   / "val_features.parquet")
test_df.write_parquet(DATA_FEATURED_DIR        / "test_features.parquet")

print("Done!")
print(f"  train: {len(train_split_df):,} rows, {len(train_split_df.columns)} cols")
print(f"  val:   {len(val_split_df):,} rows, {len(val_split_df.columns)} cols")
print(f"  test:  {len(test_df):,} rows, {len(test_df.columns)} cols")
import polars as pl
from pathlib import Path

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(100)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_FEATURED_DIR = BASE_DIR / "data" / "featured"

DATA_FEATURED_DIR.mkdir(parents=True, exist_ok=True)

DATA_PROCESSED_TRAIN_PATH = DATA_PROCESSED_DIR / "training_set_VU_DM.parquet"
DATA_PROCESSED_TEST_PATH = DATA_PROCESSED_DIR / "test_set_VU_DM.parquet"

# ==========================================================
# Load Data
# ==========================================================

print("Loading Train & Test Parquet data...")
train_df = pl.read_parquet(DATA_PROCESSED_TRAIN_PATH)
test_df = pl.read_parquet(DATA_PROCESSED_TEST_PATH)

# ==========================================================
# Clip outliers
# ==========================================================

def clip_outliers(df):

    # helper quantiles
    price_low, price_high = df.select([
        pl.col("price_usd").quantile(0.001).alias("price_low"),
        pl.col("price_usd").quantile(0.999).alias("price_high")
    ]).row(0)

    book_high = df.select([
        pl.col("srch_booking_window").quantile(0.999).alias("booking_window_high")
    ]).item()

    stay_high = df.select([
        pl.col("srch_length_of_stay").quantile(0.999).alias("los_low")
    ]).item()

    df = df.with_columns([

        # price clipping
        pl.col("price_usd")
        .clip(price_low, price_high)
        .alias("price_usd"),

        # booking window clipping
        pl.col("srch_booking_window")
        .clip(0, book_high)
        .alias("srch_booking_window"),

        # length of stay clipping
        pl.col("srch_length_of_stay")
        .clip(1, stay_high)
        .alias("srch_length_of_stay"),
    ])

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def engineer_features(df):
    # =====================================================
    # Add log price
    # =====================================================
    df = df.with_columns([
        pl.col("price_usd").log1p().alias("log_price_usd")
    ])


    # =====================================================
    # QUERY-LEVEL AGGREGATES
    # =====================================================

    query_stats = df.group_by("srch_id").agg([

        pl.col("price_usd").mean().alias("query_price_mean"),
        pl.col("price_usd").std().alias("query_price_std"),
        pl.col("price_usd").min().alias("query_price_min"),
        pl.col("price_usd").max().alias("query_price_max"),

        pl.col("log_price_usd").mean().alias("query_log_price_mean"),
        pl.col("log_price_usd").std().alias("query_log_price_std"),
        pl.col("log_price_usd").min().alias("query_log_price_min"),
        pl.col("log_price_usd").max().alias("query_log_price_max"),

        pl.col("prop_starrating")
        # .filter(pl.col("prop_starrating") > 0)
        .mean()
        .alias("query_star_mean"),

        pl.col("prop_review_score")
        # .filter(pl.col("prop_review_score") > 0)
        .mean()
        .alias("query_review_mean"),

        pl.col("prop_review_score").min().alias("query_review_min"),
        pl.col("prop_review_score").max().alias("query_review_max"),

        pl.col("prop_location_score1").mean().alias("query_location_mean1"),

        pl.col("prop_location_score1").max().alias("query_location_max1"),

        pl.col("prop_location_score2").mean().alias("query_location_mean2"),

        pl.col("prop_location_score1").max().alias("query_location_max1"),

        pl.len().alias("query_hotel_count")
    ])

    df = df.join(query_stats, on="srch_id", how="left")

    # =====================================================
    # PRICE FEATURES
    # =====================================================

    df = df.with_columns([

        (
            pl.col("price_usd") - pl.col("query_price_mean")
        ).alias("price_diff_from_mean"),

        (
            (pl.col("price_usd") - pl.col("query_price_mean")) / (pl.col("query_price_std") + 1e-6)
        ).alias("price_zscore"),

        (
            pl.col("log_price_usd") - pl.col("query_log_price_mean")
        ).alias("log_price_diff_from_mean"),

        (
            (pl.col("log_price_usd") - pl.col("query_log_price_mean")) / (pl.col("query_log_price_std") + 1e-6)
        ).alias("log_price_zscore"),

        (
            pl.col("price_usd").rank("ordinal").over("srch_id")
        ).alias("price_rank"),


        (
            pl.col("price_usd").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")
        ).alias("price_pct_rank"),

        (
            pl.col("price_usd") == pl.col("price_usd").min().over("srch_id")
        ).cast(pl.Int8).alias("cheapest_hotel_flag"),
    ])

    # =====================================================
    # STAR FEATURES
    # =====================================================

    df = df.with_columns([

        (
            pl.col("prop_starrating") - pl.col("query_star_mean")
        ).alias("star_diff_from_mean"),

        (
            pl.col("prop_starrating").rank("ordinal").over("srch_id")
        ).alias("star_rank"),

        (
            pl.col("prop_starrating").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")
        ).alias("star_pct_rank"),
    ])

    # =====================================================
    # REVIEW FEATURES
    # =====================================================

    df = df.with_columns([

        (
            pl.col("prop_review_score") - pl.col("query_review_mean")
        ).alias("review_diff_from_mean"),

        (
            pl.col("prop_review_score").rank("ordinal").over("srch_id")
        ).alias("review_rank"),

        (
            pl.col("prop_review_score").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")
        ).alias("review_pct_rank"),
    ])

    # =====================================================
    # LOCATION FEATURES
    # =====================================================

    df = df.with_columns([

        (
            pl.col("prop_location_score1") - pl.col("query_location_mean1")
        ).alias("location_diff_from_mean1"),

        (
            pl.col("prop_location_score1").rank("ordinal").over("srch_id")
        ).alias("location_rank1"),

        (
            pl.col("prop_location_score2") - pl.col("query_location_mean2")
        ).alias("location_diff_from_mean2"),

        (
            pl.col("prop_location_score2").rank("ordinal").over("srch_id")
        ).alias("location_rank2"),
    ])

    # =====================================================
    # VISITOR ALIGNMENT FEATURES
    # =====================================================

    df = df.with_columns([

        (
            pl.col("prop_starrating") - pl.col("visitor_hist_starrating")
        ).abs().alias("star_rating_alignment"),

        (
            pl.col("price_usd") - pl.col("visitor_hist_adr_usd")
        ).abs().alias("price_alignment"),
    ])

    # =====================================================
    # COMPETITOR FEATURES
    # =====================================================

    comp_rate_cols = [f"comp{i}_rate" for i in range(1, 9)]

    comp_diff_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]

    comp_inv_cols = [f"comp{i}_inv" for i in range(1, 9)]

    df = df.with_columns([

        sum([(pl.col(c) == -1).cast(pl.Int8).fill_null(0) for c in comp_rate_cols]).alias("num_comp_cheaper"),

        sum([(pl.col(c) == 1).cast(pl.Int8).fill_null(0) for c in comp_rate_cols]).alias("num_comp_more_expensive"),

        sum([(pl.col(c) == 1).cast(pl.Int8).fill_null(0) for c in comp_inv_cols]).alias("competitor_availability_pressure"),

        pl.mean_horizontal(comp_diff_cols).alias("avg_comp_price_diff"),
    ])

    # =====================================================
    # TRAVEL FEATURES
    # =====================================================

    df = df.with_columns([

        (
            pl.col("srch_children_count") > 0
        ).cast(pl.Int8).alias("family_trip_flag"),

        (
            pl.col("srch_adults_count") + pl.col("srch_children_count")
        ).alias("group_travel_size"),

        (
            (pl.col("srch_adults_count") + pl.col("srch_children_count")) / pl.col("srch_room_count")
        ).alias("guests_per_room"),
    ])

    # =====================================================
    # TEMPORAL FEATURES
    # =====================================================

    df = df.with_columns([
        pl.col("date_time").dt.year().alias("search_year"),
        pl.col("date_time").dt.month().alias("search_month"),
        pl.col("date_time").dt.weekday().alias("search_day_of_week"),
        pl.col("date_time").dt.hour().alias("search_hour")
    ])

    # =====================================================
    # CHECK-IN DATE FEATURES
    # =====================================================

    df = df.with_columns(
        (pl.col("date_time") + pl.col("srch_booking_window") * pl.duration(days=1)
        ).alias("checkin_datetime")
    )

    df = df.with_columns([
        pl.col("checkin_datetime").dt.year().alias("checkin_year"),
        pl.col("checkin_datetime").dt.month().alias("checkin_month"),
        pl.col("checkin_datetime").dt.weekday().alias("checkin_day_of_week")
    ])

    # =====================================================
    # PRICE VS HISTORICAL HOTEL PRICE
    # =====================================================

    df = df.with_columns([
        (pl.col("price_usd") - pl.col("prop_log_historical_price").exp()).alias("price_vs_historical")
    ])

    # =====================================================
    # PROMOTION INTERACTION
    # =====================================================

    df = df.with_columns([
        (pl.col("promotion_flag") * pl.col("price_diff_from_mean")).alias("promotion_price_interaction")
    ])

    # =====================================================
    # MISSINGNESS FEATURES
    # =====================================================

    df = df.with_columns([
        (
            pl.col("visitor_hist_starrating").is_null()
        ).cast(pl.Int8).alias("visitor_history_star_rating_missing_flag"),

                (
            pl.col("visitor_hist_adr_usd").is_null()
        ).cast(pl.Int8).alias("visitor_history_mean_price_missing_flag"),

        (
            pl.col("prop_starrating") == 0
        ).cast(pl.Int8).alias("missing_star_rating_flag"),

        (
            pl.col("prop_review_score") == 0
        ).cast(pl.Int8).alias("review_score_zero_flag"),

        (
            pl.col("prop_review_score").is_null()
        ).cast(pl.Int8).alias("review_score_missing_flag"),

        (
            pl.col("prop_log_historical_price") == 0
        ).cast(pl.Int8).alias("missing_historical_price_flag"),

        (
            pl.col("srch_query_affinity_score").is_null()
        ).cast(pl.Int8).alias("affinity_score_missing_flag"),

        (
            pl.col("orig_destination_distance").is_null()
        ).cast(pl.Int8).alias("distance_missing_flag"),

        (
            pl.col("prop_location_score2").is_null()
        ).cast(pl.Int8).alias("location_score2_missing_flag"),

    ])

    return df

# =========================================================
# Clip outliers
# =========================================================

train_df = clip_outliers(train_df)
test_df = clip_outliers(test_df)



# =========================================================
# ENGINEER FEATURES
# =========================================================

print("Engineering train features...")
train_df = engineer_features(train_df)

print("Engineering test features...")
test_df = engineer_features(test_df)


# =========================================================
#  Missing value analysis
#  =========================================================

def missing_values(df, name):
    
    print(f"Missing value analysis for {name}")

    missing_count = df.null_count().transpose(include_header=True).rename({"column": "variable", "column_0": "missing count"})
    missing_fraction = missing_count.with_columns((pl.col("missing count")/len(df)).alias("missing fraction"))
    missing_sorted = missing_fraction.sort("missing fraction", descending = True)

    print("Missing value summary:")
    print(missing_sorted)

missing_values(train_df, "training dataset")
missing_values(test_df, "test dataset")

# =========================================================
# SAVE FEATURE DATASETS
# =========================================================

print("Saving feature parquet files...")

train_df.write_parquet(DATA_FEATURED_DIR / "train_features.parquet")

test_df.write_parquet(DATA_FEATURED_DIR / "test_features.parquet")

print("Feature engineering complete!")
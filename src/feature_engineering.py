"""
feature_engineering.py

In:  train_clean.parquet, test_clean.parquet
Out: train_features.parquet, test_features.parquet

Zwei Stufen:
  1. engineer_features()     — reine Transformationen, kein Split-Wissen nötig
  2. add_prop_aggregates()   — prop-level Statistiken aus train+test KOMBINIERT
                               (kein Leakage: nur Rohfeatures, keine Labels)

Was NICHT hier passiert:
  - Train/Val Split
  - Target-Encodings (booking_rate etc.) → das ist train_val_split.py
  - Int64-Casting für torchfm → das ist pointwise_fm.py (transform())
"""
import polars as pl
from pathlib import Path


# -----------------------------------------------------------------------
# Stufe 1: Feature Engineering pro Datei (kein Wissen über andere Dateien)
# -----------------------------------------------------------------------
def engineer_features(df: pl.DataFrame) -> pl.DataFrame:

    df = df.with_columns(pl.col("price_usd").log1p().alias("log_price_usd"))

    # ------------------------------------------------------------------
    # Query-level aggregates
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Within-query relative features
    # ------------------------------------------------------------------
    df = df.with_columns([
        (pl.col("price_usd") - pl.col("query_price_mean")).alias("price_diff_from_query_mean"),
        ((pl.col("price_usd") - pl.col("query_price_mean")) / (pl.col("query_price_std") + 1e-6)).alias("price_zscore"),
        (pl.col("log_price_usd") - pl.col("query_log_price_mean")).alias("log_price_diff_from_mean"),
        ((pl.col("log_price_usd") - pl.col("query_log_price_mean")) / (pl.col("query_log_price_std") + 1e-6)).alias("log_price_zscore"),
        # Percentile ranks
        (pl.col("price_usd").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("price_pct_rank"),
        (pl.col("prop_starrating").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("star_pct_rank"),
        (pl.col("prop_review_score").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("review_pct_rank"),
        (pl.col("prop_location_score1").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("location_pct_rank1"),
        (pl.col("prop_location_score2").rank("ordinal").over("srch_id") / pl.col("query_hotel_count")).alias("location_pct_rank2"),
        # Absolute ranks (deine bestehenden Features behalten)
        pl.col("price_usd").rank("min").over("srch_id").alias("price_rank_in_group"),
        pl.col("prop_starrating").rank("min").over("srch_id").alias("star_rank"),
        (pl.col("log_price_usd") - pl.col("prop_log_historical_price")).alias("price_vs_historical_diff"),
        (pl.col("price_usd") == pl.col("price_usd").min().over("srch_id")).cast(pl.Int8).alias("cheapest_hotel_flag"),
        (pl.col("prop_starrating") - pl.col("query_star_mean")).alias("star_diff_from_mean"),
        (pl.col("prop_review_score") - pl.col("query_review_mean")).alias("review_diff_from_mean"),
        (pl.col("prop_location_score1") - pl.col("query_location_mean1")).alias("location_diff_from_mean1"),
        (pl.col("prop_location_score2") - pl.col("query_location_mean2")).alias("location_diff_from_mean2"),
    ])

    # price_diff_rank: Rang der Abweichung vom historischen Preis
    df = df.with_columns(
        pl.col("price_vs_historical_diff").rank("min").over("srch_id").alias("price_diff_rank")
    )

    # ------------------------------------------------------------------
    # Composite features (Liu et al. 2013)
    # score2ma und promotion_price_interaction raus: FM lernt diese
    # Kreuzterme selbst (multiplikative Feature-Interaktionen)
    # ------------------------------------------------------------------
    df = df.with_columns([
        (pl.col("prop_log_historical_price").exp() - pl.col("price_usd")).alias("ump"),
        (pl.col("visitor_hist_adr_usd") - pl.col("price_usd")).alias("price_diff"),
        (pl.col("visitor_hist_starrating") - pl.col("prop_starrating")).alias("starrating_diff"),
        (
            pl.col("price_usd") * pl.col("srch_room_count") /
            (pl.col("srch_adults_count") + pl.col("srch_children_count") + 1e-6)
        ).alias("per_fee"),
        (pl.col("price_usd") * pl.col("srch_room_count")).alias("total_fee"),
        (
            (pl.col("prop_location_score2").fill_null(0) + 0.0001) /
            (pl.col("prop_location_score1").fill_null(0) + 0.0001)
        ).alias("score1d2"),
        # count_window: bugfix — .max() braucht kein .over() da srch_booking_window
        # pro Suche konstant ist; trotzdem explizit als einfaches Feature:
        (pl.col("srch_room_count") * pl.col("srch_booking_window")).alias("room_window"),
    ])

    # ------------------------------------------------------------------
    # Visitor alignment
    # ------------------------------------------------------------------
    df = df.with_columns([
        (pl.col("prop_starrating") - pl.col("visitor_hist_starrating")).abs().alias("star_rating_alignment"),
        (pl.col("price_usd") - pl.col("visitor_hist_adr_usd")).abs().alias("price_alignment"),
    ])

    # ------------------------------------------------------------------
    # Competitor features
    # ------------------------------------------------------------------
    comp_rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    comp_diff_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]
    comp_inv_cols  = [f"comp{i}_inv"  for i in range(1, 9)]

    df = df.with_columns([
        sum([(pl.col(c) == -1).cast(pl.Int8).fill_null(0) for c in comp_rate_cols]).alias("num_comp_cheaper"),
        sum([(pl.col(c) ==  1).cast(pl.Int8).fill_null(0) for c in comp_rate_cols]).alias("num_comp_more_expensive"),
        sum([(pl.col(c) ==  1).cast(pl.Int8).fill_null(0) for c in comp_inv_cols ]).alias("competitor_availability_pressure"),
        pl.mean_horizontal(comp_diff_cols).alias("avg_comp_price_diff"),
    ])
    # NaN-Handling: avg_comp_price_diff ist NaN wenn alle 8 comp_diff fehlen
    df = df.with_columns([
        pl.col("avg_comp_price_diff").is_null().cast(pl.Int8).alias("avg_comp_price_diff_missing"),
        pl.col("avg_comp_price_diff").fill_null(0.0),
    ])

    # ------------------------------------------------------------------
    # Travel party
    # ------------------------------------------------------------------
    df = df.with_columns([
        (pl.col("srch_children_count") > 0).cast(pl.Int8).alias("family_trip_flag"),
        (pl.col("srch_adults_count") + pl.col("srch_children_count")).alias("group_travel_size"),
        ((pl.col("srch_adults_count") + pl.col("srch_children_count")) / pl.col("srch_room_count")).alias("guests_per_room"),
    ])

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------
    df = df.with_columns([
        pl.col("date_time").dt.month().cast(pl.Int8).alias("search_month"),
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
    df = df.drop("checkin_datetime")

    # ------------------------------------------------------------------
    # Missingness flags
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Relevance label (nur im Trainingsset vorhanden)
    # ------------------------------------------------------------------
    if "booking_bool" in df.columns and "click_bool" in df.columns:
        df = df.with_columns(
            (pl.col("booking_bool") * 5 + pl.col("click_bool") * (1 - pl.col("booking_bool"))).alias("relevance")
        )

    return df


# -----------------------------------------------------------------------
# Stufe 2: prop-level Aggregate aus train+test kombiniert
#
# KEIN LEAKAGE: nur Rohfeatures (Preis, Score, etc.), KEINE Labels.
# Deshalb dürfen train und test hier kombiniert werden — das Hotel
# bekommt eine stabilere "Beschreibung" je mehr Zeilen es hat.
# -----------------------------------------------------------------------
PROP_AGG_COLS = [
    "price_usd", "log_price_usd", "prop_review_score",
    "prop_location_score1", "prop_location_score2",
    "prop_log_historical_price", "srch_booking_window",
    "srch_length_of_stay",
]

def add_prop_aggregates(df_train: pl.DataFrame, df_test: pl.DataFrame):
    """
    Berechnet prop-level Statistiken aus train+test kombiniert.
    Gibt (df_train_mit_agg, df_test_mit_agg) zurück.
    """
    # Nur die relevanten Spalten kombinieren (kein booking_bool etc.)
    shared_cols = ["prop_id"] + [c for c in PROP_AGG_COLS if c in df_train.columns]
    combined = pl.concat([df_train.select(shared_cols), df_test.select(shared_cols)])

    agg_exprs = []
    for col in PROP_AGG_COLS:
        if col in combined.columns:
            agg_exprs += [
                pl.col(col).mean().alias(f"prop_{col}_mean"),
                pl.col(col).std().alias(f"prop_{col}_std"),
                pl.col(col).median().alias(f"prop_{col}_median"),
            ]
    agg_exprs.append(pl.len().alias("prop_count"))

    prop_stats = combined.group_by("prop_id").agg(agg_exprs)
    print(f"prop-level Aggregate: {len(prop_stats):,} Hotels, {prop_stats.shape[1]} Spalten")

    df_train = df_train.join(prop_stats, on="prop_id", how="left")
    df_test  = df_test.join(prop_stats,  on="prop_id", how="left")
    return df_train, df_test


# -----------------------------------------------------------------------
# Hauptpfad
# -----------------------------------------------------------------------
def run(train_path: str, test_path: str, out_dir: str = "."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_train = pl.read_parquet(train_path)
    df_test  = pl.read_parquet(test_path)

    print("Feature Engineering Train...")
    df_train = engineer_features(df_train)
    print("Feature Engineering Test...")
    df_test  = engineer_features(df_test)

    print("Prop-level Aggregate (train+test kombiniert)...")
    df_train, df_test = add_prop_aggregates(df_train, df_test)

    df_train.write_parquet(out / "train_features.parquet")
    df_test.write_parquet(out / "test_features.parquet")
    print(f"Gespeichert: train_features {df_train.shape}, test_features {df_test.shape}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train_clean.parquet")
    p.add_argument("--test",  default="test_clean.parquet")
    p.add_argument("--out",   default=".")
    args = p.parse_args()
    run(args.train, args.test, args.out)

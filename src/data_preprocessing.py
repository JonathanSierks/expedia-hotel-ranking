"""
data_preprocessing.py

In:  training_set_VU_DM.csv, test_set_VU_DM.csv
Out: train_clean.parquet, test_clean.parquet

Macht nur: clipping, dtype-parsing, position entfernen.
Kein Feature Engineering hier.
"""
import polars as pl
from pathlib import Path


PRICE_LOWER_QUANTILE = 0.001
PRICE_UPPER_QUANTILE = 0.999


def load_and_clean(path: str, is_train: bool) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True, null_values=["NULL", "null", ""])
    print(f"Geladen: {df.shape[0]:,} Zeilen, {df.shape[1]} Spalten — {'Train' if is_train else 'Test'}")

    # Spalten die Polars manchmal als String einliest erzwingen als Float
    float_cols = [
    "prop_location_score2", "prop_location_score1",
    "prop_review_score", "srch_query_affinity_score",
    "orig_destination_distance", "visitor_hist_starrating",
    "visitor_hist_adr_usd", "prop_log_historical_price",
    # comp-Spalten ebenfalls als String eingelesen
    *[f"comp{i}_rate"              for i in range(1, 9)],
    *[f"comp{i}_inv"               for i in range(1, 9)],
    *[f"comp{i}_rate_percent_diff" for i in range(1, 9)],
]
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False)
        for c in float_cols if c in df.columns
    ])

    # date_time parsen falls nicht automatisch erkannt
    if df["date_time"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("date_time").str.to_datetime(format="%Y-%m-%d %H:%M:%S")
        )

    # position raus: kodiert Expedias eigenen Bias, fehlt im Testset
    if "position" in df.columns:
        df = df.drop("position")

    # gross_bookings_usd raus: nur bei gebuchten Zeilen gefüllt → Label-Leakage
    if "gross_bookings_usd" in df.columns:
        df = df.drop("gross_bookings_usd")

    # Preis-Clipping: Quantile NUR aus Trainingsdaten berechnen
    # (beim Testset werden dieselben Grenzen übergeben)
    return df


def compute_price_bounds(df_train: pl.DataFrame):
    """Quantil-Grenzen NUR aus Trainingsdaten."""
    lower = df_train["price_usd"].quantile(PRICE_LOWER_QUANTILE)
    upper = df_train["price_usd"].quantile(PRICE_UPPER_QUANTILE)
    print(f"Preis-Clipping: [{lower:.2f}, {upper:.2f}]")
    return lower, upper


def apply_clipping(df: pl.DataFrame, lower: float, upper: float) -> pl.DataFrame:
    return df.with_columns(
        pl.col("price_usd").clip(lower, upper)
    )


def run(train_path: str, test_path: str, out_dir: str = "/local/data/ipv577/expedia-hotel-ranking/data"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_train = load_and_clean(train_path, is_train=True)
    df_test  = load_and_clean(test_path,  is_train=False)

    # Clipping-Grenzen nur aus Train lernen
    lower, upper = compute_price_bounds(df_train)
    df_train = apply_clipping(df_train, lower, upper)
    df_test  = apply_clipping(df_test,  lower, upper)

    df_train.write_parquet(out / "train_clean.parquet")
    df_test.write_parquet(out / "test_clean.parquet")
    print(f"Gespeichert: train_clean.parquet ({df_train.shape}), test_clean.parquet ({df_test.shape})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="training_set_VU_DM.csv")
    p.add_argument("--test",  default="test_set_VU_DM.csv")
    p.add_argument("--out",   default=".")
    args = p.parse_args()
    run(args.train, args.test, args.out)

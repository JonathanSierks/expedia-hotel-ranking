"""
train_val_split.py

In:  train_features.parquet
Out: df_train, df_val (in memory) — mit Target-Encodings

Target-Encodings (booking_rate, click_rate pro prop_id etc.)
werden NACH dem Split berechnet, NUR aus df_train.
Dann auf df_train UND df_val gejoined.

Aufruf aus dem Notebook:
    from train_val_split import split_and_encode
    df_train, df_val = split_and_encode("train_features.parquet")
"""
import polars as pl
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def split_and_encode(
    features_path: str,
    val_frac: float = 0.2,
    seed: int = 42,
    min_prop_count: int = 10,    # Hotels mit weniger Zeilen bekommen globalen Mittelwert
) -> tuple[pl.DataFrame, pl.DataFrame]:

    df = pl.read_parquet(features_path)
    print(f"Geladen: {df.shape[0]:,} Zeilen")

    # ------------------------------------------------------------------
    # Split nach srch_id — ganze Suchen bleiben zusammen
    # ------------------------------------------------------------------
    groups = df["srch_id"].to_numpy()
    idx    = np.arange(len(df))

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(idx, groups=groups))

    df_train = df[tr_idx]
    df_val   = df[va_idx]
    print(f"Train: {len(df_train):,} Zeilen, {df_train['srch_id'].n_unique():,} Suchen")
    print(f"Val:   {len(df_val):,} Zeilen,   {df_val['srch_id'].n_unique():,} Suchen")

    # ------------------------------------------------------------------
    # Target-Encodings — NUR aus df_train berechnen
    #
    # prop_id: Buchungs- und Klickrate + Exposure-Count
    # srch_destination_id: Buchungsrate als Popularitätssignal
    # site_id: Buchungsrate (manche Sites konvertieren besser)
    #
    # Hotels/Destinations mit wenig Daten (< min_prop_count) bekommen
    # den globalen Mittelwert — verhindert Overfit auf seltene IDs.
    # ------------------------------------------------------------------
    global_booking_rate = df_train["booking_bool"].mean()
    global_click_rate   = df_train["click_bool"].mean()

    # prop_id
    prop_stats = (
        df_train.group_by("prop_id").agg([
            pl.col("booking_bool").mean().alias("prop_booking_rate"),
            pl.col("click_bool").mean().alias("prop_click_rate"),
            pl.len().alias("prop_label_count"),
        ])
        .with_columns([
            # Smoothing: seltene Hotels → globaler Mittelwert
            pl.when(pl.col("prop_label_count") >= min_prop_count)
              .then(pl.col("prop_booking_rate"))
              .otherwise(global_booking_rate)
              .alias("prop_booking_rate"),
            pl.when(pl.col("prop_label_count") >= min_prop_count)
              .then(pl.col("prop_click_rate"))
              .otherwise(global_click_rate)
              .alias("prop_click_rate"),
        ])
    )

    # srch_destination_id
    dest_stats = (
        df_train.group_by("srch_destination_id").agg([
            pl.col("booking_bool").mean().alias("dest_booking_rate"),
            pl.len().alias("dest_label_count"),
        ])
        .with_columns(
            pl.when(pl.col("dest_label_count") >= min_prop_count)
              .then(pl.col("dest_booking_rate"))
              .otherwise(global_booking_rate)
              .alias("dest_booking_rate")
        )
    )

    # site_id
    site_stats = (
        df_train.group_by("site_id").agg(
            pl.col("booking_bool").mean().alias("site_booking_rate"),
        )
    )

    # Auf beide Sets joinen — Val bekommt dieselben Train-Statistiken
    for df_ref, name in [(df_train, "train"), (df_val, "val")]:
        pass  # Join unten direkt auf den Variablen

    df_train = (df_train
        .join(prop_stats.select(["prop_id", "prop_booking_rate", "prop_click_rate"]),
              on="prop_id", how="left")
        .join(dest_stats.select(["srch_destination_id", "dest_booking_rate"]),
              on="srch_destination_id", how="left")
        .join(site_stats, on="site_id", how="left")
    )
    df_val = (df_val
        .join(prop_stats.select(["prop_id", "prop_booking_rate", "prop_click_rate"]),
              on="prop_id", how="left")
        .join(dest_stats.select(["srch_destination_id", "dest_booking_rate"]),
              on="srch_destination_id", how="left")
        .join(site_stats, on="site_id", how="left")
    )

    # NaN bei unbekannten IDs (kommen in Val aber nicht in Train vor)
    # → globalen Mittelwert als Fallback
    df_train = df_train.with_columns([
        pl.col("prop_booking_rate").fill_null(global_booking_rate),
        pl.col("prop_click_rate").fill_null(global_click_rate),
        pl.col("dest_booking_rate").fill_null(global_booking_rate),
        pl.col("site_booking_rate").fill_null(global_booking_rate),
    ])
    df_val = df_val.with_columns([
        pl.col("prop_booking_rate").fill_null(global_booking_rate),
        pl.col("prop_click_rate").fill_null(global_click_rate),
        pl.col("dest_booking_rate").fill_null(global_booking_rate),
        pl.col("site_booking_rate").fill_null(global_booking_rate),
    ])

    print(f"Target-Encodings hinzugefügt: prop_booking_rate, prop_click_rate, "
          f"dest_booking_rate, site_booking_rate")
    return df_train, df_val


if __name__ == "__main__":
    df_train, df_val = split_and_encode("train_features.parquet")
    print(df_train.select(["prop_id", "prop_booking_rate", "prop_click_rate"]).head(5))

import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# pl.Config.set_tbl_rows(-1)
# pl.Config.set_tbl_cols(-1)
# pl.Config.set_fmt_str_lengths(100)


# ===============================================================
# Load feature engineered data
# ===============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FEATURED_DIR = BASE_DIR / "data" / "featured"

DATA_FEATURED_TRAIN_PATH = DATA_FEATURED_DIR / "train_features.parquet"
DATA_FEATURED_TEST_PATH = DATA_FEATURED_DIR / "test_features.parquet"

print("Loading featured engineerd data...")
train_df = pl.read_parquet(DATA_FEATURED_TRAIN_PATH)
test_df = pl.read_parquet(DATA_FEATURED_TEST_PATH)

print(train_df.describe())
print(test_df.describe())


# ===============================================================
# Query size distribution
# ===============================================================

query_sizes = (
    train_df.group_by("srch_id")
    .agg(pl.len().alias("query_size"))
)

plt.figure()
plt.hist(query_sizes["query_size"], bins=50)
plt.title("Distribution of Query Sizes")
plt.xlabel("Hotels per query")
plt.ylabel("Frequency")
plt.show()

# ===============================================================
# Quantiles for risk distributions
# ===============================================================

def quantiles(df, col):
    return df.select([
        pl.col(col).min().alias("min"),
        pl.col(col).quantile(0.001).alias("q001"),
        pl.col(col).quantile(0.01).alias("q01"),
        pl.col(col).quantile(0.05).alias("q05"),
        pl.col(col).quantile(0.25).alias("q25"),
        pl.col(col).quantile(0.5).alias("median"),
        pl.col(col).quantile(0.75).alias("q75"),
        pl.col(col).quantile(0.95).alias("q95"),
        pl.col(col).quantile(0.99).alias("q99"),
        pl.col(col).quantile(0.999).alias("q999"),
        pl.col(col).max().alias("max")
    ])

print("Train price:")
print(quantiles(train_df, "price_usd"))

print("Test price:")
print(quantiles(test_df, "price_usd"))

print("Train booking window:")
print(quantiles(train_df, "srch_booking_window"))

print("Test booking window:")
print(quantiles(test_df, "srch_booking_window"))

print("Train search lentgh of stay:")
print(quantiles(train_df, "srch_length_of_stay"))

print("Test search length of stay:")
print(quantiles(test_df, "srch_length_of_stay"))

# ===============================================================
# Price distribution
# ===============================================================
plt.figure()
plt.hist(train_df["price_usd"], bins=100)
plt.title("Price Distribution (raw)")
plt.xlabel("price_usd")
plt.ylabel("count")
plt.show()

# log scale
plt.figure()
plt.hist(np.log1p(train_df["price_usd"]), bins=100)
plt.title("Price Distribution (log scale)")
plt.xlabel("log(price_usd + 1)")
plt.ylabel("count")
plt.show()


# booked vs not booked price distribution
booked = train_df.filter(pl.col("booking_bool") == 1)["price_usd"]
not_booked = train_df.filter(pl.col("booking_bool") == 0)["price_usd"]

plt.figure()
plt.hist(booked, bins=80, alpha=0.6, label="Booked")
plt.hist(not_booked, bins=80, alpha=0.6, label="Not booked")
plt.legend()
plt.title("Price: Booked vs Not Booked")
plt.show()

# log
plt.figure()
plt.hist(np.log1p(booked), bins=80, alpha=0.6, label="Booked")
plt.hist(np.log1p(not_booked), bins=80, alpha=0.6, label="Not booked")
plt.legend()
plt.title("Price: Booked vs Not Booked")
plt.show()


# per query
query_price_stats = (
    train_df.group_by("srch_id")
    .agg([
        pl.col("price_usd").mean().alias("query_price_mean"),
        pl.col("price_usd").std().alias("query_price_std"),
    ])
)

plt.figure()
plt.hist(query_price_stats["query_price_mean"], bins=80)
plt.title("Distribution of Query Mean Prices")
plt.xlabel("Mean price per query")
plt.ylabel("Number of queries")
plt.show()

plt.figure()
plt.hist(query_price_stats["query_price_std"].fill_null(0), bins=80)
plt.title("Distribution of Query Price Std")
plt.xlabel("Std of price per query")
plt.ylabel("Number of queries")
plt.show()

plt.figure()
plt.scatter(
    query_price_stats["query_price_mean"],
    query_price_stats["query_price_std"],
    alpha=0.3
)
plt.title("Query Mean Price vs Price Std")
plt.xlabel("Mean price")
plt.ylabel("Price std")
plt.show()


# ===============================================================
# Booking window distribution
# ===============================================================
plt.figure()
plt.hist(train_df["srch_booking_window"], bins=100)
plt.title("Booking Window Distribution")
plt.xlabel("days")
plt.ylabel("count")
plt.show()

# log scale
plt.figure()
plt.hist(np.log1p(train_df["srch_booking_window"]), bins=100)
plt.title("Booking Window Distribution (log scale)")
plt.xlabel("log(Booking Window Distribution + 1)")
plt.ylabel("count")
plt.show()

# ===============================================================
# Search length stay distribution
# ===============================================================
plt.figure()
plt.hist(train_df["srch_length_of_stay"], bins=100)
plt.title("Length of Stay Distribution")
plt.xlabel("days")
plt.ylabel("count")
plt.show()

# log scale
log_vals = train_df.select(pl.col("srch_length_of_stay").log1p()).to_series()
plt.figure()
plt.hist(log_vals, bins=100)
plt.title("Length of Stay Distribution (log scale)")
plt.xlabel("log(Length of Stay Distribution + 1)")
plt.ylabel("count")
plt.show()


# ===============================================================
# Star rating distribution
# ===============================================================

plt.figure()
plt.hist(train_df["prop_starrating"], bins=10)
plt.title("Star Rating Distribution")
plt.xlabel("Stars")
plt.ylabel("count")
plt.show()



# ===============================================================
# Clicking vs Booking rates
# ===============================================================

target_stats = train_df.select([
    pl.col("click_bool").mean().alias("ctr"),
    pl.col("booking_bool").mean().alias("booking_rate"),
    (pl.col("booking_bool") / (pl.col("click_bool") + 1e-6)).mean().alias("booking_given_click")

])
print(target_stats)

click_rate = train_df["click_bool"].mean()
book_rate = train_df["booking_bool"].mean()
conditional_booking_given_click = (train_df["booking_bool"] / (train_df["click_bool"] + 1e-6)).mean()

plt.figure()
plt.bar(["Click", "Booking", "Booking given Click"], [click_rate, book_rate, conditional_booking_given_click])
plt.title("Click vs Booking vs Booking given Click Rate")
plt.ylabel("Rate")
plt.show()



# per query
per_query_targets = (
    train_df.group_by("srch_id")
    .agg([
        pl.col("click_bool").mean().alias("query_ctr"),
        pl.col("booking_bool").mean().alias("query_booking_rate"),
        pl.len().alias("query_size")
    ])
)

print(per_query_targets.select(["query_ctr", "query_booking_rate"]).describe())

plt.figure()
plt.hist(per_query_targets["query_ctr"], bins=50)
plt.title("Distribution of Query CTR")
plt.xlabel("CTR per query")
plt.ylabel("Number of queries")
plt.show()

plt.figure()
plt.hist(per_query_targets["query_booking_rate"], bins=50)
plt.title("Distribution of Query Booking Rate")
plt.xlabel("Booking rate per query")
plt.ylabel("Number of queries")
plt.show()


# query size vs booking rate
agg = (
    per_query_targets
    .group_by("query_size")
    .agg(pl.col("query_booking_rate").mean().alias("avg_booking_rate"))
    .sort("query_size")
)

plt.figure()
plt.plot(agg["query_size"], agg["avg_booking_rate"])
plt.title("Booking Rate vs Query Size")
plt.xlabel("Query size (number of hotels)")
plt.ylabel("Average booking rate")
plt.show()


# ===============================================================
# Price rank vs booking rate (within query)
# ===============================================================
rank_df = train_df.with_columns([
    pl.col("price_usd").rank("ordinal").over("srch_id").alias("price_rank")
])

agg = (
    rank_df.group_by("price_rank")
    .agg(pl.col("booking_bool").mean().alias("booking_rate"))
    .sort("price_rank")
)

plt.figure()
plt.plot(agg["price_rank"], agg["booking_rate"])
plt.title("Booking Rate vs Price Rank (within query)")
plt.xlabel("Price rank (1 = cheapest)")
plt.ylabel("Booking rate")
plt.show()



# ===============================================================
# booking vs non-booking price
# ===============================================================
booked = train_df.filter(pl.col("booking_bool") == 1)["price_usd"]
not_booked = train_df.filter(pl.col("booking_bool") == 0)["price_usd"]

plt.figure()
plt.hist(booked, bins=80, alpha=0.6, label="Booked")
plt.hist(not_booked, bins=80, alpha=0.6, label="Not booked")
plt.legend()
plt.title("Price: Booked vs Not Booked")
plt.show()

plt.figure()
plt.hist(np.log1p(booked), bins=80, alpha=0.6, label="Booked")
plt.hist(np.log1p(not_booked), bins=80, alpha=0.6, label="Not booked")
plt.legend()
plt.title("Log Price: Booked vs Not Booked")
plt.show()



# ===============================================================
# booking vs non-booking star rating
# ===============================================================
booked = train_df.filter(pl.col("booking_bool") == 1)["prop_starrating"]
not_booked = train_df.filter(pl.col("booking_bool") == 0)["prop_starrating"]

plt.figure()
plt.hist(booked, bins=80, alpha=0.6, label="Booked")
plt.hist(not_booked, bins=80, alpha=0.6, label="Not booked")
plt.legend()
plt.title("Property Star Rating: Booked vs Not Booked")
plt.show()


# ===============================================================
# booking vs non-booking review score
# ===============================================================
booked = train_df.filter(pl.col("booking_bool") == 1)["prop_review_score"]
not_booked = train_df.filter(pl.col("booking_bool") == 0)["prop_review_score"]

plt.figure()
plt.hist(booked, bins=80, alpha=0.6, label="Booked")
plt.hist(not_booked, bins=80, alpha=0.6, label="Not booked")
plt.legend()
plt.title("Property Review Score: Booked vs Not Booked")
plt.show()


# ===============================================================
# booking vs non-booking location scores 
# ===============================================================
booked = train_df.filter(pl.col("booking_bool") == 1)["prop_location_score1"]
not_booked = train_df.filter(pl.col("booking_bool") == 0)["prop_location_score1"]

plt.figure()
plt.hist(np.log1p(booked), bins=80, alpha=0.6, label="Booked")
plt.hist(np.log1p(not_booked), bins=80, alpha=0.6, label="Not booked")
plt.legend()
plt.title("Property Location score 1: Booked vs Not Booked")
plt.show()

booked = train_df.filter(pl.col("booking_bool") == 1)["prop_location_score2"]
not_booked = train_df.filter(pl.col("booking_bool") == 0)["prop_location_score2"]

plt.figure()
plt.hist(np.log1p(booked), bins=80, alpha=0.6, label="Booked")
plt.hist(np.log1p(not_booked), bins=80, alpha=0.6, label="Not booked")
plt.legend()
plt.title("Property Location score 2: Booked vs Not Booked")
plt.show()


# ===============================================================
# Train vs Test distirbution shift
# ===============================================================
plt.figure()
plt.hist(np.log1p(train_df["price_usd"]), bins=100, alpha=0.5, label="train")
plt.hist(np.log1p(test_df["price_usd"]), bins=100, alpha=0.5, label="test")
plt.legend()
plt.title("Train vs Test Price Distribution")
plt.show()

plt.figure()
plt.hist(train_df["prop_starrating"], bins=100, alpha=0.5, label="train")
plt.hist(test_df["prop_starrating"], bins=100, alpha=0.5, label="test")
plt.legend()
plt.title("Train vs Test Property Star Rating Distribution")
plt.show()

plt.figure()
plt.hist(train_df["prop_review_score"], bins=100, alpha=0.5, label="train")
plt.hist(test_df["prop_review_score"], bins=100, alpha=0.5, label="test")
plt.legend()
plt.title("Train vs Test Property Review Score Distribution")
plt.show()


plt.figure()
plt.hist(train_df["prop_id"], bins=100, alpha=0.5, label="train")
plt.hist(test_df["prop_id"], bins=100, alpha=0.5, label="test")
plt.legend()
plt.title("Train vs Test Property ID Distribution")
plt.show()


# ===============================================================
# Correlations
# ===============================================================

df = train_df

numeric_cols = [
    col for col, dtype in df.schema.items()
    if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]
]

corr_df = df.select(numeric_cols).to_pandas().corr()

# remove inf/nan
corr_df = corr_df.replace([np.inf, -np.inf], np.nan)

# fill NaN with 0 (safe for clustering)
corr_df = corr_df.fillna(0)

valid_cols = corr_df.columns[corr_df.std() > 0]
corr_df = corr_df.loc[valid_cols, valid_cols]

plt.figure(figsize=(18, 14))

sns.heatmap(
    corr_df,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0
)

plt.title("Full Feature Correlation Heatmap")
plt.show()


plt.figure(figsize=(18, 14))
sns.clustermap(
    corr_df,
    cmap="coolwarm",
    center=0,
    figsize=(18, 18),
    method="average",
    metric="euclidean"
)
plt.title("Full Clustered Feature Correlation Heatmap")
plt.show()



corr = corr_df.copy()

# flatten matrix
corr_pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)

corr_pairs.columns = ["feature_1", "feature_2", "corr"]

top_corr = corr_pairs.reindex(
    corr_pairs["corr"].abs().sort_values(ascending=False).index
).head(30)

print(top_corr)
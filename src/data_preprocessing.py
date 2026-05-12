import polars as pl
from pathlib import Path 

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(100)

# ##################################
#  LOAD DATA 
# ##################################
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
train_csv = DATA_RAW_DIR / "training_set_VU_DM.csv"
test_csv = DATA_RAW_DIR / "test_set_VU_DM.csv"

DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Load raw data into script
print('Loading Raw CSV files...')
train_df = pl.read_csv(train_csv)
test_df = pl.read_csv(test_csv)


# ##################################
#  Dimensions, Schema & Headers
# ##################################

print('train dimensions:')
print(train_df.shape)

print('test dimensions:')
print(test_df.shape)

print('train schema:')
print(train_df.schema)

print('test schema:')
print(test_df.schema)

print('train header:')
print(train_df.head())

print('test header:')
print(test_df.head())


# ##################################
#  Memory size
# ##################################

print('train estimated memory size:')
print(train_df.estimated_size("mb"), "MB")

print('test estimated memory size:')
print(test_df.estimated_size("mb"), "MB")


# ##################################
#  Convert datatypes
# ##################################

float_cols_train = [
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "comp1_rate_percent_diff",
    "comp2_rate_percent_diff",
    "comp3_rate_percent_diff",
    "comp4_rate_percent_diff",
    "comp5_rate_percent_diff",
    "comp6_rate_percent_diff",
    "comp7_rate_percent_diff",
    "comp8_rate_percent_diff",
    "gross_bookings_usd",
]

int_cols_train = [
    "srch_id",
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "prop_id",
    "prop_starrating",
    "srch_destination_id",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "comp1_rate",
    "comp2_rate",
    "comp3_rate",
    "comp4_rate",
    "comp5_rate",
    "comp6_rate",
    "comp7_rate",
    "comp8_rate",
    "comp1_inv",
    "comp2_inv",
    "comp3_inv",
    "comp4_inv",
    "comp5_inv",
    "comp6_inv",
    "comp7_inv",
    "comp8_inv",
    "position"
]

bool_cols_train = [
    "prop_brand_bool",
    "promotion_flag",
    "srch_saturday_night_bool",
    "random_bool",
    "click_bool",
    "booking_bool"
]

float_cols_test = [
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "comp1_rate_percent_diff",
    "comp2_rate_percent_diff",
    "comp3_rate_percent_diff",
    "comp4_rate_percent_diff",
    "comp5_rate_percent_diff",
    "comp6_rate_percent_diff",
    "comp7_rate_percent_diff",
    "comp8_rate_percent_diff",
]

int_cols_test = [
    "srch_id",
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "prop_id",
    "prop_starrating",
    "srch_destination_id",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "comp1_rate",
    "comp2_rate",
    "comp3_rate",
    "comp4_rate",
    "comp5_rate",
    "comp6_rate",
    "comp7_rate",
    "comp8_rate",
    "comp1_inv",
    "comp2_inv",
    "comp3_inv",
    "comp4_inv",
    "comp5_inv",
    "comp6_inv",
    "comp7_inv",
    "comp8_inv",
]

bool_cols_test = [
    "prop_brand_bool",
    "promotion_flag",
    "srch_saturday_night_bool",
    "random_bool",
]

def clean_null_strings(df):
    df = df.with_columns([pl.when((pl.col(col) == "NULL") ).then(None).otherwise(pl.col(col)).alias(col) for col in df.columns if df[col].dtype == pl.Utf8])
    return df #| (pl.col(col) == "")

def smart_cast(df, name):

    dtype_report = {}

    float_cols = globals()[f"float_cols_{name}"]
    int_cols   = globals()[f"int_cols_{name}"]
    bool_cols  = globals()[f"bool_cols_{name}"]

    exprs = []

    # =========================
    # float columns (always Float32)
    # =========================
    for col in float_cols:

        if col not in df.columns:
            print(f"WARNING: missing float column {col}")
            continue

        if not df[col].dtype == pl.Float32:
            intial_dtype = df[col].dtype
            exprs.append(pl.col(col).cast(pl.Float32, strict=False))
        else:
            intial_dtype = pl.Float32
            exprs.append(pl.col(col))

        dtype_report[col] = {"intial_dtype": intial_dtype, "final_dtype": pl.Float32}

    # =========================
    # bool columns (always Int8)
    # =========================
    for col in bool_cols:

        if col not in df.columns:
            print(f"WARNING: missing bool column {col}")
            continue

        if not df[col].dtype == pl.Int8:
            intial_dtype = df[col].dtype
            exprs.append(pl.col(col).cast(pl.Int8, strict=False))
        else:
            intial_dtype = pl.Int8
            exprs.append(pl.col(col))

        dtype_report[col] = {"intial_dtype": intial_dtype, "final_dtype": pl.Int8}

    # =========================
    # int columns (adaptive dtype)
    # =========================
    for col in int_cols:

        if col not in df.columns:
            print(f"WARNING: missing int column {col}")
            continue

        # convert from string first if needed
        if df[col].dtype == pl.Utf8:
            temp_col = pl.col(col).cast(pl.Int64, strict=False)
        else:
            temp_col = pl.col(col)


        stats = df.select([
            temp_col.min().alias("min"),
            temp_col.max().alias("max")
        ]).row(0)

        min_val, max_val = stats
        intial_dtype = df[col].dtype

        # choose best dtype
        if min_val is None or max_val is None:
            print("!!int min or max val is None!!")
            suggested = pl.Int32
        elif min_val >= -128 and max_val <= 127:
            suggested = pl.Int8
        elif min_val >= -32768 and max_val <= 32767:
            suggested = pl.Int16
        elif min_val >= -2147483648 and max_val <= 2147483647:
            suggested = pl.Int32
        else:
            suggested = pl.Int64

        exprs.append(temp_col.cast(suggested, strict=False))

        dtype_report[col] = {
            "intial_dtype": intial_dtype,
            "min": min_val,
            "max": max_val,
            "final_dtype": suggested
        }

    # apply transformations
    df = df.with_columns(exprs)

    return df, dtype_report

train_df = clean_null_strings(train_df)
test_df = clean_null_strings(test_df)

train_df, train_report = smart_cast(train_df, "train")
test_df, test_report = smart_cast(test_df, "test")

# ##################################
#  Missing value analysis
# ##################################

def missing_values(df, name):
    
    print(f"Missing value analysis for {name}")

    missing_count = df.null_count().transpose(include_header=True).rename({"column": "variable", "column_0": "missing count"})
    missing_fraction = missing_count.with_columns((pl.col("missing count")/len(df)).alias("missing fraction"))
    missing_sorted = missing_fraction.sort("missing fraction", descending = True)

    print("Missing value summary:")
    print(missing_sorted)

missing_values(train_df, "training dataset")
missing_values(test_df, "test dataset")

# #################################################
#  Convert datatime
# #################################################

def convert_datetime(df):
    df = df.with_columns(pl.col("date_time").str.strptime(pl.Datetime,format="%Y-%m-%d %H:%M:%S",strict=False))
    return df

train_df = convert_datetime(train_df)
test_df = convert_datetime(test_df)


# #################################################
#  Print reports
# #################################################
def print_dtype_report(dtype_report, name):

    print("\n")
    print("=" * 100)
    print(f"DTYPE REPORT: {name.upper()}")
    print("=" * 100)
    rows = []

    for col, info in dtype_report.items():

        rows.append({
            "column": col,
            "initial_dtype": str(info.get("intial_dtype")),
            "final_dtype": str(info.get("final_dtype")),
            "min": info.get("min"),
            "max": info.get("max"),
        })

    report_df = (
        pl.DataFrame(rows)
        .sort("column")

    )

    print(report_df)
    print("\nSummary:")
    print(f"Total analyzed columns: {len(report_df)}")
    print("\nInitial dtype counts:")
    initial_counts = (
        report_df
        .group_by("initial_dtype")
        .len()
        .sort("len", descending=True)
    )
    print(initial_counts)
    print("\nFinal dtype counts:")
    print(
        report_df
        .group_by("final_dtype")
        .len()
        .sort("len", descending=True)
    )
    print("=" * 100)

# print_dtype_report(train_report, "train")
# print_dtype_report(test_report, "test")

# #################################################
#  Compress & Save data in Parquet
# #################################################

print('Converting data to parquet...')

train_df.write_parquet(DATA_PROCESSED_DIR / "training_set_VU_DM.parquet")
test_df.write_parquet(DATA_PROCESSED_DIR / "test_set_VU_DM.parquet")

print('Completed data loading & intial preprocessing!')
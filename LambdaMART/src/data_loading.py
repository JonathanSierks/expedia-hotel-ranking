from pathlib import Path 
import polars as pl
import kagglehub
import shutil



# ##################################
#  LOAD DATA 
# ##################################

# Set directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Make directories if not present already
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
train_csv = DATA_RAW_DIR / "training_set_VU_DM.csv"
test_csv = DATA_RAW_DIR / "test_set_VU_DM.csv"

# Check if raw csv files already exist in project folder & download if not
if not train_csv.exists() or not test_csv.exists():
    print('Downloading dataset from Kaggle')
    path = kagglehub.competition_download("dmt-2026-2nd-assignment")
    downloaded_path = Path(path)

    for file in downloaded_path.iterdir():
        target = DATA_RAW_DIR / file.name
        shutil.copy(file, target)

    print('Downloading data complete')

else:
    print('Raw CSV data files already exist')

# Load raw data into script
print('Loading Raw CSV files...')
train_df = pl.read_csv(train_csv)
test_df = pl.read_csv(test_csv)

# # ####################################################
# #  Optimize datatypes for efficient storage
# # ####################################################

# train_dtype_maping = {
#     "site_id": pl.Int16,
#     "visitor_location_country_id": pl.Int16,
#     "prop_country_id": pl.Int16,
#     "prop_starrating": pl.Int8,
#     "prop_brand_bool": pl.Int8,
#     "promotion_flag": pl.Int8,
#     "srch_adults_count": pl.Int8,
#     "srch_children_count": pl.Int8,
#     "srch_room_count": pl.Int8,
#     "random_bool": pl.Int8,
#     "click_bool": pl.Int8,
#     "booking_bool": pl.Int8,
# }

# test_dtype_maping = {
#     "site_id": pl.Int16,
#     "visitor_location_country_id": pl.Int16,
#     "prop_country_id": pl.Int16,
#     "prop_starrating": pl.Int8,
#     "prop_brand_bool": pl.Int8,
#     "promotion_flag": pl.Int8,
#     "srch_adults_count": pl.Int8,
#     "srch_children_count": pl.Int8,
#     "srch_room_count": pl.Int8,
#     "random_bool": pl.Int8,
# }

# train_df = train_df.cast(train_dtype_maping, strict=False)
# test_df = test_df.cast(test_dtype_maping, strict=False)

# #################################################
#  Compress & Save data in Parquet
# #################################################

# print('Converting data to parquet...')

# train_df.write_parquet(DATA_PROCESSED_DIR / "training_set_VU_DMT.parquet")
# test_df.write_parquet(DATA_PROCESSED_DIR / "test_set_VU_DMT.parquet")

# print('Completed data loading & intial preprocessing!')
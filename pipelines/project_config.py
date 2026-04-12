import warnings

warnings.filterwarnings("ignore")

DATASET_GOOGLE_DRIVE_FOLDER_ID = "1rGwpgGk-XODILNLoh0tcsvC1MvwVq5ga"
ORIGINAL_DIR = "dataset/original"

TARGET_COL = "TARGET"

# Aggregator pipeline directories and filenames
AGGREGATED_DIR = "dataset/aggregated"
AGGREGATED_TRADITIONAL_DIR = "dataset/aggregated/traditional"
AGGREGATED_COMBINED_DIR = "dataset/aggregated/combined"

AGGREGATED_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}

# Preprocessor pipeline directories and filenames
PREPROCESSED_DIR = "dataset/preprocessed"
PREPROCESSED_TRADITIONAL_DIR = "dataset/preprocessed/traditional"
PREPROCESSED_COMBINED_DIR = "dataset/preprocessed/combined"

PREPROCESSED_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}

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

# Traditional pipeline
PREPROCESSED_TRADITIONAL_PCA_DIR = "dataset/preprocessed/traditional_pca"
PREPROCESSED_TRADITIONAL_NO_PCA_DIR = "dataset/preprocessed/traditional_no_pca"

# Combined pipeline
PREPROCESSED_COMBINED_PCA_DIR = "dataset/preprocessed/combined_pca"
PREPROCESSED_COMBINED_NO_PCA_DIR = "dataset/preprocessed/combined_no_pca"

# Backwards compatibility
PREPROCESSED_TRADITIONAL_DIR = PREPROCESSED_TRADITIONAL_PCA_DIR
PREPROCESSED_COMBINED_DIR = PREPROCESSED_COMBINED_PCA_DIR

PREPROCESSED_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}

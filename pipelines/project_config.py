import warnings

warnings.filterwarnings("ignore")

DATASET_ROOT = "dataset"
DATASET_GOOGLE_DRIVE_FOLDER_ID = "1rGwpgGk-XODILNLoh0tcsvC1MvwVq5ga"
ORIGINAL_DIR = f"{DATASET_ROOT}/original"

TARGET_COL = "TARGET"

AGGREGATED_DIR = f"{DATASET_ROOT}/aggregated"
AGGREGATED_TRADITIONAL_DIR = f"{AGGREGATED_DIR}/traditional"
AGGREGATED_COMBINED_DIR = f"{AGGREGATED_DIR}/combined"

AGGREGATED_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}

PREPROCESSED_DIR = f"{DATASET_ROOT}/preprocessed"

PREPROCESSED_TRADITIONAL_PCA_DIR = f"{PREPROCESSED_DIR}/traditional_pca"
PREPROCESSED_TRADITIONAL_NO_PCA_DIR = f"{PREPROCESSED_DIR}/traditional_no_pca"

PREPROCESSED_COMBINED_PCA_DIR = f"{PREPROCESSED_DIR}/combined_pca"
PREPROCESSED_COMBINED_NO_PCA_DIR = f"{PREPROCESSED_DIR}/combined_no_pca"

PREPROCESSED_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
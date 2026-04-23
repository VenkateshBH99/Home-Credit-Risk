import os
import subprocess
import sys

from . import project_config
from . import project_config


def ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
        return gdown


def run_pipeline(check_existing=True):
    gdown = ensure_gdown()

    os.makedirs(project_config.ORIGINAL_DIR, exist_ok=True)

    folder_url = f"https://drive.google.com/drive/folders/{project_config.DATASET_GOOGLE_DRIVE_FOLDER_ID}"
    print(f"Fetching dataset from: {folder_url}\n")

    gdown.download_folder(
        url=folder_url,
        output=project_config.ORIGINAL_DIR,
        quiet=False,
        use_cookies=False,
        skip_download=check_existing,
    )

    print(f"\nDone. Files saved to: {project_config.ORIGINAL_DIR}/")


if __name__ == "__main__":
    run_pipeline()

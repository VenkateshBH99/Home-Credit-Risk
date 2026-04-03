import os
import subprocess
import sys

FOLDER_ID = "1rGwpgGk-XODILNLoh0tcsvC1MvwVq5ga"
DATASET_DIR = "./dataset"


def ensure_gdown():
    try:
        import gdown

        return gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

        return gdown


def main():
    gdown = ensure_gdown()

    os.makedirs(DATASET_DIR, exist_ok=True)

    folder_url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    print(f"Downloading from: {folder_url}\n")

    gdown.download_folder(
        url=folder_url,
        output=DATASET_DIR,
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )

    print(f"\nDone. Files saved to: {DATASET_DIR}/")


if __name__ == "__main__":
    main()

"""
Downloads the HASH STI Q&A dataset (Harvard Dataverse, doi:10.7910/DVN/5SWM5U, CC BY-NC 4.0)
into data/raw/. The data folder is git-ignored, so the dataset never ends up in the repo.

Usage: python -m src.download_data
"""

import hashlib
import os
import urllib.request

# file id and checksum taken from the dataset's metadata on Dataverse
FILE_ID = 13436948
EXPECTED_MD5 = "feefd3a26a0cdc32c5a6da9bb5fa5350"
DOWNLOAD_URL = f"https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}"

RAW_DIR = os.path.join("data", "raw")
RAW_FILE = os.path.join(RAW_DIR, "hash_sti_qa.xlsx")


def md5_of(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(RAW_FILE) and md5_of(RAW_FILE) == EXPECTED_MD5:
        print(f"Already downloaded: {RAW_FILE}")
        return RAW_FILE

    print(f"Downloading {DOWNLOAD_URL} ...")
    # Dataverse returns 403 for Python's default user agent
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(RAW_FILE, "wb") as out:
        out.write(response.read())

    if md5_of(RAW_FILE) != EXPECTED_MD5:
        raise RuntimeError(
            "Checksum mismatch: the file on Dataverse may have changed. "
            "Check the dataset page before continuing."
        )
    print(f"Saved to {RAW_FILE} (checksum OK)")
    return RAW_FILE


if __name__ == "__main__":
    download()

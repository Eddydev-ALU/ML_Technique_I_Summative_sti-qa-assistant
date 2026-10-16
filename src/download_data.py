"""
download_data.py - Download the HASH STI Q&A dataset from Harvard Dataverse into data/raw/.

Dataset : The HASH Sexually Transmitted Infection Question and Answer Crowdsourced Dataset
          Dataset Club, HASH (Infectious Diseases Institute, Makerere AI Lab and Sunbird AI), 2026
DOI     : https://doi.org/10.7910/DVN/5SWM5U
License : CC BY-NC 4.0 (attribution required, non-commercial use only)

The file is saved to data/raw/, which is git-ignored: we never commit the data.
Run from the project root:  python -m src.download_data
"""

import hashlib
import os
import urllib.request

# Dataverse file id and MD5 checksum, read from the dataset's public metadata:
# https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/5SWM5U
FILE_ID = 13436948
EXPECTED_MD5 = "feefd3a26a0cdc32c5a6da9bb5fa5350"
DOWNLOAD_URL = f"https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}"

RAW_DIR = os.path.join("data", "raw")
RAW_FILE = os.path.join(RAW_DIR, "hash_sti_qa.xlsx")


def md5_of(path):
    """Return the MD5 checksum of a file, used to check the download is complete and unchanged."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Skip the download if we already have the correct file
    if os.path.exists(RAW_FILE) and md5_of(RAW_FILE) == EXPECTED_MD5:
        print(f"Already downloaded: {RAW_FILE}")
        return RAW_FILE

    print(f"Downloading {DOWNLOAD_URL} ...")
    # Dataverse rejects Python's default User-Agent (HTTP 403), so we send a normal one.
    # urllib follows the redirect to the actual file storage automatically.
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

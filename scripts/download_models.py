import os
import urllib.request
import hashlib
import zipfile
import shutil
import sys

URL = "https://github.com/AidenStickney/EarlyPerf/releases/download/models-v1/earlyperf-models-v1.zip"
FILENAME = "earlyperf-models-v1.zip"
EXPECTED_SHA256 = "7dd09a2caccf11ae397f8d972dcc25b38792c15c3ceed28335f29fe6fcdb5d7c"
DEST_DIR = "artifacts"

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_and_extract():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        
    filepath = os.path.join(DEST_DIR, FILENAME)
    
    print(f"Downloading {URL}...")
    try:
        urllib.request.urlretrieve(URL, filepath)
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)
        
    print("Verifying checksum...")
    calculated_sha = calculate_sha256(filepath)
    if calculated_sha != EXPECTED_SHA256:
        print(f"Checksum Mismatch!")
        print(f"Expected: {EXPECTED_SHA256}")
        print(f"Got:      {calculated_sha}")
        sys.exit(1)
    print("Checksum Verified.")
    
    print(f"Extracting to {DEST_DIR}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(DEST_DIR)
        
    print("Models installed successfully to artifacts/models/")
    
    os.remove(filepath)

if __name__ == "__main__":
    download_and_extract()

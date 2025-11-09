import requests
import os
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download WILI-2018 dataset
url = "https://zenodo.org/records/841984/files/wili-2018.zip"
filename = "data/wili-2018.zip"

print("Downloading WILI-2018 dataset...")
download_file(url, filename)
print("Download completed!")
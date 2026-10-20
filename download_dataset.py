"""Download and extract the pet image dataset."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import requests
from tqdm import tqdm


def download_dataset() -> None:
    root = Path(__file__).resolve().parent
    folders = [root / "PetImages" / split / label
               for split in ("train", "val") for label in ("Cat", "Dog")]
    if all(folder.is_dir() and any(folder.glob("*.jpg")) for folder in folders):
        print("Dataset already exists.")
        return
    url = "https://storage.googleapis.com/ai-fundamentals/data/petsimages.zip"
    buffer = BytesIO()
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tqdm(total=int(response.headers.get("content-length", 0)),
                  desc="Downloading", unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=65536):
                buffer.write(chunk)
                bar.update(len(chunk))
    buffer.seek(0)
    with ZipFile(buffer) as archive:
        for name in archive.namelist():
            target = (root / name).resolve()
            if not target.is_relative_to(root / "PetImages"):
                raise ValueError("Archive contains an unexpected path.")
        archive.extractall(root)
    if not all(folder.is_dir() and any(folder.glob("*.jpg")) for folder in folders):
        raise ValueError("The dataset is missing image folders.")
    print("Dataset downloaded and extracted successfully.")


if __name__ == "__main__":
    download_dataset()

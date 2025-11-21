from audioset_download import Downloader
from tqdm import tqdm
from src.config import CLASSES, DOWNLOAD_PATH


def download_audioset(labels, root_path="raw/test"):
    d = Downloader(
        root_path=root_path,
        labels=labels,
        n_jobs=20,
        download_type="balanced_train",
        copy_and_replicate=False,
    )
    d.download_strong(
        root_path=root_path, format="wav", quality=5, download_sets=["train", "eval"]
    )


if __name__ == "__main__":
    for audio_class in tqdm(CLASSES):
        print(f"Downloading {audio_class}")
        download_audioset(
            labels=[audio_class], root_path=DOWNLOAD_PATH + f"/{audio_class}"
        )
        print(f"Downloaded {audio_class}")

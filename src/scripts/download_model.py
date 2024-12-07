from pathlib import Path

import gdown

URLS = {
    "https://drive.google.com/uc?id=1KiznzcX7TzlviiKusinPWw1ixLPf92E0": "saved/model_best.pth",
}


def main():
    path_gzip = Path("saved/").absolute().resolve()
    path_gzip.mkdir(exist_ok=True, parents=True)

    for url, path in URLS.items():
        gdown.download(url, path)


if __name__ == "__main__":
    main()

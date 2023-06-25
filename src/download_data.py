#!/usr/bin/env python

"""
download_data.py: Downloads raw and preprocessed datasets as well models training results.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import gdown

datasets_url = "https://drive.google.com/drive/folders/1jfArLi57zRsHUDp7uCtJtQYirz3F7IgB"
models_url = "https://drive.google.com/drive/folders/1U7b45D3YUwz90nlLAmTFpZmnZ2dgSC4a"

def main():
    print("Downloading datasets.")
    gdown.download_folder(datasets_url, quiet=False)

    print("Downloading models.")
    gdown.download_folder(models_url, quiet=False)

if __name__ == "__main__":
    main()
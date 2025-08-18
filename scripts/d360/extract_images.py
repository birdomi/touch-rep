"""
Sample script to extract D360 tactile images from pickle files
Images are saved in binaries in the same directory structure as the original pickle files.
Useful for image inspection and for lazy loading in training scripts.
"""

import pandas as pd
import os
import argparse


def main(src_path: str):
    datasets = sorted([dataset for dataset in os.listdir(src_path) if dataset[0].isdigit()])
    for dataset in datasets:
        print(f"Dataset {dataset}")
        src_folder = os.path.join(src_path, dataset)
        sessions = sorted([session for session in os.listdir(src_folder) if session[0].isdigit()])

        for session in sessions:
            src_session = os.path.join(src_folder, session, "seqs")

            print(src_session)
            if not os.path.exists(src_session):
                continue

            seqs = sorted([seq for seq in os.listdir(src_session) if seq[0].isdigit()])
            for seq in seqs:
                src_seq = os.path.join(src_session, seq)
                src_file = os.path.join(src_seq, "data.pickle")
                print(src_seq)

                if not os.path.exists(src_file):
                    continue

                print(f"Extract images form {src_file}")
                src_data = pd.read_pickle(src_file)
                for dev, dev_data in src_data.items():
                    src_img = os.path.join(src_seq, dev, "img")
                    if not os.path.exists(src_img):
                        os.makedirs(src_img)

                    imgs = dev_data["image_raw/compressed"]["data"]
                    for n, img in enumerate(imgs):
                        with open(os.path.join(src_img, f"{n}"), "wb") as file:
                            file.write(img)
                del src_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract images from the D360 pickle file")
    parser.add_argument(
        "--data-path",
        type=str,
    )

    args = parser.parse_args()
    main(args.data_path)

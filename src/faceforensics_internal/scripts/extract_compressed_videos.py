"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h or https://github.com/ondyari/FaceForensics

Author: Andreas Roessler
Date: 25.01.2019
"""
import argparse
import multiprocessing as mp
import os
import subprocess
from os.path import join

import cv2
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType

DATASET_PATHS = {
    "original": "original_sequences/youtube",
    "DeepFakeDetection_original": "original_sequences/actors",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "DeepFakeDetection": "manipulated_sequences/DeepFakeDetection",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}


def extract_frames(data_path, output_path, method="cv2"):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    if method == "ffmpeg":
        subprocess.check_output(
            "ffmpeg -i {} {}".format(data_path, join(output_path, "%04d.png")),
            shell=True,
            stderr=subprocess.STDOUT,
        )
    elif method == "cv2":
        reader = cv2.VideoCapture(data_path)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            cv2.imwrite(join(output_path, "{:04d}.png".format(frame_num)), image)
            frame_num += 1
        reader.release()
    else:
        raise Exception("Wrong extract frames method: {}".format(method))


def extract_method_videos(data_path, dataset, compression):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(
        data_path, DATASET_PATHS[dataset], str(compression), DataType.videos.__str__()
    )
    images_path = join(
        data_path,
        DATASET_PATHS[dataset],
        str(compression),
        DataType.full_images.__str__(),
    )

    Parallel(n_jobs=mp.cpu_count())(
        delayed(
            lambda _video: extract_frames(
                join(videos_path, _video), join(images_path, _video.split(".")[0])
            )
        )(video)
        for video in tqdm(os.listdir(videos_path))
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("data_path", type=str, help="Directory containing the videos.")
    p.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=list(DATASET_PATHS.keys()) + ["all"],
        default="all",
    )
    p.add_argument(
        "--compression",
        "-c",
        type=Compression.argparse,
        choices=Compression,
        default=Compression.raw,
    )
    args = p.parse_args()

    if args.dataset == "all":
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))

import random
import subprocess
from pathlib import Path

import click
import cv2
from tqdm import tqdm

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure


def create_random_resolution_video(video_path: Path, output_path: Path):

    video_capture = cv2.VideoCapture(str(video_path))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 1.0 + random.uniform(-0.5, 0.0)
    new_size = (int(width * scale), int(height * scale))

    ffmpeg_command = (
        f"/home/christoph/bin/ffmpeg "
        f"-i {video_path} "
        f"-vf scale={new_size[0]}:{new_size[1]} "
        f"{output_path}"
    )
    subprocess.check_output(ffmpeg_command, shell=True, stderr=subprocess.STDOUT)


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--target_dir_root", required=True, type=click.Path(exists=True))
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.FF_METHODS
)
def create_random_resolution_videos(source_dir_root, target_dir_root, methods):
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=Compression.raw,
        data_types=(DataType.videos,),
    )

    target_dir_data_structure = FaceForensicsDataStructure(
        target_dir_root,
        methods=methods,
        compressions=Compression.random_resolution,
        data_types=(DataType.videos,),
    )

    for source_sub_dir, target_sub_dir in zip(
        source_dir_data_structure.get_subdirs(), target_dir_data_structure.get_subdirs()
    ):

        if not source_sub_dir.exists():
            continue

        target_sub_dir.mkdir(parents=True, exist_ok=True)
        compression = source_sub_dir.parts[-2]
        method = source_sub_dir.parts[-3]
        print(f"Processing {compression}, {method}")

        for video_path in tqdm(sorted(source_sub_dir.iterdir())):

            output_path = target_sub_dir / video_path.name
            if output_path.exists():
                continue

            create_random_resolution_video(video_path, output_path)


if __name__ == "__main__":
    create_random_resolution_videos()

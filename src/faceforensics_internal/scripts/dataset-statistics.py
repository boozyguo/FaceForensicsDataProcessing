import json

import click
import cv2
import pandas as pd

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure


@click.command()
@click.option("--video_root_dir", required=True, type=click.Path(exists=True))
@click.option("--ff_all_root_dir", required=True, type=click.Path(exists=True))
@click.option(
    "--compressions",
    "-c",
    multiple=True,
    default=[Compression.raw, Compression.c23, Compression.c40],
)
@click.option(
    "--methods",
    "-m",
    multiple=True,
    default=FaceForensicsDataStructure.METHODS_WITHOUT_GOOGLE,
)
def main(video_root_dir, ff_all_root_dir, compressions, methods):
    videos_data_structure = FaceForensicsDataStructure(
        video_root_dir,
        methods=methods,
        compressions=compressions,
        data_types=(DataType.videos,),
    )

    face_information_data_structure = FaceForensicsDataStructure(
        ff_all_root_dir,
        methods=methods,
        compressions=compressions,
        data_types=(DataType.face_information,),
    )

    bounding_boxes_data_structure = FaceForensicsDataStructure(
        ff_all_root_dir,
        methods=methods,
        compressions=compressions,
        data_types=(DataType.bounding_boxes,),
    )

    face_images_tracked_data_structure = FaceForensicsDataStructure(
        ff_all_root_dir,
        methods=methods,
        compressions=compressions,
        data_types=(DataType.face_images_tracked,),
    )

    # masks_tracked_data_structure = FaceForensicsDataStructure(
    #     ff_all_root_dir,
    #     methods=methods,
    #     compressions=(Compression.masks,),
    #     data_types=(DataType.bounding_boxes,),
    # )

    videos_frame_counts = {}
    face_information_frame_counts = {}
    bounding_boxes_frame_counts = {}
    face_images_tracked_frame_counts = {}

    for videos_sub_dir in videos_data_structure.get_subdirs():

        method, compression = videos_sub_dir.parts[-3:-1]

        key = f"{method}_{compression}"
        videos_frame_counts[key] = 0

        for video_path in sorted(videos_sub_dir.iterdir()):
            video_cap = cv2.VideoCapture(str(video_path))
            num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            videos_frame_counts[key] += num_frames

    for face_information_sub_dir in face_information_data_structure.get_subdirs():
        method, compression, data_type = face_information_sub_dir.parts[-3:]

        key = f"{method}_{compression}"
        face_information_frame_counts[key] = 0

        for json_path in sorted(face_information_sub_dir.iterdir()):
            with open(str(json_path), "r") as f:
                json_dict = json.load(f)

            face_information_frame_counts[key] += len(json_dict)

    for bounding_boxes_sub_dir in bounding_boxes_data_structure.get_subdirs():
        method, compression, data_type = bounding_boxes_sub_dir.parts[-3:]

        key = f"{method}_{compression}"
        bounding_boxes_frame_counts[key] = 0

        for json_path in sorted(bounding_boxes_sub_dir.iterdir()):
            with open(str(json_path), "r") as f:
                json_dict = json.load(f)

            bounding_boxes_frame_counts[key] += len(json_dict)

    for face_images_tracked_sub_dir in face_images_tracked_data_structure.get_subdirs():
        method, compression, data_type = face_images_tracked_sub_dir.parts[-3:]

        key = f"{method}_{compression}"
        face_images_tracked_frame_counts[key] = 0

        for sub_dir_path in sorted(face_images_tracked_sub_dir.iterdir()):
            face_images_tracked_frame_counts[key] += len(list(sub_dir_path.iterdir()))

    df_videos = pd.DataFrame.from_dict(
        videos_frame_counts, orient="index", columns=["videos"]
    )
    df_face_information = pd.DataFrame.from_dict(
        face_information_frame_counts, orient="index", columns=["face_information"]
    )
    df_bounding_boxes = pd.DataFrame.from_dict(
        bounding_boxes_frame_counts, orient="index", columns=["bounding_boxes"]
    )
    df_face_images_tracked = pd.DataFrame.from_dict(
        face_images_tracked_frame_counts,
        orient="index",
        columns=["face_images_tracked"],
    )

    df_frame_counts = (
        df_videos.join(df_face_information)
        .join(df_bounding_boxes)
        .join(df_face_images_tracked)
    )

    print(df_frame_counts)
    print(df_frame_counts.describe())

    df_frame_counts.to_csv("frame_counts.csv")


if __name__ == "__main__":
    main()

"""
Written at 📅 2024-10-14 21:39:25
https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md
# Annotations
  # keypoint (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as total_frames); V: number of keypoints (25 for NTURGB+D 3D skeleton, 📍 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
"""

import os
import pickle
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np


class DatasetInfo(NamedTuple):
    name: str  # Dataset name
    relative_path: str  # Relative path to the dataset file
    expected_keypoints: int  # Expected number of keypoints
    download_url: str  # URL to download the dataset


# Adjust the base directory dynamically depending on the current environment
current_path = Path(__file__) if "__file__" in globals() else Path.cwd()
path_parts = current_path.parts
repo_index = path_parts.index("repo")
project_root = Path(*path_parts[: repo_index + 2])
dataset_path: Path = project_root / "external/pyskl/data"


# List of datasets with their expected keypoint numbers
datasets: dict[str, DatasetInfo] = {
    "ntu60_hrnet": DatasetInfo(
        name="NTU60 HRNet 2D Skeleton",
        relative_path="nturgbd/ntu60_hrnet.pkl",
        expected_keypoints=17,
        download_url="https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl",
    ),
    "ntu60_3danno": DatasetInfo(
        name="NTU60 3D Annotation",
        relative_path="nturgbd/ntu60_3danno.pkl",
        expected_keypoints=25,
        download_url="https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl",
    ),
    "ntu120_hrnet": DatasetInfo(
        name="NTU120 HRNet 2D Skeleton",
        relative_path="nturgbd/ntu120_hrnet.pkl",
        expected_keypoints=17,
        download_url="https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl",
    ),
    "ntu120_3danno": DatasetInfo(
        name="NTU120 3D Annotation",
        relative_path="nturgbd/ntu120_3danno.pkl",
        expected_keypoints=25,
        download_url="https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl",
    ),
    "gym_hrnet": DatasetInfo(
        name="GYM HRNet 2D Skeleton",
        relative_path="gym/gym_hrnet.pkl",
        expected_keypoints=17,
        download_url="https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl",
    ),
    "ucf101_hrnet": DatasetInfo(
        name="UCF101 HRNet 2D Skeleton",
        relative_path="ucf101/ucf101_hrnet.pkl",
        expected_keypoints=17,
        download_url="https://download.openmmlab.com/mmaction/pyskl/data/ucf101/ucf101_hrnet.pkl",
    ),
    "hmdb51_hrnet": DatasetInfo(
        name="HMDB51 HRNet 2D Skeleton",
        relative_path="hmdb51/hmdb51_hrnet.pkl",
        expected_keypoints=17,
        download_url="https://download.openmmlab.com/mmaction/pyskl/data/hmdb51/hmdb51_hrnet.pkl",
    ),
}


# Function to check if file exists and download if necessary
def download_if_not_exists(dataset_info: DatasetInfo) -> None:
    """
    Downloads the dataset if it does not already exist.

    Args:
        dataset_info (DatasetInfo): Information about the dataset including name, relative path, and download URL.
    """
    destination: Path = dataset_path / dataset_info.relative_path

    # Check if the file already exists
    if not destination.exists():
        print(f"Downloading {dataset_info.name} to {destination}...")
        os.makedirs(destination.parent, exist_ok=True)
        urllib.request.urlretrieve(dataset_info.download_url, destination)
        print(f"Downloaded {dataset_info.name}")
    else:
        print(f"{dataset_info.name} already exists at {destination}.")


# Function to load and check keypoints shape
def check_keypoints_shape(dataset_info: DatasetInfo) -> None:
    """
    Loads the dataset and checks if the keypoints shape matches the expected number.

    Args:
        dataset_info (DatasetInfo): Information about the dataset including name, relative path, and expected keypoints.
    """
    file_path: Path = dataset_path / dataset_info.relative_path

    # Load the dataset
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # Loop through all annotations and check the shape of keypoints
    for i, annotation in enumerate(iterable=data["annotations"]):
        keypoints: np.ndarray = annotation["keypoint"]
        # Check if the keypoints match the expected shape
        if keypoints.shape[2] != dataset_info.expected_keypoints:
            raise AssertionError(
                f"Annotation {i} in {dataset_info.name} has {keypoints.shape[2]} keypoints, expected {dataset_info.expected_keypoints}."
            )

    # Print the result
    print(
        f"All annotations in {dataset_info.name} have {dataset_info.expected_keypoints} keypoints."
    )


# Iterate over specific datasets based on keys (or all if "*")
def process_datasets(
    datasets: dict[str, DatasetInfo], dataset_keys: str | list[str]
) -> None:
    """
    Processes specific datasets by downloading them if necessary and checking keypoints shape.

    Args:
        datasets (dict[str, DatasetInfo]): A dictionary of dataset information.
        dataset_keys (str or list[str]): "*" for all datasets or a list of specific dataset keys to process.
    """
    if dataset_keys == "*":
        selected_datasets = datasets
    else:
        # Filter the selected datasets based on the keys provided
        selected_datasets = {
            key: datasets[key] for key in dataset_keys if key in datasets
        }

    for dataset_name, dataset_info in selected_datasets.items():
        try:
            download_if_not_exists(dataset_info)
            check_keypoints_shape(dataset_info)
        except FileNotFoundError:
            print(
                f"File not found for {dataset_info.name}, please download it manually from {dataset_info.download_url}"
            )


# Example: Process all datasets
process_datasets(datasets=datasets, dataset_keys="*")

# Example: Process only specific datasets
# process_datasets(datasets=datasets, dataset_keys=["ntu60_hrnet", "gym_hrnet"])

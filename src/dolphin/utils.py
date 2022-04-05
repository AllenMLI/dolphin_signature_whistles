"""
Utils file for support generic needs
"""
import os
import random
from pathlib import Path
import cv2


def get_input_shape(preprocessed_data_dir: Path) -> tuple:
    """
    Load in a random image from the preprocessed directory and return the shape.

    Args:
        preprocessed_data_dir (Path): directory where all of the preprocessed images live

    Returns:
        (tuple) input shape of shape (3,)
    """
    for root, _, files in os.walk(preprocessed_data_dir):
        for file in files:
            if file.endswith('.png'):
                im = cv2.imread(os.path.join(root, file))
                return im.shape


def get_savename(preprocessed_dir: Path, fp: Path, split: str, root_idx: int) -> Path:
    """
    This function puts together the savename for the preprocessed features, based on which dataset.
    Very much a hardcoded function, but not sure how to better handle discrepancies between datasets.

    Args:
        preprocessed_dir (Path): the path to the preprocessed features
        fp (Path): the filepath to the current datapoint
        split (str): the train/val/test split for this file
        root_idx (int): index into the root, used in get_savename for hardcoded paths

    Returns:
        savedir (Path): path to the directory where files will be saved
        savename (Path): path where the file is going to be saved (including filename)
    """
    s = fp.split('/')  # split the filepath into parts
    savedir = preprocessed_dir / split / s[root_idx]
    savename = savedir / (s[-1][:-4] + '.png')
    return savedir, savename


def split_files(data_dir: Path, dataset: str, split_ratios: list = [.7, .15, .15]) -> dict:
    """
    Splits files in the given data directory into a dictionary of train/val/test.

    Args:
        data_dir (Path): path to the original data directory provided by the user
        split_ratios (list): list of length 3, with proportions for each data split
        dataset (str): the name of the dataset being used

    Returns:
        (dict): dictionary with train/val/test filepaths
    """
    splits = {'train': [], 'val': [], 'test': []}
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                if ((dataset == 'sarasota') and ("-sw-" in file.lower() or "-swv-" in file.lower())) or (dataset != 'sarasota'):
                    split = random.choices(['train', 'val', 'test'], weights=split_ratios, k=1)[0]
                    splits[split].append(os.path.join(root,file))
    return splits


def grab_splits(preprocessed_dir: Path, data_dir: Path) -> dict:
    """
    Based on the already existing preprocessed directory, extract the train/val/test splits.
    This function should return an output in the same form as the split_files() function.

    Args:
        preprocessed_dir (Path): directory where preprocessed data lives
        data_dir (Path): path to the original data directory provided by the user

    Returns:
        (dict): dictionary with train/val/test filepaths
    """
    splits = {'train': [], 'val': [], 'test': []}
    for root, _, files in os.walk(preprocessed_dir):
        s = root.split('/')  # outputs/{preprocessed_dir}/{feature_type}/{split}/{class_name}
        if len(s) < 5:
            continue

        # Iterate over each preprocessed file and extract the raw wav filepath (exactly like the split_files() func)
        for filename in files:
            split = s[3]  # train, val, or test split
            if split not in ['train', 'val', 'test']:
                continue

            savename = data_dir / s[4] / (filename[:-3] + 'wav')
            splits[split].append(str(savename))
    return splits

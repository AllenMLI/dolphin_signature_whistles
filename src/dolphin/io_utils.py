"""
Utils file for support generic needs
"""
import os
from pathlib import Path
import datetime
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np


def save_history(history_dir:Path, history: dict) -> None:
    """
    Save history to given path.

    Args:
        history (dict): the history object for saving
        history_dir (Path): path where this history should be saved
        datetime_path (str): the datetime path to save history in different path in between runs
    """
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)  # leave existing directory unaltered

    with open(os.path.join(history_dir, str(datetime.datetime.now()) + '.pkl'), 'wb') as f:
        pickle.dump(history, f)


def load_history(history_dir: Path) -> np.ndarray:
    """
    Load history

    Args:
        history_dir (Path): location where the history file is

    Returns:
        out (dict): history dictionary
    """
    with open(history_dir, 'rb') as f:
        out = pickle.load(f)
    return out


def save_fig(image: np.ndarray, f: np.ndarray, t: np.ndarray, output_dir: Path, cfg: dict) -> None:
    """
    Save image using librosa library to the given output directory.

    Args:
        image (np.ndarray): image to be saved
        output_dir (Path): output directory
        cfg (dict): configuration parameters
    """
    fig = plt.figure()
    fig.set_size_inches(3 * cfg['output']['inches_per_sec'],
                        cfg['preprocess']['sampling_rate'] / 1000 / 2 * cfg['output']['inches_per_KHz'])
    plt.pcolormesh(t, f, image, vmin=image.min(), vmax=image.max(), cmap=cfg['output']['color_map'])
    plt.axis('off')
    plt.savefig(output_dir, transparent=False, bbox_inches='tight', format='png')
    plt.close()


def write_to_csv_classifier(csv_file: Path, predicted_labels: dict, predicted_probabilities: dict):
    """
    Writes results to CSV in two formats, Raven selection table and top five predictions per file

    Args:
        csv_path (str): path to where we want to output CSV results
        predictions (dict): keys=spec paths, values=probability outputs
    """
    columns = ['filename', 'label1', 'p1', 'label2', 'p2', 'label3', 'p3', 'label4', 'p4', 'label5', 'p5']
    with open(csv_file, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, delimiter='\t')
        writer.writeheader()

        for fp, labels in predicted_labels.items():
            conf = predicted_probabilities[fp]
            writer.writerow({'filename': fp,
                             'label1': labels[0], 'p1': conf[0],
                             'label2': labels[1], 'p2': conf[1],
                             'label3': labels[2], 'p3': conf[2],
                             'label4': labels[3], 'p4': conf[3],
                             'label5': labels[4], 'p5': conf[4]})

    print(f'Results saved to {csv_file}')


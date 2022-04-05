"""
Preprocess class
"""
import os
from pathlib import Path

import librosa
import soundfile as sf

import dolphin.preprocess.feature_extraction as feature_extraction
import dolphin.io_utils as io_utils
import dolphin.utils as utils


class Preprocessor():
    """Object that preprocesses all of the data - including segmentation, generating spectrograms, and structural organization."""

    def __init__(self, cfg: dict):

        # Preprocessing Arguments
        self.features = cfg['preprocess']['features']  # method of feature extraction; will most often be 'spectrogram'

        self.cfg = cfg

    def run(self, audio_splits: dict, preprocessed_data_dir: Path, root_idx: int) -> Path:
        """
        Runs the data through the necessary preprocessing, based on configuration arguments.

        Args:
            audio_splits (dict): dictionary containing train/val/test filepaths to raw wav files
            preprocessed_data_dir (Path): directory where preprocessed data will be saved
            root_idx (int): index into the root, used in get_savename for hardcoded paths

        Returns:
            (Path): the path to desired features
        """
        # Check to make sure the user has specified a valid feature option
        if not self.features in ['spec', 'melspec']:
            raise ValueError('Invalid argument for features. Must be spec or melspec.')

        # Extract and save features if this has NOT been done before
        preprocessed_dir = preprocessed_data_dir / self.features
        if not os.path.exists(preprocessed_dir):
            self.generate_features(audio_splits, preprocessed_dir, self.features, self.cfg, root_idx)

        return preprocessed_dir

    @staticmethod
    def generate_features(audio_splits: dict, preprocessed_dir: Path, feature_type: str, cfg: dict, root_idx: int):
        """
        Given a directory of raw wav files, generate normal spectrograms using arguments from the experiment config file.
        The raw wav files comes from data_dir and the resulting spectrograms are saved to spec_dir.

        Args:
            audio_splits (dict): dictionary containing train/val/test filepaths to raw wav files
            preprocessed_paths (Path): path to the preprocessed directory we want to save to
            feature_type (str): features to be extracted, in [spec, melspec, pcen]
            cfg (dict): dictionary of configuration arguments
            root_idx (int): index into the root, used in get_savename for hardcoded paths
        """
        random_pad = False

        # Load in all wavs, extract desired features and save to preprocessed/ directory
        for split, fps in audio_splits.items():

            if split == 'train':
                random_pad = True

            for fp in fps:
                spec_max_length = cfg["preprocess"]["spectrogram_max_length"]

                data, sr = None, None
                if fp.lower().endswith(".wav"):
                    data, sr = librosa.load(fp, sr=cfg['preprocess']['sampling_rate'], duration=spec_max_length)
                elif fp.lower().endswith(".flac"):
                    data, _ = sf.read(fp, dtype="float32")
                    sr = cfg['preprocess']['sampling_rate']

                if feature_type == 'spec':
                    feature, f, t = feature_extraction.compute_spectrogram(data, sr=sr, cfg=cfg, random_pad=random_pad)
                elif feature_type == 'melspec':
                    feature = feature_extraction.compute_melspec(data, sr=sr, cfg=cfg)

                savedir, savename = utils.get_savename(preprocessed_dir, fp, split, root_idx)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                io_utils.save_fig(feature, f, t, output_dir=savename, cfg=cfg)

import os
import random
import numpy as np
from pathlib import Path

from dolphin.preprocess.preprocessor import Preprocessor
import dolphin.augment.waveform_augment as waveform_augment
import dolphin.augment.mixture_augment as mixture_augment


def generate_augmentations(output_dir: Path, splits: dict, cfg: dict, preprocessor: Preprocessor, 
                    speed_augment: float, pitch_augment: float, noise_augment: float, mix_augment: float):
    """
    Generates all augmentations that are desired for the current experiment. Then preprocesses those augmented wavs.

    Args:
        output_dir (Path): the directory where all outputs are saved
        splits (dict): the dictionary holding train/val/test filepaths to original wav files
        cfg (dict): dictionary holding all config file arguments
        preprocessor (Preprocessor): the preprocessor object that is responsible for feature extraction
        speed_augment (float): probability that a speed augmentation will be applied
        pitch_augment (float): probability that a pitch augmentation will be applied
        noise_augment (float): probability that a noise augmentation will be applied
        mix_augment (float): probability that a mixture augmentation will be applied
    """
    splits = {'train': splits['train']}

    # Path where all of the following augmented files are saved
    aug_path = output_dir / "augments"
    
    # Speed augmentations
    speed_path = aug_path / "speed_augments"
    if speed_augment > 0.0 and not os.path.exists(speed_path):
        speedup_path = speed_path / "speedup"
        speed_splits = waveform_augment.main(splits, speedup_path, 'speedup', cfg['preprocess']['sampling_rate'])
        preprocessor.run(speed_splits, speedup_path / "preprocessed", 6)

        slowdown_path = speed_path / "slowdown"
        speed_splits = waveform_augment.main(splits, slowdown_path, 'slowdown', cfg['preprocess']['sampling_rate'])
        preprocessor.run(speed_splits, slowdown_path / "preprocessed", 6)

    # Pitch augmentations
    pitch_path = aug_path / "pitch_augments"
    if pitch_augment > 0.0 and not os.path.exists(pitch_path):
        pitchup_path = pitch_path / "shiftpitchup"
        pitch_splits = waveform_augment.main(splits, pitchup_path, 'shiftpitchup', cfg['preprocess']['sampling_rate'])
        preprocessor.run(pitch_splits, pitchup_path / "preprocessed", 6)

        pitchdown_path = pitch_path / "shiftpitchdown"
        pitch_splits = waveform_augment.main(splits, pitchdown_path, 'shiftpitchdown', cfg['preprocess']['sampling_rate'])
        preprocessor.run(pitch_splits, pitchdown_path / "preprocessed", 6)

    # Random noise augmentation
    noise_path = aug_path / "noise_augments"
    if noise_augment > 0.0 and not os.path.exists(noise_path):
        noise_splits = waveform_augment.main(splits, noise_path, 'addrandomnoise', cfg['preprocess']['sampling_rate'])
        preprocessor.run(noise_splits, noise_path / "preprocessed", 5)

    # Generate mixtures of the original dolphin data with background noises contained in the bg_audio_dir directory
    mixture_path = aug_path / "mix_augments"
    if mix_augment and not os.path.exists(mixture_path):
        snrs = [cfg['augment']['min_SNR'], cfg['augment']['max_SNR']]
        mixture_splits = mixture_augment.main(splits, mixture_path, cfg['augment']['bg_audio_dir'], snrs,
                                              cfg['preprocess']['sampling_rate'], cfg['preprocess']['spectrogram_max_length'])
        preprocessor.run(mixture_splits, mixture_path / "preprocessed", 5)


def check_augmentation(image_id: str, speed_augment: float, pitch_augment: float, noise_augment: float, 
                        mix_augment: float) -> str:
    """
    Based on the probabilities specified in the config, chooses an augmentation type and return respective image path.

    Args:
        image_id (str): the original image id, without any augmentation
        speed_augment (float): probability that a speed augmentation will be applied
        pitch_augment (float): probability that a pitch augmentation will be applied
        noise_augment (float): probability that a noise augmentation will be applied
        mix_augment (float): probability that a mixture augmentation will be applied
    
    Returns:
        (str): the altered image_id, pointing to an augmented version (if probabilities specified)
    """ 

    # Choose one of the augmentation types, based on the user-provided probabilities 
    probabilities = {
        'speed': speed_augment, 'pitch': pitch_augment,
        'noise': noise_augment, 'mix': mix_augment,
        'none': 1.0 - (speed_augment + pitch_augment + noise_augment + mix_augment)
    }
    augment_type = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))

    # Grab the file with the chosen augmentation type applied 
    l = image_id.split("/", 1)
    if augment_type == 'speed':
        aug = random.choice(["speedup", "slowdown"])
        image_id = l[0] + "/augments/speed_augments/" + str(aug) + "/" + l[1]
    elif augment_type == 'pitch':
        aug = random.choice(["shiftpitchup", "shiftpitchdown"])
        image_id = l[0] + "/augments/pitch_augments/" + str(aug) + "/" + l[1] 
    elif augment_type == 'noise':
        image_id = l[0] + "/augments/noise_augments/" + l[1]
    elif augment_type == 'mix':
        image_id = l[0] + "/augments/mix_augments/" + l[1]
    else:
        image_id = image_id

    return image_id

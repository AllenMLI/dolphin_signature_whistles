# This script will apply some augmentation, based on the augmentation argument passed in, to every
# wav file recursively within the given directory, in place. Make sure to pass in the directory
# path where you would like the augmented data saved, not the original audio files.
#
# Possible augmentations are as follow...
# speedup, slowdown, shiftpitchup, shiftpitchdown, resampleaudio, addrandomnoise


import sys
import os
import librosa
import numpy as np
import soundfile as sf
import random
from pathlib import Path


def augment_waveform(wav, sr, a, amount=None):
    """
    Augments the given raw waveform with the specified augmentation technique.

    Args:
        wav (np.array): time series wav file
        sr (int): sample rate
        a (str): augmentation technique
    """
    if a.lower() == 'speedup':
        if amount is None:
            amount = random.uniform(.75, .99)
        return librosa.effects.time_stretch(wav, amount)  # speed up the wav
    elif a.lower() == 'slowdown':
        if amount is None:
            amount = random.uniform(1.01, 1.25)
        return librosa.effects.time_stretch(wav, amount)  # slow down the wav
    elif a.lower() == 'shiftpitchup':
        if amount is None:
            amount = random.uniform(.01, 1)
        return librosa.effects.pitch_shift(wav, sr, n_steps=amount, bins_per_octave=10)  # shift the pitch up
    elif a.lower() == 'shiftpitchdown':
        if amount is None:
            amount = random.uniform(-1, -.01)
        return librosa.effects.pitch_shift(wav, sr, n_steps=amount, bins_per_octave=10)  # shift the pitch down
    elif a.lower() == 'resampleaudio':
        target_sr = random.choice([16000, 32000, 60000])
        return librosa.resample(wav, sr, target_sr)  # resample the audio
    elif a.lower() == 'addrandomnoise':
        if amount is None:
            amount = random.uniform(0.001, 0.500)
        noise_amp = amount*np.random.uniform()*np.amax(wav)
        return wav.astype('float64') + noise_amp * np.random.normal(size=wav.shape[0])  # add distributed random noise
    else:
        print("Please enter a valid augmentation option: speedup, slowdown, shiftpitchup, shiftpitchdown, resampleaudio, addrandomnoise")


def main(splits: dict, audio_dir: str, aug: str, sr: int = 60000):
    """
    Applies the specified augmentation to the given directory of wav files.
    Then preprocesses the augmented wav in the specified manner and saves as png.
    Args:
        splits (dict): dictionary of filepaths to train/val/test wavs
        audio_dir (str): path to where the augmented wavs will be saved
        aug (str): augmentation technique, can be: 'speedup', 'slowdown', 'shiftpitchup', 'shiftpitchdown'
        sr (int): sample rate for loading the audio
    Returns:
        augment_splits (dict): dictionary of augmented wav filepaths
    """
    print("Applying augmentation...", aug)

    augment_splits = {'train': [], 'val': [], 'test': []}
    for split, fps in splits.items():
        for fp in fps:

            if fp.endswith(".wav") or fp.endswith(".flac"):
                print("Augmenting...", fp)
            else:
                continue

            wav = None
            if fp.lower().endswith(".wav"):
                wav, sr = librosa.load(fp, sr)  # load in the audio
            elif fp.lower().endswith(".flac"):
                wav, fs = sf.read(fp, dtype='float32')

            augmented_wav = augment_waveform(wav, sr, aug)  # apply the desired augmentation

            s = fp.split("/")
            savename = audio_dir / "wavs" / split / s[3]
            if not os.path.exists(savename):
                os.makedirs(savename)
            sf.write(savename / s[-1], augmented_wav, sr)  # save the augmented waveform

            augment_splits[split].append(str(savename / s[-1]))  # add the augmented wav filepath to the augmented dict

    return augment_splits

# File containing functionality needing to mix various audio files together - such as overlaying underwater noise onto dolphin whistles.
#
# Example call of this script : python -m augment.mixing.mix bg_audio_path event_audio_path [0, 24], 3.0

from pathlib import Path
import os
import math
import random
import numpy as np
from scipy import signal
import soundfile as sf
import librosa


def to_shape(a, shape, random_pad=False):
    """
    Crops or pads a wav (time-series) to be a certain shape.
    Args:
        a (np.ndarray): time series array
        shape (int): the shape that the 1D time series should be
        random_pad (bool): whether to randomly pad or center pad
    Returns:
        (np.ndarray): altered time series array that is of shape (shape,)
    """
    x, x_ = a.shape[0], shape
    if shape < x:
        diff = int(x - shape)
        return a[:-diff]
    elif shape == x:
        return a  # no adjustment needed
    else:
        padding_required = math.ceil(x_ - x)
        if random_pad:
            left_padding = random.randint(1, padding_required)
        else:
            left_padding = padding_required // 2

        right_padding = padding_required - left_padding
        return np.pad(a, ((left_padding, right_padding)), mode = 'constant')


def rms(y: np.ndarray) -> float:
    """
    Compute root mean square amplitude of the given audio
    Args:
        y (np.ndarray): time series
    Returns:
        (float) root mean square amplitude
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))


def median_rms(y: np.ndarray, sr: int) -> float:
    """
    Compute the median value of the given audio
    Args:
        y (np.ndarray): time series
        sr (int): sampling rate
    Returns:
        (float) median rms value of the segments
    """
    block_length = int(0.1 * sr)
    num_segments = int(len(y) // block_length)
    segments = list()
    for i in range(1, num_segments+1):
        segments.append(rms(y[(i-1)*block_length: i*block_length]))
    return np.median(segments)


def get_event_amplitude_scaling_factor(s: np.ndarray, n: np.ndarray, target_snr_db: float, sr: int) -> float:
    """
    Different lengths for signal and noise allowed: longer noise assumed to be stationary enough,
    and rms is calculated over the whole signal
    Args:
        s (np.ndarray): the event audio time series
        n (np.ndarray): the background audio time series
        target_snr_db (float): target signal-to-noise-ratio
        sr (int): sampling frequency
    Returns:
        (float) the resulting scaling factor
    """

    # add 4-pole butterworth filter to bandpass 4-20kHz
    nyquist = 0.5 * sr
    bpf = signal.butter(2, [4000/nyquist, 20000/nyquist], analog=False, btype='bandpass', output='ba')
    s = signal.filtfilt(bpf[0], bpf[1], s)
    n = signal.filtfilt(bpf[0], bpf[1], n)

    original_sn_rms_ratio = median_rms(s, sr) / median_rms(n, sr)
    target_sn_rms_ratio = 10 ** (target_snr_db / float(20))
    signal_scaling_factor = target_sn_rms_ratio / (original_sn_rms_ratio + 0.000001)

    return signal_scaling_factor


def mix(bg_audio: np.ndarray, orig_audio: np.ndarray, pad_audio: np.ndarray, snr: int, sr: int) -> np.ndarray:
    """
    Mix np arrays of background and event audio according to the given snr and an anticlipping factor.
    Args:
        bg_audio (np.ndarray): background audio time series
        orig_audio (np.ndarray): dolphin audio time series
        pad_audio (np.ndarray): dolphin audio time series, padded to max_spec_length
        snr (int): signal-to-noise ratio chosen randomly for this mixture
        sr (int): sampling frequency
    Returns:
        mixture (np.ndarray): a query with the events mixed in and scaled according to ebr
    """
    # Multiply the event audio by a scaling factor that's dependent on the snr
    ssf = get_event_amplitude_scaling_factor(orig_audio, bg_audio, snr, sr)
    event_audio = pad_audio * ssf

    # Mix the scaled event with the background audio
    mixture = event_audio + bg_audio

    # An attempt at preventing clipping - normalize if the max value of time series is >= 1
    mixture_max = np.max(np.abs(mixture))
    if mixture_max >= 1:
        mixture /= mixture_max
   
    return mixture


def get_bg_clip(bg_files: list, event_wav: np.ndarray, dur: float, sr: int) -> np.ndarray:
    """
    Grab a background wav file from bg_dir and return a clip of it equal to duration.
    Args:
        bg_files (list): list of filepaths pointing to background wav files
        event_wav (np.ndarray): the dolphin time series in numpy form
        dur (float): duration of the event we want to mix the bg audio into
        sr (int): sample rate for loading the audio
    Returns:
        (np.ndarray): time series of the background audio clip
    """
    while True:
        bg_file = random.choice(bg_files)
        bg_wav, _ = librosa.load(bg_file, sr)
        bg_dur = librosa.get_duration(bg_wav, sr)

        # Check to see if this background file is sufficiently long
        if bg_dur < dur:
            print("The duration of the background wav file ", bg_file, " is too short. Choosing another.")
            continue

        # Grab a chunk of audio out of the background audio, equal to dur
        rand_start = random.randint(0, int(bg_dur - dur) * sr)  # random start to bg clip, in frames
        bg_wav = bg_wav[rand_start : rand_start + int(dur * sr)]
        diff = bg_wav.shape[0] - event_wav.shape[0]
        if diff != 0:
            bg_wav = np.append(bg_wav, bg_wav[-1])

        return bg_wav


def find_bg_files(bg_dir: Path) -> list:
    """
    Finds all wav files in the background audio directory recursively.
    
    Args:
        bg_dir (Path): path to the directory where background audio is located
    Returns:
        (list): list of filepaths pointing to background wav files
    """
    bg_fps = []
    for root, _, files in os.walk(bg_dir):
        for fp in files:
            if fp.endswith('.wav'):
                bg_fps.append(os.path.join(root, fp))
    return bg_fps


def main(splits: dict, audio_dir: Path, bg_dir: Path, snrs: list, sr: int, max_dur: float, random_pad: bool = False):
    """
    Mixes in background noise (bg_dir) to each wav file contained in splits.
    Args:
        splits (dict): dictionary of filepaths to train/val/test wavs 
        audio_dir (Path): path to where the augmented wavs will be saved
        snrs (list): list of [minimum_ratio, maximum_ratio]
        sr (int): sample rate for loading the audio
        max_dur (float): maximum duration that will be used as the window size
        random_pad (bool): whether to randomly pad or center pad the time series
    """
    print("Mixing background noises into whistles...")

    assert (splits is not None and audio_dir is not None and bg_dir is not None) 

    # Find all background audio wav files recursively in bg_dir
    bg_files = find_bg_files(bg_dir)

    # Generate mixtures and save these to the outputs/mixtures/ directory
    mix_splits = {'train': [], 'val': [], 'test': []}
    for split, fps in splits.items():

        for fp in fps:

            if split == 'train':
                continue

            if fp.endswith(".wav"):
                print("Mixing...", fp)
            else:
                continue

            wav, _ = librosa.load(fp, sr)  # load in the given original dolphin audio wav
            pad_wav = to_shape(wav, (max_dur * sr) + 1, random_pad=random_pad)  # pad or crop the wav to be max_dur seconds

            bg_wav = get_bg_clip(bg_files, pad_wav, max_dur, sr)  # extract audio from a randomly chosen background clip
            snr = random.uniform(min(snrs), max(snrs))  # choose random signal-to-noise ratio from uniform distribution

            mixture = mix(bg_wav, wav, pad_wav, snr, sr)

            # Save the raw wav mixture 
            s = fp.split("/")
            savename = audio_dir / "wavs" / split / s[3]
            if not os.path.exists(savename):
                os.makedirs(savename)
            sf.write(savename / s[-1], mixture, sr)

            # Add the path to the saved mixture to the dictionary that will be returned
            mix_splits[split].append(str(savename / s[-1]))

    return mix_splits

"""
Preprocessing steps to create spectrogram, melspec, and pcen of input
Reference: https://www.lostanlen.com/wp-content/uploads/2019/12/lostanlen2019spl.pdf
"""

from typing import Tuple
import random
import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
import cv2


def compute_spectrogram(y: np.ndarray, sr: int, cfg: dict, random_pad: bool) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a basic spectrogram with the time series array
    Args:
        y: time series data
        sr: sampling rate used by librosa for generating the time series
        cfg: configuration variables
        random_pad: to pad randomly or not to pad randomly

    Returns:
        spectrogram in db, frequency vector, and time vector
    """

    nfft = cfg['preprocess']['nfft']
    spec_max_length = cfg['preprocess']['spectrogram_max_length']

    f, t, spectrum = signal.spectrogram(y, sr, scaling='density', window=cfg['preprocess']['window'],
                                        nfft=nfft, nperseg=nfft, noverlap=nfft//2)
    spectrum_db = 10 * np.log10(spectrum)

    # set the number of samples we need per image for padding
    max_num_sample = int(np.ceil(spec_max_length / t[0]))

    # remove lower x percentiles of a data set, set to 0 to prevent thresholding
    # b[b < x-th perentile] = x-th pecentile value
    _percentile = np.percentile(spectrum_db, cfg['preprocess']['contrast_percentile'])
    spectrum_db[spectrum_db < _percentile] = _percentile

    # need padding in the x-axis
    padding_required = max_num_sample - spectrum_db.shape[1]
    if padding_required > 0:

        # optional random padding, else center padding
        if random_pad:
            left_padding = random.randint(1, padding_required)
        else:
            left_padding = padding_required // 2

        right_padding = padding_required - left_padding
        c = np.float64(spectrum_db.min())
        spectrum_db = cv2.copyMakeBorder(spectrum_db, 0, 0, left_padding, right_padding,
                                         cv2.BORDER_CONSTANT, value=[c, c, c])
        t = np.linspace(t[0], spec_max_length, max_num_sample)

    return spectrum_db, f, t


def compute_melspec(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Create a melspec with the time series array
    Args:
        y: time series data
        sr: sampling rate used by librosa for generating the time series
        cfg: configuration variables

    Returns:
        mel-spectrogram
    """
    return librosa.feature.melspectrogram(y,
                                          sr=sr,
                                          fmin=cfg['preprocess']['melspec_fmin'],  # lowest frequency in Hz
                                          fmax=cfg['preprocess']['melspec_fmax'],  # highest frequency (Hz)
                                          hop_length=cfg['preprocess']['melspec_hop_length'],  # num samples between successive frames
                                          n_mels=cfg['preprocess']['melspec_n_mels'])  # number of mel cepstral

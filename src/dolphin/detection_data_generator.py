"""
Data generator and supporting functions for detection data
"""
# ---------------------------------------------------------------------------------------------------------------------
# Library import
# ---------------------------------------------------------------------------------------------------------------------
import sys
import os
import random
import subprocess
import csv
import shutil
import numpy as np
import librosa
import soundfile as sf
import cv2
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ---------------------------------------------------------------------------------------------------------------------
# Internal package import
# ---------------------------------------------------------------------------------------------------------------------
sys.path.append('.')
import dolphin.utils as utils
import dolphin.io_utils as io_utils
from dolphin.preprocess import feature_extraction

def read_in_spec(fpath):
    """
    Read in a spectrogram
    Input:
        fpath: full path to a spectrogram already resized/padded to correct input size for model
                and with wav or mixture augmentation already applied
    Returns:
        spec: the spectrogram opened, with spec augmentations applied - ready to be input to the model
    """
    # open and normalize the image values to range of [0,1]
    spec = cv2.imread(fpath) / 255

    return np.array(spec, dtype='float32')

def remove_clicks_from_one_spec(save_dir, input_path, split, data, nfft, window, cfg):
    """
    Remove the clicks from just one audio file and save it as a spectrogram

    Args:
        save_dir (str): path to directory where spectrograms will be saved
        input_path (str): path to original wav file we're removing clicks from
        split (str): train, val or test - the set this file belongs to
        data: wav file pre-loaded by librosa.load() OR flac file pre-loaded by sf.read()
        nfft (int): nfft to use when generating this spectrogram
        window (str): windowing method to use when generating this spectrogram
        cfg (dict): dictionary holding all config file arguments
    """
    # generate save name
    base_wav_fname = os.path.basename(input_path)
    spec_base = base_wav_fname.rsplit('.', 1)[0]
    spec_name = str(spec_base) + '.png'

    class_name = 'whistles'
    if 'non-whistles' in input_path:
        class_name = 'non-whistles'

    # saving to output_dir/augments/no_clicks/preprocessed/spec/[train,test,val]/[whistles,non-whistles]/spec_name
    save_path = os.path.join(save_dir, split, class_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    savename = os.path.join(save_path, spec_name)

    # Compute harmonic and percussive components of the freq spectrum
    freq = librosa.stft(data, n_fft=nfft, hop_length=int(nfft/2), window=window)
    d_harmonic, d_percussive = librosa.decompose.hpss(freq, margin=2.0)
    rp = np.max(np.abs(freq))

    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plt.axis('off')
    fig.set_size_inches(cfg['preprocess']['spectrogram_max_length'] * cfg['output']['inches_per_sec'],
                        cfg['preprocess']['sampling_rate'] / 1000 / 2 * cfg['output']['inches_per_KHz'])

    # remove the clicks
    full_spectrum = librosa.amplitude_to_db(np.abs(freq), ref=rp)
    percussive = librosa.amplitude_to_db(np.abs(d_percussive), ref=rp)
    without_clicks = full_spectrum - 0.5 * percussive
    without_clicks_spec = librosa.display.specshow(without_clicks, cmap=cfg['output']['color_map'])
    fig.savefig(savename, transparent=False, bbox_inches='tight', format='png')
    plt.close()


def remove_clicks(removed_clicks_path: str, splits: dict, cfg: dict):
    """
    Generate whole folder of spectrograms without clicks (percussive elements)

    Args:
        removed_clicks_path (str): path to folder where specs with clicks removed will be saved
        splits (dict): the dictionary holding train/val/test filepaths to original wav files
        cfg (dict): dictionary holding all config file arguments
    """

    sampling_rate = cfg['preprocess']['sampling_rate']
    nfft = cfg['preprocess']['nfft']
    window = cfg['preprocess']['window']
    for split, fps in splits.items():
        for fp in fps:
            if fp.lower().endswith('.wav') or fp.lower().endswith('.flac'):

                try:
                    if fp.lower().endswith('.wav'):
                        data, sampling_rate = librosa.load(fp, sr=sampling_rate)
                    elif fp.lower().endswith('.flac'):
                        data, fs = sf.read(fp, dtype='float32')

                    remove_clicks_from_one_spec(removed_clicks_path, fp, split, data, nfft, window, cfg)

                except Exception as e:
                    print('error msg: ', e)


def decide_augment(file_path, preprocessed_dir, split, pitch_augment, speed_augment, remove_clicks_aug):
    """
    Grab randomly from the pitch and/or speed shifted and/or removed clicks spectrogram buckets

    Args:
        file_path (str): the filepath to the original audio version of this clip
        preprocessed_dir (Path): the path to the preprocessed features
        split (str): the train/val/test split for this file
        pitch_augment (float): likelihood of pitch augmentation being added to file
                                (defined in config)
        speed_augment (float): likelihood of speed augmentation being added to file
                                (defined in config)
        remove_clicks_aug (float): likelihood of clicks being removed from file
                                (defined in config)
    Returns: str filepath to the augmented (or not augmented) version of the
            spectrogram of the input file_path

    """
    # Roll the die for the purpose of deciding if/what augmentation we use for this image
    # Currently, pitch and speed augmentations are supported
    spec_dir, spec_filepath = utils.get_savename(preprocessed_dir, file_path, split, 3)
    spec_filepath = str(spec_filepath)

    # Choose one of the augmentation types, based on the user-provided probabilities
    probabilities = {
        'speed': speed_augment, 'pitch': pitch_augment,
        'remove_clicks': remove_clicks_aug,
        'none': 1.0 - (speed_augment + pitch_augment + remove_clicks_aug)
    }
    augment_type = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))

    # Grab the file with the chosen augmentation type applied
    l = spec_filepath.split('/', 1)
    if augment_type == 'speed':
        aug = random.choice(['speedup', 'slowdown'])
        spec_filepath = os.path.join(l[0], 'augments', 'speed_augments', str(aug), l[1]) #l[0] + '/augments/speed_augments/' + str(aug) + '/' + l[1]
    elif augment_type == 'pitch':
        aug = random.choice(['shiftpitchup', 'shiftpitchdown'])
        spec_filepath = os.path.join(l[0],'augments','pitch_augments',str(aug),l[1])
    elif augment_type == 'remove_clicks':
        spec_filepath = os.path.join(l[0],'augments','no_clicks',l[1])
    else:
        spec_filepath = spec_filepath


    return spec_filepath

def segment_audio_for_detection(test_data_folder, cfg):
    """
    Given an folder of audio files, split them into clips of spectrogram_max_length
    and convert them to spectrograms and save

    Args:
        test_data_folder (str): path to a folder of wav or flac files of any length
        cfg (dict): config file

    Returns:
        spec_paths_list: list of paths to spectrograms of spectrogram_max_length
        start_times: dict with spectrogram names as keys, values = integer start times
                    relative to the longer wav or flac this clip came from
    """
    spec_paths_list = []
    start_times = {}
    spec_output_folder = os.path.join(test_data_folder, 'spec')
    audio_clip_output_folder = os.path.join(test_data_folder, 'audio_segments')
    spectrogram_length = cfg['preprocess']['spectrogram_max_length']

    if not os.path.exists(spec_output_folder):
        os.makedirs(spec_output_folder)
    if not os.path.exists(audio_clip_output_folder):
        os.makedirs(audio_clip_output_folder)

    # split each input audio file into multiple spectrogram_length-long clips
    for audio_file in os.listdir(test_data_folder):
        if audio_file.lower().endswith('.wav') or audio_file.lower().endswith('.flac'):
            audio_file_path = os.path.join(test_data_folder, audio_file)
            split_file_name = os.path.splitext(audio_file)
            output_name_pattern = audio_clip_output_folder+os.sep+split_file_name[0]+'_%05d.wav'
            intervals = subprocess.call(['ffmpeg', '-i', audio_file_path, '-f', 'segment', '-segment_time',
                            str(spectrogram_length), output_name_pattern])

    for audio_segment in os.listdir(audio_clip_output_folder):
        # convert from wav to spectrogram
        audio_fp = os.path.join(audio_clip_output_folder, audio_segment)
        data, sr = librosa.load(audio_fp, sr=cfg['preprocess']['sampling_rate'], duration=spectrogram_length)
        feature, f, t = feature_extraction.compute_spectrogram(data, sr=sr, cfg=cfg, random_pad=False)
        file_base = os.path.splitext(audio_segment)[0]
        save_path = os.path.join(spec_output_folder, file_base+'.png')

        # calculate the start time of this clip
        clip_num = save_path.split('_')[-1].split('.')[0]
        start_time = int(clip_num)*spectrogram_length
        start_times[save_path] = start_time

        # save the spectrogram
        io_utils.save_fig(feature, f, t, output_dir=save_path, cfg=cfg)
        spec_paths_list.append(save_path)

    shutil.rmtree(audio_clip_output_folder)

    return spec_paths_list, start_times

def write_to_csv_val_or_test(test_data_path: str, csv_folder_path: str,
                    predictions: dict, sr: int, spectrogram_max_length: int,
                    confidence_threshold: float):
    """
    Writes results to CSV for a pre-split, pre-processed validation or test dataset

    Args:
        test_data_path (str): path to folder of spectrograms that were run through detector
        csv_folder_path (str): path to folder where we want to output CSV results
        predictions (dict): keys=spec paths, values=probability output of detector
        sr (int): sampling rate
        spectrogram_max_length (int): length of detections
        confidence_threshold (float): minimum confidence value to be considered a whistle
    """
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    # iterate over each spectrogram and add to results if confidence >= confidence_threshold
    savename = os.path.join(csv_folder_path, 'all_results_{}.csv'.format(os.path.basename(test_data_path)))
    selection_index = 0
    start_time = 0
    end_time = spectrogram_max_length
    columns = ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
               'Low Freq (Hz)', 'High Freq (Hz)', 'Filepath', 'Found']

    with open(savename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns, delimiter='\t')
        writer.writeheader()

        for class_name in os.listdir(test_data_path):
            if class_name.startswith('.'): # ignore hidden files
                continue
            class_path = os.path.join(test_data_path, class_name)
            for fp in os.listdir(class_path):
                full_fp = os.path.join(class_path, fp)
                if fp.lower().endswith('.png') or fp.lower().endswith('.jpg'):
                    conf = predictions[full_fp]
                    # if we predicted this clip contained a whistle, add it to the csv file
                    if conf >= confidence_threshold:
                        writer.writerow({'Selection': selection_index, 'View': 'Spectrogram 1',
                                        'Channel': '1', 'Begin Time (s)': start_time, 'End Time (s)': end_time,
                                        'Low Freq (Hz)': 0.0, 'High Freq (Hz)': sr / 2, 'Filepath': fp,
                                        'Found': 'whistle'})
                        selection_index += 1


def write_to_csv(test_data_path: str, csv_folder_path: str, start_times: dict,
                    predictions: dict, sr: int, spectrogram_max_length: int,
                    confidence_threshold: float):
    """
    Takes the audio file folder path, output CSV folder path, start times of detections,
    and model predictions and writes to csv file.

    Args:
        test_data_path (str): path to folder of audio files that were run through detector
        csv_folder_path (str): path to folder where we want to output CSV results
        start_times (dict): keys=spec paths, values=start times of whistles, relative to orig wav
        predictions (dict): keys=spec paths, values=probability output of detector
        sr (int): sampling rate
        spectrogram_max_length (int): length of detections
        confidence_threshold (float): minimum confidence value to be considered a whistle
    """
    # grab all audio files in the test folder
    fps = os.listdir(test_data_path)

    # create folder for saving CSVs if it doesn't exist already
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    create_multiple_csvs = True
    for fp in fps:
        # check duration of first audio file in folder - if longer files, will create one csv per input file
        if fp.lower().endswith('.wav') or fp.lower().endswith('.flac'):
            full_fp = os.path.join(test_data_path, fp)
            audio_duration = librosa.get_duration(filename=full_fp)
            # if it's less than 5 seconds - going to create one overall csv of results
            # because we can be pretty sure that all inputs are short extracted clips
            if audio_duration <= 5:
                create_multiple_csvs = False
            break # only need to look at first audio file in the folder


    if create_multiple_csvs:
        # Iterate over the list of fps - creating one csv per original file path
        for fp in fps:
            if fp.lower().endswith('.wav') or fp.lower().endswith('.flac'):
                base_file_name = os.path.splitext(fp)[0] # just file name without extension
                csv_name = base_file_name+'.csv'
                savename = os.path.join(csv_folder_path, csv_name)
                selection_index = 0
                columns = ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
                           'Low Freq (Hz)', 'High Freq (Hz)', 'Filepath', 'Found']


                with open(savename, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=columns, delimiter='\t')
                    writer.writeheader()

                    # iterate over all start times to find the detections that occurred in this audio file
                    for clip_name in sorted(start_times):
                        if base_file_name in clip_name:
                            conf = predictions[clip_name]
                            # if we predicted this clip contained a whistle, add it to the csv file
                            if conf >= confidence_threshold:
                                start_time = start_times[clip_name]
                                end_time = int(start_time) + spectrogram_max_length
                                writer.writerow({'Selection': selection_index, 'View': 'Spectrogram 1',
                                                'Channel': '1', 'Begin Time (s)': start_time, 'End Time (s)': end_time,
                                                'Low Freq (Hz)': 0.0, 'High Freq (Hz)': sr / 2, 'Filepath': fp,
                                                'Found': 'whistle'})
                                selection_index += 1
    else:
        # iterate over the list of start times and add all detections to one csv file
        savename = os.path.join(csv_folder_path, 'all_results.csv')
        selection_index = 0

        csvdict = {}
        columns = ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
                   'Low Freq (Hz)', 'High Freq (Hz)', 'Filepath', 'Found']

        with open(savename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns, delimiter='\t')
            writer.writeheader()

            for fp in fps:
                # if this fp is a spectrogram or an audio file, add it to the csv
                if fp.lower().endswith('.wav') or fp.lower().endswith('.flac'):
                    base_file_name = os.path.splitext(fp)[0]
                    # iterate over all start times to find the detections that occurred in this audio file
                    for clip_name in sorted(start_times):
                        if base_file_name in clip_name:
                            conf = predictions[clip_name]
                            # if we predicted this clip contained a whistle, add it to the csv file
                            if conf >= confidence_threshold:
                                start_time = start_times[clip_name]
                                end_time = int(start_time) + spectrogram_max_length
                                writer.writerow({'Selection': selection_index, 'View': 'Spectrogram 1', 'Channel': '1',
                                                'Begin Time (s)': start_time, 'End Time (s)': end_time,
                                                'Low Freq (Hz)': 0.0, 'High Freq (Hz)': sr / 2, 'Filepath': fp,
                                                'Found': 'whistle'})
                                selection_index += 1


class BatchGenerator(Sequence):
    """
    Generates regular batches of spectrograms (just batch_size spectrograms
    returned in an array). Randomly adds in pitch, speed, and click removal
    augmentations in the percentages specified in the config.
    """

    def __init__(self, file_class_mapping, split, preprocessed_dir,
                    pitch_augment, speed_augment, batch_size,
                    remove_clicks_aug):
        self.x = np.array(list(file_class_mapping.keys()))
        self.y = np.array(list(file_class_mapping.values()))
        self.preprocessed_dir = preprocessed_dir
        self.batch_size = batch_size
        self.shuffle = True
        self.split = split
        self.speed_augment = speed_augment
        self.pitch_augment = pitch_augment
        self.remove_clicks_aug = remove_clicks_aug

    def on_epoch_end(self):
        indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(indexes)
            self.x = self.x[indexes]
            self.y = self.y[indexes]

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x_ready_to_input = []
        for item in batch_x:
            # select augmentation for this spectrogram
            spec_path = decide_augment(item, self.preprocessed_dir, self.split,
                                        self.pitch_augment, self.speed_augment,
                                        self.remove_clicks_aug)
            spec_to_input = read_in_spec(spec_path)

            # add that spectrogram to this batch
            batch_x_ready_to_input.append(spec_to_input)

        batch_x_prepped = np.array(batch_x_ready_to_input)

        return batch_x_prepped, np.array(batch_y)

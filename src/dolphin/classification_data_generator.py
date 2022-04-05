"""DataGenerator"""
import os
import sys
from typing import Tuple
from pathlib import Path
import numpy as np
import cv2
import sklearn.preprocessing

from tensorflow.keras.utils import Sequence, to_categorical

sys.path.append('src')
from dolphin.augment.augment_utils import check_augmentation


class DataGenerator(Sequence):
    """
    DataGenerator grabs and loads batches of data.
    """
    def __init__(self, data_dir: Path, input_shape: tuple, cfg: dict, speed_augment: float = 0.0,
                pitch_augment: float = 0.0, noise_augment: float = 0.0, mix_augment: float = 0.0,
                classes: list = None):

        # Model variables
        self.batch_size = cfg['model']['model_params']['batch_size']
        self.shuffle = cfg['model']['model_params']['shuffle']
        self.dim = input_shape

        # Augment Variables
        self.speed_augment = speed_augment
        self.pitch_augment = pitch_augment
        self.noise_augment = noise_augment
        self.mix_augment = mix_augment

        # Data dictionary with the format {dolphin ID : [list of images for that dolphin]}
        self.data_dict = {d: [os.path.join(data_dir, d, dd)
                              for dd in os.listdir(os.path.join(data_dir, d))] for d in os.listdir(data_dir)}
        self.image_ids = [(k,v) for k in self.data_dict for v in self.data_dict[k]]

        # The classes are determined by the training set
        self.classes = classes
        if self.classes is None:
            self.classes = list(self.data_dict.keys())
        self.n_classes = len(self.classes)

        # Label encoder
        label_encoder = sklearn.preprocessing.LabelEncoder()
        self.label_encoder = label_encoder.fit(np.sort(list(self.classes)))

        # Initialize variables needed for data generation
        self.indexes = None
        self.on_epoch_end()
        self.cnt = 0    # initialization for calling __next__

    def encode_label(self, y: list):
        """
        Encode label to categorical representation

        Args:
            y (list): class labels to be converted to a matrix

        Returns
            (np.ndarray) a matrix that encodes the labels
        """
        y_labels = to_categorical(self.label_encoder.transform(y), num_classes=self.n_classes)
        return y_labels

    def __len__(self):
        """
        Denotes the number of batches per epoch

        Returns:
            (int) the number of batches per epoch
        """
        length = 0
        for _, val in self.data_dict.items():
            length += len(val)
        return int(np.floor(length / self.batch_size))

    def __next__(self):
        """
        Get the next item in the generator, appear to be needed for distributed
        training to work when calling fit_generator
        Returns:
            (np.ndarray) next batch of data
        """
        if self.cnt >= self.__len__():
            # Reset iter index and shuffle samples
            self.on_epoch_end()
            self.cnt = 0

        items = self.__getitem__(self.cnt)
        self.cnt += 1
        return items

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data
        Args:
            idx (int): index of where we are in the full set of data for batching [int]
            bootstrap (bool): Boolean flag to determine to bootstrap data or not
        Returns:
            (np.ndarray) batch of data
            (np.ndarray) corresponding encoded labels
        """
        # Indexes are shuffled
        idxs = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        ids = [self.image_ids[k] for k in idxs]  # get list of ids

        X, y = self.__data_generation(ids)
        return X, self.encode_label(y)

    def on_epoch_end(self):
        """
        Shuffle our data between epochs
        """
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of data and corresponding labels with option augmentation applied

        Args:
            ids (list): list of integers representing indices into image_ids
        Returns:
            (Tuple[np.ndarray, np.ndarray]): data and labels
        """
        X = np.empty((self.batch_size, *self.dim))
        y = list()

        for i, list_idx in enumerate(ids):
            image_label, image_id = list_idx

            # If user-provided probabilities specify, grab the file with a supported augmentation type applied
            # If no augmentation is desired, image_id is left as is
            image_id = check_augmentation(image_id, self.speed_augment, self.pitch_augment,
                                          self.noise_augment, self.mix_augment)

            X[i,] = np.expand_dims(cv2.imread(image_id) / 255, axis=0)
            y.append(image_label)

        return X, np.array(y)

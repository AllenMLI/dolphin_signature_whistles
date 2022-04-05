"""
Pipeline for start to finish in preprocess and modeling data
An attempt to minimize number of scripts doing individual things
"""

# ---------------------------------------------------------------------------------------------------------------------
# Library import
# ---------------------------------------------------------------------------------------------------------------------
import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import atexit

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import model_from_json
import wandb
from wandb.keras import WandbCallback

# ---------------------------------------------------------------------------------------------------------------------
# Internal package import
# ---------------------------------------------------------------------------------------------------------------------
sys.path.append('src')
import dolphin.utils as utils
import dolphin.io_utils as io_utils
import dolphin.preprocess as preprocess
import dolphin.augment.augment_utils as augment_utils
from dolphin.classification_data_generator import DataGenerator
from dolphin.models import MODELS


def main(cfg_filename='config.json'):
    """
    Entry point for classification
    """
    # -----------------------------------------------------------------------------------------------------------------
    # Load Configuration
    # -----------------------------------------------------------------------------------------------------------------
    with open(cfg_filename, 'r') as f:
        cfg = json.load(f)

    # -----------------------------------------------------------------------------------------------------------------
    # Set up outputs dir
    # -----------------------------------------------------------------------------------------------------------------
    output_dir = Path(cfg['paths']['output_path'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(cfg['paths']['checkpoint_path']):
        os.makedirs(cfg['paths']['checkpoint_path'])

    # -----------------------------------------------------------------------------------------------------------------
    # Set up WandB logging
    # -----------------------------------------------------------------------------------------------------------------
    # automatically set debug to True if inference is True so that we don't log to wandb during inference
    if cfg['setup']['inference']:
        debug = True

    debug = bool(cfg['setup']['debug'])
    if not debug:
        wandb.init(project='dolphins', entity='allen_mli', config=cfg)

    # -----------------------------------------------------------------------------------------------------------------
    # Data Paths
    # -----------------------------------------------------------------------------------------------------------------
    data_dir = Path(cfg['paths']['data_path'])

    # -----------------------------------------------------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------------------------------------------------
    # Directory where all preprocessed data will be saved (ex. spectrograms)
    preprocessed_data_dir = output_dir / 'preprocessed'

    # If the preprocessed directory already exists, we do NOT need to generate train/val/test splits
    # If it doesn't exist, we do need to generate new splits
    if os.path.exists(preprocessed_data_dir):
        splits = utils.grab_splits(preprocessed_data_dir, data_dir)
    else:
        splits = utils.split_files(data_dir, dataset=cfg['dataset']['name'])

    # Intialize the preprocessor object, which takes care of all preprocessing and feature extraction
    preprocessor = preprocess.Preprocessor(cfg)
    preprocessed_data_dir = preprocessor.run(splits, preprocessed_data_dir, 3)

    # Load in a single datapoint so we can extract the input shape for the model
    input_shape = utils.get_input_shape(preprocessed_data_dir)

    # -----------------------------------------------------------------------------------------------------------------
    # Augmentation and Mixing
    # -----------------------------------------------------------------------------------------------------------------
    speed_augment = cfg['augment']['speed_augment']
    pitch_augment = cfg['augment']['pitch_augment']
    noise_augment = cfg['augment']['noise_augment']
    mix_augment = cfg['augment']['mix_augment']

    # Generate all of the desired augmentations
    augment_utils.generate_augmentations(output_dir, splits, cfg, preprocessor,
                                speed_augment, pitch_augment, noise_augment, mix_augment)

    # -----------------------------------------------------------------------------------------------------------------
    # Data Generators
    # -----------------------------------------------------------------------------------------------------------------
    train_dir = preprocessed_data_dir / 'train'
    train_generator = DataGenerator(train_dir, input_shape, cfg, speed_augment=speed_augment,
                                    pitch_augment=pitch_augment, noise_augment=noise_augment,
                                    mix_augment=mix_augment)

    # n_classes for training and evaluating on dev set is based on the unique labels from training set
    n_classes = train_generator.n_classes
    if not os.path.exists(os.path.join(output_dir, 'classification')):
        Path(os.path.join(output_dir, 'classification')).mkdir(parents=True)

    np.save(os.path.join(output_dir, 'classification', 'classes.npy'), train_generator.classes)

    val_dir = preprocessed_data_dir / 'val'
    val_generator = DataGenerator(val_dir, input_shape, cfg, classes=list(train_generator.classes))

    # -----------------------------------------------------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------------------------------------------------
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = MODELS[cfg['model']['model_name']](include_top=True, weights=None, input_shape=input_shape,
                                                   classes=n_classes)

        # serialize the model to json (in case of needing to reload later)
        model_json_path = os.path.join(cfg['paths']['checkpoint_path'], 'model.json')
        model_json = model.to_json()
        with open(model_json_path, 'w') as json_file:
            json_file.write(model_json)

        model.compile(optimizer=optimizers.Adam(learning_rate=cfg['model']['model_params']['learning_rate']),
                      loss='categorical_crossentropy', metrics=['acc'])

    # -----------------------------------------------------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------------------------------------------------
    if not cfg['setup']['inference']:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=cfg['model']['model_params']['patience'], verbose=1, mode='auto'),
            ModelCheckpoint(os.path.join(cfg['paths']['checkpoint_path'], 'weights_{epoch:08d}-{val_loss:.4f}.h5'),
                            monitor='val_loss', mode='auto', save_weights_only=True,
                            save_best_only=True, save_freq='epoch')
        ]

        if not debug:
            wdb = WandbCallback(save_model=False, save_graph=False, predictions=0)
            callbacks.append(wdb)

        # using save_freq='epoch' evaluates the need to save checkpoints every epoch. If we also use validation_freq
        # option below, we should remove the save_freq above.
        history = model.fit(train_generator,
                            validation_data=val_generator,
                            steps_per_epoch=train_generator.__len__(),
                            validation_steps=val_generator.__len__(),
                            validation_freq=1,
                            epochs=cfg['model']['model_params']['num_epochs'],
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            verbose=1)

        if cfg['paths']['history_path']:
            io_utils.save_history(Path(cfg['paths']['history_path']), history.history)

    # -----------------------------------------------------------------------------------------------------------------
    # Inference and output predictions
    # -----------------------------------------------------------------------------------------------------------------
    if cfg['setup']['inference']:
        print('Evaluating model...')
        model_json_path = os.path.join(cfg['paths']['checkpoint_path'], 'model.json')
        with open(model_json_path, 'r') as f:
            saved_model = model_from_json(f.read())

        checkpoints = [cfg['paths']['checkpoint_path'] + '/' + name for name in
                       os.listdir(cfg['paths']['checkpoint_path']) if name.endswith('h5')]

        print(cfg.keys())
        if cfg['paths']['inference_weights']:
            print('Loading weights from', cfg['paths']['inference_weights'])
            latest_checkpoint = cfg['paths']['inference_weights']
        else:
            print('No specific weights director was given, restoring from the latest checkpoint, '
                  'or edit configuration file variable paths->inference_weights')
            latest_checkpoint = max(checkpoints, key=os.path.getctime)

        print('Restored weights from', latest_checkpoint)
        saved_model.load_weights(latest_checkpoint)
        print('Weights loaded into saved model.')

        # use val directory during development, switch to user defined (i.e test)
        # feed classes into the LabelEncoder if using DataGenerator
        classes = np.sort(np.load(os.path.join(output_dir, 'classification', 'classes.npy')))
        if cfg['paths']['files_for_inference']:
            inference_dir = cfg['paths']['files_for_inference']
        else:
            inference_dir = preprocessed_data_dir / 'test'

        predicted_labels = {}
        predicted_probabilities = {}
        for root, _, files in os.walk(inference_dir):
            for file in files:
                fp = os.path.join(root, file)
                x = np.expand_dims(cv2.imread(fp) / 255, axis=0)
                predictions = saved_model.predict(x)
                top_predictions = np.flip(np.argsort(predictions[0])[-5:])
                predicted_labels[file] = [classes[p] for p in top_predictions]
                predicted_probabilities[file] = [round(p, 4) for p in np.flip(np.sort(predictions[0])[-5:])]

        io_utils.write_to_csv_classifier(csv_file=Path(os.path.join(output_dir, 'classification',
                                                                    'classifier_top_predictions.csv')),
                                         predicted_labels=predicted_labels,
                                         predicted_probabilities=predicted_probabilities)

    atexit.register(strategy._extended._collective_ops._pool.close)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

"""
Pipeline for start to finish preprocessing and modeling data for a detection model
"""

# ----------------------------------------------------------------------------------
# Library import
# ----------------------------------------------------------------------------------
import sys
import os
from pathlib import Path
import json
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet import ResNet50, ResNet152
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json, save_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryFocalCrossentropy
import wandb
from wandb.keras import WandbCallback

# ----------------------------------------------------------------------------------
# Internal package import
# ----------------------------------------------------------------------------------
sys.path.append('src')
import dolphin.detection_data_generator as detection_data_generator
import dolphin.utils as utils
import dolphin.preprocess as preprocess
from dolphin.augment import augment_utils
from dolphin.visualization import gradcam

random.seed(1234)

def main(cfg_filename='detection_config.json'):
    """
    Entry point for detection
    """

    # ------------------------------------------------------------------------------
    # Load Configuration
    # ------------------------------------------------------------------------------
    with open(cfg_filename, 'r') as f:
        cfg = json.load(f)

    # ------------------------------------------------------------------------------
    # Set up outputs dir
    # ------------------------------------------------------------------------------
    output_dir = Path(cfg['paths']['output_path'])  # path for outputs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ------------------------------------------------------------------------------
    # Set up WandB logging
    # ------------------------------------------------------------------------------
    inference = cfg['setup']['inference']
    debug = bool(cfg['setup']['debug'])

    # automatically set debug to True if inference is True so that we don't log to wandb during inference
    if inference:
        debug = True

    if not debug:
        wandb.init(project='dolphins', entity='allen_mli' , config=cfg)

    # ------------------------------------------------------------------------------
    # setting up model paths
    # ------------------------------------------------------------------------------
    LR_value = cfg['model']['detection_params']['learning_rate']
    batch_size = cfg['model']['detection_params']['batch_size']
    model_type = cfg['model']['model_type']
    speed_augment = cfg['augment']['speed_augment']
    pitch_augment = cfg['augment']['pitch_augment']
    max_spec_length = cfg['preprocess']['spectrogram_max_length']
    remove_clicks_aug = cfg['augment']['remove_clicks']

    model_name = cfg['model']['model_name']
    if model_name == 'MobileNetV2':
        num_frozen_layers = cfg['model']['detection_params']['num_frozen_layers_mobilenetv2']
    elif model_name == 'ResNet50':
        num_frozen_layers = cfg['model']['detection_params']['num_frozen_layers_resnet50']
    elif model_name == 'ResNet152':
        num_frozen_layers = cfg['model']['detection_params']['num_frozen_layers_resnet152']
    else:
        print('ERROR: need to input accepted model_name in config' \
              '(either MobileNetV2, ResNet50, or ResNet152)')
        sys.exit(0)

    # generate model save name
    best_model_path = cfg['paths']['best_model_path']
    model_save_path = os.path.join(best_model_path,
                                    ('dolphin_detection/{}_{}_w_focal_loss_LR_{}_batch_size_{}_speed_aug_{}_' +
                                    'pitch_aug_{}_click_remove_aug_{}_frozen_layers_{}_spec_length_{}')
                                    .format(model_type, model_name, LR_value, batch_size, speed_augment,
                                            pitch_augment, remove_clicks_aug, num_frozen_layers, max_spec_length))

    if not debug:
        wandb.log({'model_save_path': model_save_path})


    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # set paths for model save files (h5 and json)
    reload_weights = os.path.join(model_save_path, cfg['paths']['reload_model_file_name'])
    model_json_path = os.path.join(model_save_path, '{}.json'.format(model_type))

    # ------------------------------------------------------------------------------
    # Data prep only required for training model (not inference/evaluate)
    # ------------------------------------------------------------------------------
    preprocesed_data_dir = ""
    splits = {}
    input_shape = []
    file_id_mapping_train = {}
    file_id_mapping_val = {}

    if not cfg['setup']['inference']:
        # ------------------------------------------------------------------------------
        # set up data paths
        # ------------------------------------------------------------------------------
        # raw wav files - one folder per class (whistles/no_whistles)
        data_dir = Path(cfg['paths']['data_path'])

        # partitioned data folder is data that's already transformed into
        #       spectrograms, augmented, and split into train/val/test
        splits = utils.split_files(data_dir, dataset=cfg['dataset']['name'])

        # ------------------------------------------------------------------------------
        # Preprocessing - only do this if training a model
        # ------------------------------------------------------------------------------

        # Directory where all preprocessed data will be saved (ex. spectrograms)
        preprocessed_data_dir = output_dir / 'preprocessed'

        # Intialize the preprocessor object, which takes care of all preprocessing and feature extraction
        preprocessor = preprocess.Preprocessor(cfg)
        preprocessed_data_dir = preprocessor.run(splits, preprocessed_data_dir, 3)

        # Load in a single datapoint so we can extract the input shape for the model
        input_shape = utils.get_input_shape(preprocessed_data_dir)
        print('Input shape: {}'.format(input_shape))

        # ------------------------------------------------------------------------------
        # Augmentation
        # ------------------------------------------------------------------------------

        # Generate all of the desired augmentations
        augment_utils.generate_augmentations(output_dir, splits, cfg, preprocessor,
                                    speed_augment, pitch_augment, noise_augment=0.0, mix_augment=0.0)

        removed_clicks_path = output_dir / 'augments' / 'no_clicks' / 'preprocessed' / 'spec'
        if (remove_clicks_aug > 0) and not os.path.exists(removed_clicks_path):
            os.makedirs(removed_clicks_path)
            detection_data_generator.remove_clicks(removed_clicks_path, splits, cfg)

        # ------------------------------------------------------------------------------
        # create label map
        # ------------------------------------------------------------------------------
        labels_list = ['non-whistle', 'whistle']
        label_map = {'non-whistle':0, 'whistle':1}

        # build dict for train & val: key=fpath, value=label
        for train_fp in splits['train']:
            if labels_list[0] in train_fp:
                file_id_mapping_train[train_fp] = 0
            else:
                file_id_mapping_train[train_fp] = 1

        for val_fp in splits['val']:
            if labels_list[0] in val_fp:
                file_id_mapping_val[val_fp] = 0
            else:
                file_id_mapping_val[val_fp] = 1

    # ------------------------------------------------------------------------------
    # Create or reload models
    # ------------------------------------------------------------------------------
    my_model = None
    reloaded_model = False

    if (os.path.exists(reload_weights)) and (cfg['paths']['reload_model_file_name'] != ''):
        reloaded_model = True
        model_json = open(model_json_path, 'r')
        loaded_model_json = model_json.read()
        model_json.close()

        # reload previously trained model weights
        my_model = model_from_json(loaded_model_json)
        my_model.load_weights(reload_weights)

        print('Loaded model from disk: {}'.format(reload_weights))

    else:
        if inference:
            print('ERROR: There is no model matching these parameters that ' \
                  'you previously trained, so there is no model to evaluate.')
            print('Either change the parameters in your config to match a previously trained model ')
            print('OR change inference to False to train a model with these new parameters.')
            sys.exit(0)

        backbone = None
        if model_name == 'MobileNetV2':
            backbone = MobileNetV2(weights='imagenet', include_top=False,
                                        input_shape=input_shape)
        elif model_name == 'ResNet50':
            backbone = ResNet50(weights='imagenet', include_top=False,
                                        input_shape=input_shape)
        elif model_name == 'ResNet152':
            backbone = ResNet152(weights='imagenet', include_top=False,
                                        input_shape=input_shape)
        else:
            print('ERROR: the only accepted model_name options are MobileNetV2, ResNet50, or ResNet152')
            sys.exit(0)

        x=backbone.output
        x=tf.keras.layers.GlobalAveragePooling2D()(x)
        preds=Dense(1, activation = 'sigmoid')(x)

        my_model = Model(inputs=backbone.input, outputs=preds)

        for layer in my_model.layers[:num_frozen_layers]:
            layer.trainable = False
        for layer in my_model.layers[num_frozen_layers:]:
            layer.trainable = True


        # serialize model to JSON (in case of needing to reload later)
        model_json = my_model.to_json()
        with open(model_json_path, 'w') as json_file:
            json_file.write(model_json)

    decay_steps = cfg['model']['detection_params']['decay_steps']
    decay_rate = cfg['model']['detection_params']['decay_rate']

    lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=LR_value,
                                                        decay_steps=decay_steps,
                                                        decay_rate=decay_rate)
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    # using binary focal loss to more heavily penalize 'hard' samples and focus on correcting those
    my_model.compile(loss=BinaryFocalCrossentropy(gamma=2.0), optimizer=optimizer, metrics=['acc'])


    # ------------------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------------------
    if not cfg['setup']['inference']:
        train_data_folder = preprocessed_data_dir / 'train'
        val_data_folder = preprocessed_data_dir / 'val'

        batch_size = cfg['model']['detection_params']['batch_size']
        model_type = cfg['model']['model_type']
        patience = cfg['model']['detection_params']['patience']

        train_generator = detection_data_generator.BatchGenerator(file_id_mapping_train,
                                                                  'train',
                                                                  preprocessed_data_dir,
                                                                  pitch_augment,
                                                                  speed_augment,
                                                                  batch_size,
                                                                  remove_clicks_aug)
        val_generator = detection_data_generator.BatchGenerator(file_id_mapping_val,
                                                                'val',
                                                                preprocessed_data_dir,
                                                                0.0,
                                                                0.0,
                                                                batch_size,
                                                                0.0)
        # Create callbacks for keras model
        mc = ModelCheckpoint(os.path.join(model_save_path,'weights_{epoch:08d}-{val_loss:.4f}.h5'),
                                            monitor='val_loss', save_weights_only=True, save_best_only=True,
                                            mode='min', save_freq='epoch')
        if reloaded_model:
            mc = ModelCheckpoint(os.path.join(model_save_path,'weights_contd_{epoch:08d}-{val_loss:.4f}.h5'),
                                              monitor='val_loss', save_weights_only=True, save_best_only=True,
                                              mode='min', save_freq='epoch')
        es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=False)
        callbacks = [mc, es]

        # only include wandb callback to track metrics if not in debug mode
        if not debug:
            wdb = WandbCallback(save_model=False, save_graph=False,
                                predictions=0)
            callbacks.append(wdb)

        # train a new model
        my_model.fit(x=train_generator,
                      validation_data=val_generator,
                      steps_per_epoch=train_generator.__len__(),
                      validation_steps=val_generator.__len__(),
                      validation_freq=1,
                      epochs=cfg['model']['detection_params']['num_epochs'],
                      callbacks=callbacks)
                      #use_multiprocessing=True)

        # save the trained model to checkpoint (saving both ckpt & h5 just in case one is needed over the other)
        simple_cnn_model_save_file = os.path.join(model_save_path, 'simple_cnn_final.ckpt')
        my_model.save_weights(simple_cnn_model_save_file.format(epoch=cfg['model']['detection_params']['num_epochs']))

        # save the trained model to h5 file
        save_model(my_model, os.path.join(model_save_path,'simple_cnn_weights_final.h5'))


    # ----------------------------------------------------------------------------------
    # Inference & evaluation
    # ----------------------------------------------------------------------------------
    if cfg['setup']['inference']:
        test_data_folder = cfg['paths']['audio_files_for_inference']
        sr = cfg['preprocess']['sampling_rate']
        spectrogram_max_length = cfg['preprocess']['spectrogram_max_length']
        confidence_threshold = cfg['model']['detection_params']['confidence_threshold']
        output_path = cfg['paths']['output_path']
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        labels = {}
        probabilities = {}
        segmented_spec_paths = []
        thresholds = [0.3, 0.5, 0.75]
        if confidence_threshold not in thresholds:
            thresholds.append(confidence_threshold)
        csv_folder_path = ''

        print('Detecting whistles in your data...')
        start_times = {}

        # if we already preprocessed (i.e. running inference on test or val) no need to preprocess again
        if 'preprocessed' in test_data_folder:
            csv_folder_path = os.path.join(output_path, 'csv_results')
            for c in os.listdir(test_data_folder):
                if c.startswith('.'): # ignore hidden files
                    continue
                class_path = os.path.join(test_data_folder, c)
                for spec_path in os.listdir(class_path):
                    if spec_path.lower().endswith('.png'):
                        full_spec_path = os.path.join(class_path, spec_path)
                        segmented_spec_paths.append(full_spec_path)
                        labels[full_spec_path] = c
                        # start time of each of these pre-clipped files will be 0
                        start_times[full_spec_path] = 0
        else:
            # sliding window along the input audio & generate spectrograms
            segmented_spec_paths, start_times = detection_data_generator.segment_audio_for_detection(test_data_folder, cfg)
            csv_folder_path = os.path.join(test_data_folder, 'csv_results')

        # create the grad cam object of this model
        cam = gradcam.GradCAM(my_model, classIdx=1)
        grad_cam_output_path = os.path.join(cfg['paths']['vis_output_path'], 'grad_cam')
        if not os.path.exists(grad_cam_output_path):
            os.makedirs(grad_cam_output_path)
            os.mkdir(os.path.join(grad_cam_output_path, 'predicted_whistles'))
            os.mkdir(os.path.join(grad_cam_output_path, 'predicted_non-whistles'))

        # run each spectrogram of audio through the model and gather predictions
        for spec_path in segmented_spec_paths:
            spec = cv2.imread(spec_path) / 255
            spec = np.expand_dims(spec, axis=0)
            spec = np.array(spec, dtype='float32')
            prob = my_model.predict(spec)[0][0]
            probabilities[spec_path] = prob

            # save grad cam visualizations
            if cfg['setup']['save_grad_cam'] == True:
                # generate heatmap for this image using gradcam
                heatmap = cam.compute_heatmap(spec)
                (heatmap, output) = cam.overlay_heatmap_detection(heatmap, spec)
                vis = np.concatenate((heatmap, output), axis=1)
                spec_name = os.path.basename(spec_path)

                if prob >= confidence_threshold:
                    vis_save_path = os.path.join(grad_cam_output_path, 'predicted_whistles',
                                                '{}_confidence_{}'.format(prob, spec_name))
                    cv2.imwrite(vis_save_path, vis)
                elif prob < confidence_threshold:
                    vis_save_path = os.path.join(grad_cam_output_path, 'predicted_non-whistles',
                                                '{}_confidence_{}'.format(prob, spec_name))
                    cv2.imwrite(vis_save_path, vis)

        if 'preprocessed' in test_data_folder:
            # save one CSV file of results for all the spectrograms in the val or test set
            detection_data_generator.write_to_csv_val_or_test(test_data_folder, csv_folder_path,
                                                                probabilities, sr, spectrogram_max_length,
                                                                confidence_threshold)
        else:
            # output CSV(s) of the results on the audio files in this test_data_folder
            detection_data_generator.write_to_csv(test_data_folder, csv_folder_path, start_times,
                                                    probabilities, sr, spectrogram_max_length,
                                                    confidence_threshold)

        # if we have labels, calculate metrics based on different thresholds for whistle/non-whistle
        if len(labels) > 0:
            for threshold in thresholds:
                acc = 0
                false_pos = 0
                false_neg = 0
                true_pos = 0
                true_neg = 0

                for spec_path in segmented_spec_paths:
                    if (probabilities[spec_path] >= threshold) and (labels[spec_path] == 'whistles'):
                        acc += 1
                        true_pos += 1
                    elif (probabilities[spec_path] < threshold) and labels[spec_path] == 'non-whistles':
                        acc += 1
                        true_neg += 1
                    elif (probabilities[spec_path] >= threshold) and (labels[spec_path] == 'non-whistles'):
                        false_pos += 1
                    elif (probabilities[spec_path] < threshold) and (labels[spec_path] == 'whistles'):
                        false_neg += 1

                acc = float(acc / len(labels))
                precision = float(true_pos / (true_pos + false_pos))
                recall = float(true_pos / (true_pos + false_neg))

                print('WHEN THRESHOLD FOR WHISTLE IS >= {}: '.format(threshold))
                print('Accuracy on this test set: {}%'.format(acc*100))
                print('Precision: {}\nRecall: {}'.format(precision*100, recall*100))
                print('True positives: {}, True Negatives: {}, False Positives: {}, False Negatives: {}'.format(true_pos, true_neg, false_pos, false_neg))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

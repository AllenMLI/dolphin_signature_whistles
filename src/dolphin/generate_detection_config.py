"""
Accept parameters specified by users and generate the config.json for running the pipeline
"""
import json


def generate_config(filename: str ="detection_config.json") -> None:
    """

    Args:
        filename: output filename, default as config.json

    Returns:
        Void and this function generates a json file in the same directory level
    """

    config = {
        "setup":{
            "debug": True, # if False, model will save results to wandb
            "inference": False, # if False, training will happen. If True, data will be run a trained model
            "save_grad_cam": True, # if True, grad cam viz will be saved when evaluate==True
        },
        "dataset": {
            "name": "sbln"
        },
        "paths": {
            "log_filename": "outputs_detection/dolphin.log",
            "data_path": "data/detection/test_SBLN_data",
            "output_path": "outputs_detection",
            "best_model_path": "models",
            "history_path": "history",
            "reload_model_file_name": "weights_00000002-0.3104.h5", #"weights_00000040-0.0704.h5", # can be left as empty "" if don't want to reload any weights
            "audio_files_for_inference": "data/detection/test_long_audio",
            "vis_output_path": "visualizations",
        },
        "preprocess": {
            "nfft": 1024,
            "window": "hamming",
            "sampling_rate": 60000,
            "contrast_percentile": 50,
            "spectrogram_max_length": 3,
            "melspec_fmin": 1000,
            "melspec_fmax": 20000,
            "melspec_hop_length": 512,
            "melspec_n_mels": 128,
            "features": "spec"
        },
        "augment":{
            # NOTE: sum(pitch_augment, speed_augment, remove_clicks) must be <= 1.0
            "pitch_augment": 0.25,
            "speed_augment": 0.25,
            "remove_clicks": 0.5
        },
        "model": {
            "model_name": "MobileNetV2",  # Options: ResNet50, ResNet152, MobileNetV2
            "model_type": "simple_cnn",  # Options: simple_cnn
            "detection_params": {
                "confidence_threshold":0.6,
                "learning_rate": 0.00001,
                "input_shape": None,
                "num_classes": 2,
                "num_epochs": 2,
                "batch_size": 16,
                "patience":50,
                "num_frozen_layers_resnet152":500,
                "num_frozen_layers_resnet50":170,  # there are 175 layers in resnet50
                "num_frozen_layers_mobilenetv2":150,  # there are 154 layers in mobilenetv2
                "decay_steps":100000,
                "decay_rate":0.96,
            },
        },
        "output": {
            "color_map": "YlGnBu_r",
            "inches_per_sec": 2,
            "inches_per_KHz": 0.1,
        }

    }

    with open(filename, 'w') as json_f:
        json.dump(config, json_f, indent=4)


if __name__ == "__main__":
    generate_config(filename="detection_config.json")

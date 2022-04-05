"""
Accept parameters specified by users and generate the config.json for running the pipeline
"""
import json


def generate_config(filename: str ="config.json") -> None:
    """
    Args:
        filename: output filename, default as config.json
    Returns:
        Void and this function generates a json file in the same directory level
    """

    config = {
        "setup":{
            "debug": True,
            "inference": False
        },
        "dataset": {
            "name": "sarasota"
        },
        "paths": {
            "data_path": "data/classification/train_Sarasota_Database",
            "output_path": "outputs",
            "checkpoint_path": "outputs/checkpoint",
            "best_model_path": "outputs/best_model",
            "inference_weights": "outputs/checkpoint/weights_test_classifier.h5", # use "None" if loading from latest
            "files_for_inference": None, #"data/classification/test_Sarasota_Database_images",
            "history_path": None,  # "history", or None if not saving metrics
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
            # NOTE: sum(pitch_augment, speed_augment, noise_augment, mix_augment) must be <= 1.0
            "pitch_augment": 0.0,
            "speed_augment": 0.0,
            "noise_augment": 0.0,
            "mix_augment": 0.0,
            "min_SNR": 0.0,
            "max_SNR": 30.0,
            "bg_audio_dir": "data/bg_audio"
        },
        "model": {
            "model_name": "mobilenetv2",
            "model_params": {
                "learning_rate": 0.0001,
                "input_shape": None,
                "num_classes": 70,
                "dropout": 0.5,
                "shuffle": True,
                "num_epochs": 2,
                "metrics": "acc",
                "batch_size": 4,
                "patience": 10,
                "checkpoint_step": 1
            }
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
    generate_config(filename="config.json")

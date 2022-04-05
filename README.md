# dolphin_whistles

## SETUP

### Get this Codebase onto Your Computer

Clone this repository
1. Log into your github account in the terminal.
2. `git clone https://github.com/AllenMLI/dolphin_whistles.git`

OR

Download the zip file of this repository
1. Click on the green Code button
2. Select Download Zip
3. Unzip the code directly under:
   * WINDOWS: `C:\Users\<YOUR-USERNAME>\dolphin_whistles`
   * MAC or LINUX: `/home/<YOUR-USERNAME>/dolphin_whistles`

### Install Anaconda (if not already installed)

Install Anaconda using the instructions for your operating system: https://docs.anaconda.com/anaconda/install
* NOTE: make sure to check the “add to PATH” box during installation
* If you get an error about “Failed to create Anaconda menus”:
    * If you have other versions of python installed, uninstall them
    * Turn off your antivirus software while installing
    * If you have Java Development Kit installed, uninstall that

Verify Anaconda install:
* Open Anaconda Powershell Prompt and run: 
    * `conda list`
        * If conda installed properly, you’ll see a list of installed packages and their versions
    * `python`
        * A python shell should open up and tell you your version of python
    * Type in `quit()` to exit python


### Conda Environment:

Open Anaconda Powershell Prompt and run these commands (type in “y” and hit enter/return each time it asks if you want to proceed) 
* NOTE: can’t use ^C/^V to copy/paste into Anaconda Prompt and right clicking also doesn’t seem to be an option, so need to type out each command

1) `conda create --name dolphin-env python=3.8`
2) `conda activate dolphin-env`
3) `conda install wandb`  ( If error occurs, try: `conda install -c conda-forge wandb`)
4) `conda install -c conda-forge opencv`
5) `conda install matplotlib`
6) `conda install -c conda-forge librosa`
7) `conda install git`
8) `pip install tensorflow`
9) `pip install opencv-python`

Optionally, you may also install using the environment.yml file
`conda env create -f environment.yml`. This will install a virtual environment named "dolphin-whistles".
### Getting Your Data in the Required Format

The main script of this codebase expects the user's data directory to live within the `dolphin_whistles/data/` directory.
Within `dolphin_whistles/data/` there is a `classification` folder and a `detection` folder.
Your data directory should be placed in the respective folder. And it should be structured as follows:

We are making the assumption that there is a folder for each class:

    1) For the Sarasota dataset, there should be a folder for each of F123, F345, and so on.

    2) For detection data, there should be a folder for whistles and non-whistles.

Here is an example of the directory tree for `data/`:
```
dolphin_whistles
|-- data
    |-- bg_audio
    |   |-- background_example1.wav
    |   |-- background_example2.wav
    |   `-- background_example3.wav
    |-- classification
    |   |-- Sarasota_Database
    |       |-- F123
    |       |   |-- 2017
    |       |   |  |--florida
    |       |   |       |--example1.wav
    |       |   `-- 2018
    |       |       |-- example1.wav
    |       |       `-- example2.wav
    |       |-- F234
    |       |   |-- example1.wav
    |       |   |-- example2.wav
    |       |   `-- example3.wav
    |       `-- F345
    |           |-- 2016
    |           |   |-- example1.wav
    |           |   |-- example2.wav
    |           `-- example1.wav
    `-- detection
        |-- SBLN_data
            |-- non-whistles
            |   |-- example1.wav
            |   |-- example2.wav
            |   `-- example3.wav
            `-- whistles
                |-- example1.wav
                |-- example2.wav
                |-- example3.wav
                `-- example4.wav
```

## Running Code

### Classification Model Training:
1) `cd` into dolphin_whistles, where `src`, `outputs`, `data`, and `tests` live.
2) Edit src/generate_classification_config.py to have all the parameters you want - making sure to set the following parameters to these values: 
   * Set inference to False
   * Set debug to True if you DO NOT want to log to wandb
3) Generate the configuration json file, by running the following:
   * Windows: `python src\dolphin\generate_classification_config.py`
   * Linux or Mac: `python src/dolphin/generate_classification_config.py`
4) This will generate a config.json, then run the pipeline:
   * Windows: `python src\dolphin\generate_classification_config.py`
   * Linux or Mac: `python src/dolphin/generate_classification_config.py`

### Classification Model Inference:
1) `cd` into dolphin_whistles, where `src`, `outputs`, `data`, and `tests` live.
2) Edit generate_classification_config.py to have all the parameters you want - making sure to set the following parameters to these values:
   * Set inference to True
   * Set debug to True (won't need to log to wandb for inference)
4) Generate the configuration json file, by running the following:
   * Windows: `python src\dolphin\generate_classification_config.py`
   * Linux or Mac: `python src/dolphin/generate_classification_config.py`
5) This will generate a config.json, then run the pipeline:
   * Windows: `python src\dolphin\generate_classification_config.py`
   * Linux or Mac: `python src/dolphin/generate_classification_config.py`

### Detection Model Training:
1) Edit src/generate_detection_config.py to have all the params you want - making sure to set the following parameters to these values: 
   * Set inference to False
   * Set debug to True if you DO NOT want to log to wandb
2) `cd dolphin_whistles`
3) Generate the configuration json file by running the following: 
   * Windows: `python src\dolphin\generate_detection_config.py`
   * Linux or Mac: `python src/dolphin/generate_detection_config.py`
5) This will generate a detection_config.json, then run the pipeline:
   * Windows: `python src\dolphin\detection_pipeline.py detection_config.json`
   * Linux or Mac: `python src/dolphin/detection_pipeline.py detection_config.json`

### Detection Model Inference:
1) Edit generate_detection_config.py to have all the params you want - making sure to set the following parameters to these values: 
   * Set inference to True
   * Set reload_model_file_name to be the name of the previously trained detector weights that you want to run inference on
   * Set audio_files_for_inference to the relative path of a folder containing audio files (of any length) that you want to run through the detector
   * The parameters under “augment” and “model” need to match the parameters of the previously trained model that you want to use to detect whistles in this new data
   * Set save_grad_cam to be True if you want to save an additional folder of visualizations of the model results
2) `cd dolphin_whistles`
3) Generate the configuration json file by running the following: 
   * Windows: `python src\dolphin\generate_detection_config.py`
   * Linux or Mac: `python src/dolphin/generate_detection_config.py`
5) This will generate a detection_config.json, then run the pipeline:
   * Windows: `python src\dolphin\detection_pipeline.py detection_config.json`
   * Linux or Mac: `python src/dolphin/detection_pipeline.py detection_config.json`


## Notes

If the augmentation flags are set to true, then the pipeline will apply augmentation to copies of the original data - this takes a while for a lot of data, but it will only occur the first time.The same for preprocessing the data - if the code finds that either of these directories already exist, it will not run that code again.

By running the either the classification or detection pipeline, it will automatically:
1) Split the data into train/val/test, preprocess said data, and save these images to [outputs | outputs_detection]/preprocessed
2) Apply augmentation (if speed_augment, mixture_augment, pitch_augment or remove_clicks are True in the config) and save those files plus their respective spectrograms to [outputs | outputs_detection]/augments/[speed_augments | mix_augments | pitch_augments | no_clicks]
3) Load in batches of data
4) Train the model on the batches of data

To backup your environment,

`conda env export > environment.yml`

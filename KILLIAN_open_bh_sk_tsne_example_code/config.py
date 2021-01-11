"""
Set up runtime configuration here.

*** Do not set up any variables in this file by hand. If you need to instantiate new variables,
    write them into the config.ini file, then parse with the ConfigParser ***


If you want to use this file to instantiate your config options (because, for example,
pathing may be easier), then leave the config file value blank but do not delete the key.

Reading the key/value pair from the config object requires you to first index the section then index the key.
    e.g.: input: config['SectionOfInterest']['KeyOfInterest'] -> output: 'value of interest'
Another way to od it is using the object method:
    e.g.: input: config.get('section', 'key') -> output: 'value of interest'

"""
### Imports
from ast import literal_eval
from pathlib import Path
from typing import List
import configparser
import numpy as np
import os
import pandas as pd
import random
import sys
import time


from bsoid import logging_bsoid


### Debug options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=1_000)


########################################################################################################################
### Set default variables

# Fetch the B-SOiD project directory regardless of clone location
BSOID_BASE_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Output directory to where you want the analysis to be stored
default_output_path = os.path.join(BSOID_BASE_PROJECT_PATH, 'output')
# Set runtime string for consistency
runtime_timestr: str = time.strftime("%Y-%m-%d_%HH%MM")
# Set loggers default vars
default_log_folder_path = Path(BSOID_BASE_PROJECT_PATH, 'logs').absolute()
default_log_file_name = 'default.log'
# set default config file name
config_file_name = 'config.ini'
valid_dlc_output_extensions = {'csv', 'h5'}
# Load up config file
configuration = configparser.ConfigParser()
configuration.read(os.path.join(BSOID_BASE_PROJECT_PATH, config_file_name))

# Default variables asserts


### PATH ################################################################################
DLC_PROJECT_PATH = configuration.get('PATH', 'DLC_PROJECT_PATH')
OUTPUT_PATH = config_output_path = configuration.get('PATH', 'OUTPUT_PATH').strip() if configuration.get('PATH', 'OUTPUT_PATH').strip() else default_output_path
VIDEO_OUTPUT_FOLDER_PATH = configuration.get('PATH', 'VIDEOS_OUTPUT_PATH') if configuration.get('PATH', 'VIDEOS_OUTPUT_PATH') else os.path.join(OUTPUT_PATH, 'videos')
GRAPH_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'graphs')
FRAMES_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'frames')
EXAMPLE_VIDEOS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'example_videos')

# PATH asserts
assert not DLC_PROJECT_PATH or os.path.isdir(DLC_PROJECT_PATH),\
    f'DLC_PROJECT_PATH SPECIFIED DOES NOT EXIST: {DLC_PROJECT_PATH}'
assert os.path.isdir(OUTPUT_PATH), f'SPECIFIED OUTPUT PATH INVALID/DOES NOT EXIST: {OUTPUT_PATH}'
assert os.path.isdir(VIDEO_OUTPUT_FOLDER_PATH), \
    f'`short_video_output_directory` dir. (value={VIDEO_OUTPUT_FOLDER_PATH}) must exist for runtime but does not.'


### APP #######################################################
MODEL_NAME = configuration.get('APP', 'OUTPUT_MODEL_NAME')  # Machine learning model name?
PIPELINE_NAME = configuration.get('APP', 'PIPELINE_NAME')
VIDEO_TO_LABEL_PATH: str = configuration.get('APP', 'VIDEO_TO_LABEL_PATH')  # Now, pick an example video that corresponds to one of the csv files from the PREDICT_FOLDERS  # TODO: ************* This note from the original author implies that VID_NAME must be a video that corresponds to a csv from PREDICT_FOLDERS
VIDEO_FPS: int = configuration.getint('APP', 'VIDEO_FRAME_RATE')
COMPILE_CSVS_FOR_TRAINING: int = configuration.getint('APP', 'COMPILE_CSVS_FOR_TRAINING')  # COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.  # TODO: low: remove? re-evaluate
PLOT_GRAPHS: bool = configuration.getboolean('APP', 'PLOT_GRAPHS')
SAVE_GRAPHS_TO_FILE: bool = configuration.getboolean('APP', 'SAVE_GRAPHS_TO_FILE')
GENERATE_VIDEOS: bool = configuration.getboolean('APP', 'GENERATE_VIDEOS')
FRAMES_OUTPUT_FORMAT: str = configuration.get('APP', 'FRAMES_OUTPUT_FORMAT')
DEFAULT_SAVED_GRAPH_FILE_FORMAT: str = configuration.get('APP', 'DEFAULT_SAVED_GRAPH_FILE_FORMAT')
PERCENT_FRAMES_TO_LABEL: float = configuration.getfloat('APP', 'PERCENT_FRAMES_TO_LABEL')
N_JOBS = configuration.getint('APP', 'N_JOBS')
IDENTIFICATION_ORDER: int = configuration.getint('APP', 'FILE_IDENTIFICATION_ORDER_LEGACY')  # TODO: low: deprecate

# App meta-variables which help standardize things
PIPELINE_FILENAME = f'bsoid_pipeline__{PIPELINE_NAME}.sav'
MODEL_FILENAME = f'bsoid_model__{MODEL_NAME}.sav'

# OUTPUT_VIDEO_FPS: an attempt at keeping output video fps consistent to input fps relative to PERCENT_FRAMES_TO_LABEL
OUTPUT_VIDEO_FPS = configuration.getint('APP', 'OUTPUT_VIDEO_FPS') \
    if configuration.get('APP', 'OUTPUT_VIDEO_FPS').isnumeric() \
    else int(VIDEO_FPS * PERCENT_FRAMES_TO_LABEL)


# APP asserts
assert not VIDEO_TO_LABEL_PATH or os.path.isfile(VIDEO_TO_LABEL_PATH), \
    f'Video does not exist: {VIDEO_TO_LABEL_PATH}. Check pathing in config.ini file.'
assert COMPILE_CSVS_FOR_TRAINING in {0, 1}, f'Invalid COMP value detected: {COMPILE_CSVS_FOR_TRAINING}.'
assert isinstance(PERCENT_FRAMES_TO_LABEL, float) and 0. < PERCENT_FRAMES_TO_LABEL < 1., \
    f'PERCENT_FRAMES_TO_LABEL is invalid. Value = {PERCENT_FRAMES_TO_LABEL}, type = {type(PERCENT_FRAMES_TO_LABEL)}.'
assert isinstance(IDENTIFICATION_ORDER, int), f'check IDENTIFICATION_ORDER for type validity'


### STREAMLIT ############################################################
default_pipeline_file_path = configuration.get('STREAMLIT', 'default_pipeline_location')

if default_pipeline_file_path:
    assert os.path.isfile(default_pipeline_file_path)


### DLC_FEATURES #########################################################
def get_part(part) -> str:
    """
    For some DLC projects, there are different naming conventions for body parts and their associated
    column names in the DLC output. This function resolves that name aliasing by actively
    checking the configuration file to find the true name that is expected for the given bodypart.
    Get the actual body part name as per the DLC data.
    """
    return configuration['DLC_FEATURES'][part]

# TODO: add literals?


### MODEL ###############################################################
RANDOM_STATE: int = configuration.getint('MODEL', 'RANDOM_STATE', fallback=random.randint(1, 100_000_000))
HOLDOUT_PERCENT: float = configuration.getfloat('MODEL', 'HOLDOUT_TEST_PCT')
CROSSVALIDATION_K: int = configuration.getint('MODEL', 'CROSS_VALIDATION_K')
CROSSVALIDATION_N_JOBS: int = configuration.getint('MODEL', 'CROSS_VALIDATION_N_JOBS')


### LOGGING ##########################################################

log_function_entry_exit: callable = logging_bsoid.log_entry_exit  # TODO: temporary measure to enable logging when entering/exiting functions. Times entry/exit for duration.

config_log_file_folder_path = configuration.get('LOGGING', 'LOG_FILE_FOLDER_PATH')
log_file_folder_path = config_log_file_folder_path if config_log_file_folder_path else default_log_folder_path

# Get logger variables
config_file_name = configuration.get('LOGGING', 'LOG_FILE_NAME', fallback=default_log_file_name)
logger_name = configuration.get('LOGGING', 'DEFAULT_LOGGER_NAME')
log_format = configuration.get('LOGGING', 'LOG_FORMAT', raw=True)
stdout_log_level = configuration.get('LOGGING', 'STREAM_LOG_LEVEL', fallback=None)
file_log_level = configuration.get('LOGGING', 'FILE_LOG_LEVEL', fallback=None)
log_file_file_path = str(Path(log_file_folder_path, config_file_name).absolute())


# Instantiate file-specific logger. Use at top of file as: "logger = bsoid.config.initialize_logger(__file__)"
initialize_logger: callable = logging_bsoid.preload_logger_with_config_vars(
    logger_name, log_format, stdout_log_level, file_log_level, log_file_file_path)

# Logging asserts
assert os.path.isdir(log_file_folder_path), f'Path does not exist: {log_file_folder_path}'


##### TESTING VARIABLES ################################################################################################

DEFAULT_CSV_TEST_FILE: str = os.path.join(
    BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', configuration.get('TESTING', 'DEFAULT_CSV_TEST_FILE'))
DEFAULT_H5_TEST_FILE: str = os.path.join(
    BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', configuration.get('TESTING', 'DEFAULT_H5_TEST_FILE'))

## TODO: low: address comments below
# try:
#     max_rows_to_read_in_from_csv = configuration.getint('TESTING', 'MAX_ROWS_TO_READ_IN_FROM_CSV')
# except ValueError:  # In the case that the value is empty (since it is optional), assign max possible size to read in
#     max_rows_to_read_in_from_csv = sys.maxsize
max_rows_to_read_in_from_csv: int = configuration.getint('TESTING', 'max_rows_to_read_in_from_csv') \
    if configuration.get('TESTING', 'max_rows_to_read_in_from_csv') else sys.maxsize  # TODO: potentially remove this variable. When comparing pd.read_csv and bsoid.read_csv, they dont match due to header probs


assert os.path.isfile(DEFAULT_CSV_TEST_FILE), f'CSV test file was not found: {DEFAULT_CSV_TEST_FILE}'
# assert os.path.isfile(DEFAULT_H5_TEST_FILE), f'h5 test file was not found: {DEFAULT_H5_TEST_FILE}'  # TODO: low: when h5 format finally figured-out (From an actual DLC project outcome), re-instate this assert


### GMM PARAMS ########################################

gmm_n_components = configuration.getint('EM/GMM', 'n_components')
gmm_covariance_type = configuration.get('EM/GMM', 'covariance_type')
gmm_tol = configuration.getfloat('EM/GMM', 'tol')
gmm_reg_covar = configuration.getfloat('EM/GMM', 'reg_covar')
gmm_max_iter = configuration.getint('EM/GMM', 'max_iter')
gmm_n_init = configuration.getint('EM/GMM', 'n_init')
gmm_init_params = configuration.get('EM/GMM', 'init_params')
gmm_verbose = configuration.getint('EM/GMM', 'verbose')
gmm_verbose_interval = configuration.getint('EM/GMM', 'verbose_interval') \
    if configuration.get('EM/GMM', 'verbose_interval') else 10  # 10 is the default
EMGMM_PARAMS = {
    'n_components': gmm_n_components,
    'covariance_type': gmm_covariance_type,
    'tol': gmm_tol,
    'reg_covar': gmm_reg_covar,
    'max_iter': gmm_max_iter,
    'n_init': gmm_n_init,
    'init_params': gmm_init_params,
    'verbose': gmm_verbose,
    'random_state': RANDOM_STATE,
}

### HDBSCAN -- Density-based clustering ########################################
hdbscan_min_samples: int = configuration.getint('HDBSCAN', 'min_samples')
HDBSCAN_PARAMS = {
    'min_samples': hdbscan_min_samples,
}


### MLP -- Feedforward neural network (MLP) params ########################################
MLP_PARAMS = {
    'hidden_layer_sizes': literal_eval(configuration.get('MLP', 'hidden_layer_sizes')),
    'activation': configuration.get('MLP', 'activation'),
    'solver': configuration.get('MLP', 'solver'),
    'learning_rate': configuration.get('MLP', 'learning_rate'),
    'learning_rate_init': configuration.getfloat('MLP', 'learning_rate_init'),
    'alpha': configuration.getfloat('MLP', 'alpha'),
    'max_iter': configuration.getint('MLP', 'max_iter'),
    'early_stopping': configuration.getboolean('MLP', 'early_stopping'),
    'verbose': configuration.getint('MLP', 'verbose'),
}

### SVM ################################################################################
svm_c = configuration.getfloat('SVM', 'C')
svm_gamma = configuration.getfloat('SVM', 'gamma')
svm_probability = configuration.getboolean('SVM', 'probability')
svm_verbose = configuration.getint('SVM', 'verbose')
SVM_PARAMS = {
    'C': svm_c,
    'gamma': svm_gamma,
    'probability': svm_probability,
    'verbose': svm_verbose,
    'random_state': RANDOM_STATE,
}

### UMAP ################################################################################
UMAP_PARAMS = {
    'n_neighbors': configuration.getint('UMAP', 'n_neighbors'),
    'n_components': configuration.getint('UMAP', 'n_components'),
    'min_dist': configuration.getfloat('UMAP', 'min_dist'),
    'random_state': RANDOM_STATE,
}

### TSNE (BSOID VOC) ################################################################################
# TSNE parameters, can tweak if you are getting undersplit/oversplit behaviors
# the missing perplexity is scaled with data size (1% of data for nearest neighbors)

TSNE_EARLY_EXAGGERATION = configuration.getfloat('TSNE', 'early_exaggeration')
TSNE_N_COMPONENTS = configuration.getint('TSNE', 'n_components')
TSNE_N_ITER = configuration.getint('TSNE', 'n_iter')
TSNE_N_JOBS = configuration.getint('TSNE', 'n_jobs')
TSNE_THETA = configuration.getfloat('TSNE', 'theta')
TSNE_VERBOSE = configuration.getint('TSNE', 'verbose')

TSNE_SKLEARN_PARAMS = {  # TODO: low: deprecate? This was the old, opaque way of packing kwargs. Not used much anymore.
    'n_components': TSNE_N_COMPONENTS,
    'n_jobs': TSNE_N_JOBS,
    'verbose': TSNE_VERBOSE,
    'random_state': RANDOM_STATE,
    'n_iter': TSNE_N_ITER,
    'early_exaggeration': TSNE_EARLY_EXAGGERATION,
}

assert isinstance(TSNE_N_ITER, int) and TSNE_N_ITER >= 250, \
    f'TSNE_N_ITER should be an integer above 250 but was found to be: {TSNE_N_ITER}'

########################################################################################################################
# TODO: below under construction
##### TRAIN_FOLDERS, PREDICT_FOLDERS
# TRAIN_FOLDERS & PREDICT_FOLDERS are lists of folders that are implicitly understood to exist within BASE_PATH

TRAIN_DATA_FOLDER_PATH = os.path.abspath(configuration.get('PATH', 'TRAIN_DATA_FOLDER_PATH'))

PREDICT_DATA_FOLDER_PATH = configuration.get('PATH', 'PREDICT_DATA_FOLDER_PATH')


TRAIN_FOLDERS_IN_DLC_PROJECT_toBeDeprecated = [  # TODO: DEPREC
    'sample_train_data_folder',
]
PREDICT_FOLDERS_IN_DLC_PROJECT_toBeDeprecated: List[str] = [  # TODO: DEPREC
    'sample_predic_data_folder',
]

TRAIN_FOLDERS_PATHS_toBeDeprecated = [os.path.join(DLC_PROJECT_PATH, folder)
                                      for folder in TRAIN_FOLDERS_IN_DLC_PROJECT_toBeDeprecated
                                      if not os.path.isdir(folder)]  # TODO: why the if statement?
PREDICT_FOLDERS_PATHS_toBeDeprecated = [os.path.join(DLC_PROJECT_PATH, folder)
                                        for folder in PREDICT_FOLDERS_IN_DLC_PROJECT_toBeDeprecated]

### Create a folder to store extracted images.
config_value_alternate_output_path_for_annotated_frames = configuration.get(  # TODO:low:address.deleteable?duplicate?
    'PATH', 'ALTERNATE_OUTPUT_PATH_FOR_ANNOTATED_FRAMES')

FRAMES_OUTPUT_PATH = config_value_alternate_output_path_for_annotated_frames = \
    configuration.get('PATH', 'ALTERNATE_OUTPUT_PATH_FOR_ANNOTATED_FRAMES') \
    if configuration.get('PATH', 'ALTERNATE_OUTPUT_PATH_FOR_ANNOTATED_FRAMES') \
    else FRAMES_OUTPUT_PATH  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT/frames'


# Asserts  # TODO: delete or rework these asserts below for legacy variables

# for folder_path in TRAIN_FOLDERS_PATHS_toBeDeprecated:
#     assert os.path.isdir(folder_path), f'(ToBeDeprecated): TRAIN_FOLDERS_PATH: ' \
#                                        f'Training folder does not exist: {folder_path}'
#     assert os.path.isabs(folder_path), f'(ToBeDeprecated): TRAIN_FOLDERS_PATH: ' \
#                                        f'Predict folder PATH is not absolute and should be: {folder_path}'

# for folder_path in PREDICT_FOLDERS_PATHS_toBeDeprecated:
#     assert os.path.isdir(folder_path), f'(ToBeDeprecated): PREDICT_FOLDERS_PATH: ' \
#                                        f'Prediction folder does not exist: {folder_path}'
#     assert os.path.isabs(folder_path), f'(ToBeDeprecated): PREDICT_FOLDERS_PATH: ' \
#                                        f'Predict folder PATH is not absolute and should be: {folder_path}'

assert os.path.isabs(TRAIN_DATA_FOLDER_PATH), f'TODO, NOT AN ABS PATH review me! {__file__}'

assert os.path.isdir(config_value_alternate_output_path_for_annotated_frames), \
    f'config_value_alternate_output_path_for_annotated_frames does not exist. ' \
    f'config_value_alternate_output_path_for_annotated_frames = ' \
    f'\'{config_value_alternate_output_path_for_annotated_frames}\'. Check config.ini pathing.'


##### LEGACY VARIABLES #################################################################################################
# This version requires the six body parts Snout/Head, Forepaws/Shoulders, Hindpaws/Hips, Tailbase.
#   It appears as though the names correlate to the expected index of the feature when in Numpy array form.
#   (The body parts are numbered in their respective orders)
BODYPARTS_PY_LEGACY = {
    'Snout/Head': 0,
    'Neck': None,
    'Forepaw/Shoulder1': 1,
    'Forepaw/Shoulder2': 2,
    'Bodycenter': None,
    'Hindpaw/Hip1': 3,
    'Hindpaw/Hip2': 4,
    'Tailbase': 5,
    'Tailroot': None,
}

# BSOID VOC
# # original authors' note: Order the points that are encircling the mouth.
BODYPARTS_VOC_LEGACY = {
    'Point1': 0,
    'Point2': 1,
    'Point3': 2,
    'Point4': 3,
    'Point5': 4,
    'Point6': 5,
    'Point7': 6,
    'Point8': 7,
}


########################################################################################################################

bodyparts = {key: configuration['DLC_FEATURES'][key] for key in configuration['DLC_FEATURES']}


###

def get_config_str() -> str:
    """ Debugging function """
    config_string = ''
    for section in configuration.sections():
        config_string += f'SECTION: {section} // OPTIONS: {configuration.options(section)}\n'
    return config_string.strip()


# Below dict created according to BSOID/segmented_behaviours/README_legacy.md
map_group_to_behaviour = {
    0: 'UNKNOWN',
    1: 'orient right',
    2: 'body lick',
    3: 'rearing',
    4: 'nose poke',
    5: 'tall wall-rear',
    6: 'face groom',
    7: 'wall-rear',
    8: 'head groom',
    9: 'nose poke',
    10: 'pause',
    11: 'locomote',
    12: 'orient right',
    13: 'paw groom',
    14: 'locomote',
    15: 'orient left',
    16: 'orient left',
}

"""BSOID/segmented_behaviours/README_legacy.md
## Here are the example groups that we have extracted from multiple animals using B-SOiD

### Groups 1-4:
#### Group 1 (top left): Oreint right (+); Group 2 (top right): Body lick; 

![Mouse Action Groups 1-4](../examples/group1_4.gif)
#### Group3 (bottom left): Rearing; Group 4 (bottom right): Nose poke (+).

### Groups 5-8:
#### Group 5 (top left): Tall wall-rear; Group 6 (top right): Face groom; 
![Mouse Action Groups 5-8](../examples/group5_8.gif)
#### Group 7 (bottom left): Wall-rear; Group 8 (bottom right): Head groom.

### Groups 9-12:
#### Group 9 (top left): Nose poke (-); Group 10 (top right): Pause; 
![Mouse Action Groups 9-12](../examples/group9_12.gif)
#### Group 11 (bottom left): Locomote (+); Group 12 (bottom right): Orient right (-).

### Groups 13-16:
#### Group 13 (top left): Paw groom; Group 14 (top right): Locomote (-); 
![Mouse Action Groups 13-16](../examples/group13_16.gif)
#### Group 15 (bottom left): Orient left (+); Group 16 (bottom right): Orient left (-).

More example videos are in [this](../examples) directory .

"""


### Debugging efforts below. __main__ not integral to file, can delete later. ########################################

if __name__ == '__main__':
    # print(get_config_str())
    # print(f'bodyparts: {bodyparts}')
    # print()
    # print(f'max_rows_to_read_in_from_csv = {max_rows_to_read_in_from_csv}')
    # print(f'VIDEO_FPS = {VIDEO_FPS}')
    # print(f'runtime_timestr = {runtime_timestr}')
    # print(f'log_file_folder_path = {log_file_folder_path}')
    # print(type(RANDOM_STATE))
    # print(VIDEO_TO_LABEL_PATH)
    # print('OUTPUT_VIDEO_FPS', OUTPUT_VIDEO_FPS)
    pass

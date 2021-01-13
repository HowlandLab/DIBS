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
from ast import literal_eval
from pathlib import Path
from typing import Optional, Tuple
import configparser
import numpy as np
import os
import pandas as pd
import random
import sys
import time

from dibs import logging_enhanced


### Debug options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=1_000)


########################################################################################################################
### Set default variables

# Fetch the module directory regardless of clone location
DIBS_BASE_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Set the default output directory to where outcomes are stored
default_output_path = os.path.join(DIBS_BASE_PROJECT_PATH, 'output')
# Set runtime string to track when scripts are run
runtime_timestr: str = time.strftime("%Y-%m-%d_%HH%MM")
# Set logger default variables
default_log_folder_path = Path(DIBS_BASE_PROJECT_PATH, 'logs').absolute()
default_log_file_name = 'default.log'
# set default config file name from which this module will read. File assumed to be in base project directory.
config_file_name = 'config.ini'
#
valid_dlc_output_extensions = {'csv', 'h5', }
# Load up config file
configuration = configparser.ConfigParser()
configuration.read(os.path.join(DIBS_BASE_PROJECT_PATH, config_file_name))

# Default variables asserts
assert os.path.isdir(default_log_folder_path), f'log file save folder does not exist: {default_log_folder_path}'

### PATH ################################################################################
DEFAULT_TRAIN_DATA_DIR = configuration.get('PATH', 'DEFAULT_TRAIN_DATA_DIR')
if not os.path.isabs(DEFAULT_TRAIN_DATA_DIR):
    DEFAULT_TRAIN_DATA_DIR = os.path.join(DIBS_BASE_PROJECT_PATH, DEFAULT_TRAIN_DATA_DIR)
DEFAULT_TEST_DATA_DIR = configuration.get('PATH', 'DEFAULT_TEST_DATA_DIR')
if not os.path.isabs(DEFAULT_TEST_DATA_DIR):
    DEFAULT_TEST_DATA_DIR = os.path.join(DIBS_BASE_PROJECT_PATH, DEFAULT_TEST_DATA_DIR)
OUTPUT_PATH = config_output_path = configuration.get('PATH', 'OUTPUT_PATH').strip() \
    if configuration.get('PATH', 'OUTPUT_PATH').strip() \
    else default_output_path
VIDEO_OUTPUT_FOLDER_PATH = configuration.get('PATH', 'VIDEOS_OUTPUT_PATH', fallback=os.path.join(OUTPUT_PATH, 'videos'))
GRAPH_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'graphs')
FRAMES_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'frames')
EXAMPLE_VIDEOS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'example_videos')
VIDEO_TO_LABEL_PATH: str = configuration.get('PATH', 'VIDEO_TO_LABEL_PATH')  # Now, pick an example video that corresponds to one of the csv files from the PREDICT_FOLDERS  # TODO: evaluate if remove or not
DLC_PROJECT_PATH = configuration.get('PATH', 'DLC_PROJECT_PATH', fallback='')  # TODO: deprecate?
### PATH asserts
assert os.path.isdir(DEFAULT_TRAIN_DATA_DIR), f'Path could not be found: {DEFAULT_TRAIN_DATA_DIR}'
assert os.path.isdir(DEFAULT_TEST_DATA_DIR), f'Path could not be found: {DEFAULT_TEST_DATA_DIR}'
assert not DLC_PROJECT_PATH or os.path.isdir(DLC_PROJECT_PATH),\
    f'DLC_PROJECT_PATH SPECIFIED DOES NOT EXIST: {DLC_PROJECT_PATH}'
assert os.path.isdir(OUTPUT_PATH), f'SPECIFIED OUTPUT PATH INVALID/DOES NOT EXIST: {OUTPUT_PATH}'
assert os.path.isdir(VIDEO_OUTPUT_FOLDER_PATH), \
    f'`short_video_output_directory` dir. (value={VIDEO_OUTPUT_FOLDER_PATH}) must exist for runtime but does not.'
assert not VIDEO_TO_LABEL_PATH or os.path.isfile(VIDEO_TO_LABEL_PATH), \
    f'Video does not exist: {VIDEO_TO_LABEL_PATH}. Check pathing in config.ini file.'

### APP #######################################################
MODEL_NAME = configuration.get('APP', 'OUTPUT_MODEL_NAME', fallback='DEFAULT_OUTPUT_MODEL_NAME__TODO:DEPRECATE?')  # Machine learning model name?
PIPELINE_NAME = configuration.get('APP', 'PIPELINE_NAME')
VIDEO_FPS: float = configuration.getfloat('APP', 'VIDEO_FRAME_RATE')
PLOT_GRAPHS: bool = configuration.getboolean('APP', 'PLOT_GRAPHS')
SAVE_GRAPHS_TO_FILE: bool = configuration.getboolean('APP', 'SAVE_GRAPHS_TO_FILE')
FRAMES_OUTPUT_FORMAT: str = configuration.get('APP', 'FRAMES_OUTPUT_FORMAT')
DEFAULT_SAVED_GRAPH_FILE_FORMAT: str = configuration.get('APP', 'DEFAULT_SAVED_GRAPH_FILE_FORMAT')
PERCENT_FRAMES_TO_LABEL: float = configuration.getfloat('APP', 'PERCENT_FRAMES_TO_LABEL')
N_JOBS = configuration.getint('APP', 'N_JOBS')

if 'NUMEXPR_MAX_THREADS' not in os.environ and configuration.get('APP', 'NUMEXPR_MAX_THREADS'):
    NUMEXPR_MAX_THREADS = configuration.getint('APP', 'NUMEXPR_MAX_THREADS')
    assert NUMEXPR_MAX_THREADS > 0, f'NUMEXPR_MAX_THREADS must be an integer greater than 0'
    os.environ['NUMEXPR_MAX_THREADS'] = str(NUMEXPR_MAX_THREADS)

IDENTIFICATION_ORDER: int = configuration.getint('APP', 'FILE_IDENTIFICATION_ORDER_LEGACY')  # TODO: low: deprecate
COMPILE_CSVS_FOR_TRAINING: int = configuration.getint('LEGACY', 'COMPILE_CSVS_FOR_TRAINING')  # COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.  # TODO: low: remove? re-evaluate

OUTPUT_VIDEO_FPS = configuration.getint('APP', 'OUTPUT_VIDEO_FPS')


### APP asserts
assert COMPILE_CSVS_FOR_TRAINING in {0, 1}, f'Invalid COMP value detected: {COMPILE_CSVS_FOR_TRAINING}.'
assert isinstance(PERCENT_FRAMES_TO_LABEL, float) and 0. < PERCENT_FRAMES_TO_LABEL < 1., \
    f'PERCENT_FRAMES_TO_LABEL is invalid. Value = {PERCENT_FRAMES_TO_LABEL}, type = {type(PERCENT_FRAMES_TO_LABEL)}.'
assert isinstance(IDENTIFICATION_ORDER, int), f'check IDENTIFICATION_ORDER for type validity'
assert isinstance(N_JOBS, int) and N_JOBS > 0, f'N_JOBS is invalid. Value = `{N_JOBS}`'

### STREAMLIT ############################################################
default_pipeline_file_path = configuration.get('STREAMLIT', 'default_pipeline_location')

### STREAMLIT asserts
if default_pipeline_file_path:
    assert os.path.isfile(default_pipeline_file_path), \
        f'Pipeline location could not be found: {default_pipeline_file_path}'


### MODEL ###############################################################
DEFAULT_CLASSIFIER: str = configuration.get('MODEL', 'DEFAULT_CLASSIFIER')
RANDOM_STATE: int = configuration.getint('MODEL', 'RANDOM_STATE', fallback=random.randint(1, 100_000_000))
HOLDOUT_PERCENT: float = configuration.getfloat('MODEL', 'HOLDOUT_TEST_PCT')
CROSSVALIDATION_K: int = configuration.getint('MODEL', 'CROSS_VALIDATION_K')
CROSSVALIDATION_N_JOBS: int = configuration.getint('MODEL', 'CROSS_VALIDATION_N_JOBS')

valid_classifiers = {'SVM', 'RANDOMFOREST'}

### MODEL ASSERTS
assert DEFAULT_CLASSIFIER in valid_classifiers, f'An invalid classifer was detected: "{DEFAULT_CLASSIFIER}". ' \
                                                f'Valid classifier values include: {valid_classifiers}'


### LOGGING ##########################################################

log_function_entry_exit: callable = logging_enhanced.log_entry_exit  # Temporary measure to enable logging when entering/exiting functions. Times entry/exit for duration and logs it.

config_log_file_folder_path = configuration.get('LOGGING', 'LOG_FILE_FOLDER_PATH')
log_file_folder_path = config_log_file_folder_path if config_log_file_folder_path else default_log_folder_path

# Get logger variables
config_file_name = configuration.get('LOGGING', 'LOG_FILE_NAME', fallback=default_log_file_name)
logger_name = configuration.get('LOGGING', 'DEFAULT_LOGGER_NAME')
log_format = configuration.get('LOGGING', 'LOG_FORMAT', raw=True)
stdout_log_level = configuration.get('LOGGING', 'STREAM_LOG_LEVEL', fallback=None)
file_log_level = configuration.get('LOGGING', 'FILE_LOG_LEVEL', fallback=None)
log_file_file_path = str(Path(log_file_folder_path, config_file_name).absolute())


# Instantiate file-specific logger. Use at top of file as: "logger = dibs.config.initialize_logger(__file__)"
initialize_logger: callable = logging_enhanced.preload_logger_with_config_vars(
    logger_name, log_format, stdout_log_level, file_log_level, log_file_file_path)

### Logging asserts
assert os.path.isdir(log_file_folder_path), f'Path does not exist: {log_file_folder_path}'


##### TESTING VARIABLES ################################################################################################

# Prime data files
DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE: str = configuration.get('TESTING', 'DEFAULT_PIPELINE_PRIME_CSV_TEST_FILE')
DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH = os.path.join(DIBS_BASE_PROJECT_PATH, 'tests', 'test_data', DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE)

# Mimic data files
TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE: str = configuration.get('TESTING', 'TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE')
TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH = os.path.join(DIBS_BASE_PROJECT_PATH, 'tests', 'test_data', TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE)
TEST_FILE__PipelineMimic__CSV__TEST_DATA_FILE: str = configuration.get('TESTING', 'TEST_FILE__PipelineMimic__CSV__TEST_DATA_FILE')
TEST_FILE__PipelineMimic__CSV__TEST_DATA_FILE_PATH = os.path.join(DIBS_BASE_PROJECT_PATH, 'tests', 'test_data', TEST_FILE__PipelineMimic__CSV__TEST_DATA_FILE)


# DEFAULT_PIPELINE__MIMIC__CSV_TEST_FILE_PATH = os.path.join()

# DEFAULT_PIPELINE__CHBO__CSV_TEST_FILE: str = os.path.join()
# DEFAULT_H5_TEST_FILE: str = os.path.join(DIBS_BASE_PROJECT_PATH, 'tests', 'test_data', configuration.get('TESTING', 'DEFAULT_H5_TEST_FILE'))


## TODO: low: address comments below
# try:
#     max_rows_to_read_in_from_csv = configuration.getint('TESTING', 'MAX_ROWS_TO_READ_IN_FROM_CSV')
# except ValueError:  # In the case that the value is empty (since it is optional), assign max possible size to read in
#     max_rows_to_read_in_from_csv = sys.maxsize

max_rows_to_read_in_from_csv: int = configuration.getint('TESTING', 'max_rows_to_read_in_from_csv') \
    if configuration.get('TESTING', 'max_rows_to_read_in_from_csv') \
    else sys.maxsize  # TODO: potentially remove this variable. When comparing pd.read_csv and dibs.read_csv, they dont match due to header probs

### Testing variables asserts

assert os.path.isfile(DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH), f'CSV test file was not found: {DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH}'
assert os.path.isfile(TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH), f'CSV test file was not found: {TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH}'
assert os.path.isfile(TEST_FILE__PipelineMimic__CSV__TEST_DATA_FILE_PATH), f'CSV test file not found: {TEST_FILE__PipelineMimic__CSV__TEST_DATA_FILE_PATH}'
# assert os.path.isfile(DEFAULT_H5_TEST_FILE), f'h5 test file was not found: {DEFAULT_H5_TEST_FILE}'  # TODO: low: when h5 format finally figured-out (From an actual DLC project outcome), re-instate this assert


### GMM PARAMS #########################################################################################################

gmm_n_components = configuration.getint('EM/GMM', 'n_components')
gmm_covariance_type = configuration.get('EM/GMM', 'covariance_type')
gmm_tol = configuration.getfloat('EM/GMM', 'tol')
gmm_reg_covar = configuration.getfloat('EM/GMM', 'reg_covar')
gmm_max_iter = configuration.getint('EM/GMM', 'max_iter')
gmm_n_init = configuration.getint('EM/GMM', 'n_init')
gmm_init_params = configuration.get('EM/GMM', 'init_params')
gmm_verbose = configuration.getint('EM/GMM', 'verbose')
gmm_verbose_interval = configuration.getint('EM/GMM', 'verbose_interval') \
    if configuration.get('EM/GMM', 'verbose_interval') \
    else 10  # 10 is a default that can be changed  # TODO: low: address
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

### HDBSCAN -- Density-based clustering ################################################################################
hdbscan_min_samples: int = configuration.getint('HDBSCAN', 'min_samples')


### MLP -- Feedforward neural network (MLP) params #####################################################################
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


### RANDOM FOREST ###
rf_n_estimators = configuration.getint('RANDOMFOREST', 'n_estimators')


### SVM ################################################################################################################
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


### TSNE ################################################################################
# TSNE parameters, can tweak if you are getting undersplit/oversplit behaviors
# the missing perplexity is scaled with data size (1% of data for nearest neighbors)

TSNE_EARLY_EXAGGERATION: float = configuration.getfloat('TSNE', 'early_exaggeration')
TSNE_IMPLEMENTATION: str = configuration.get('TSNE', 'implementation')
TSNE_INIT: str = configuration.get('TSNE', 'init')
TSNE_LEARNING_RATE: float = configuration.getfloat('TSNE', 'learning_rate')
TSNE_N_COMPONENTS: int = configuration.getint('TSNE', 'n_components')
TSNE_N_ITER: int = configuration.getint('TSNE', 'n_iter')
TSNE_N_JOBS: int = configuration.getint('TSNE', 'n_jobs')
TSNE_PERPLEXITY: Optional[str] = configuration.getfloat('TSNE', 'perplexity')
# TSNE_PERPLEXITY: Optional[float] = float(TSNE_PERPLEXITY) if TSNE_PERPLEXITY else -1.0
TSNE_THETA: float = configuration.getfloat('TSNE', 'theta')
TSNE_VERBOSE: int = configuration.getint('TSNE', 'verbose')

### TSNE asserts
valid_tsne_initializations = {'random', 'pca'}
valid_tsne_implementations = {'SKLEARN', 'BHTSNE', 'OPENTSNE'}
minimum_tsne_n_iter = 250
assert TSNE_INIT in valid_tsne_initializations, f'TSNE INIT parameters was not valid.' \
                                                f'Parameter is currently: {TSNE_INIT}.'
assert TSNE_IMPLEMENTATION in valid_tsne_implementations, f''
assert isinstance(TSNE_N_ITER, int) and TSNE_N_ITER >= minimum_tsne_n_iter, \
    f'TSNE_N_ITER should be an integer above {minimum_tsne_n_iter} but was found ' \
    f'to be: {TSNE_N_ITER} (type: {type(TSNE_N_ITER)})'

### UMAP ################################################################################
UMAP_PARAMS = {
    'n_neighbors': configuration.getint('UMAP', 'n_neighbors'),
    'n_components': configuration.getint('UMAP', 'n_components'),
    'min_dist': configuration.getfloat('UMAP', 'min_dist'),
    'random_state': RANDOM_STATE,
}


###### VIDEO PARAMETERS #####
DEFAULT_FONT_SCALE: int = configuration.getint('VIDEO', 'DEFAULT_FONT_SCALE')
DEFAULT_TEXT_BGR: Tuple[int] = literal_eval(configuration.get('VIDEO', 'DEFAULT_TEXT_BGR'))
DEFAULT_TEXT_BACKGROUND_BGR: Tuple[int] = literal_eval(configuration.get('VIDEO', 'DEFAULT_TEXT_BACKGROUND_BGR'))

map_ext_to_fourcc = {
    'mp4': 'mp4v',
    'avi': 'MJPG',

}

### VIDEO asserts
assert isinstance(DEFAULT_TEXT_BGR, tuple), f'DEFAULT_TEXT_BGR was expected to be a tuple but ' \
                                            f'instead found type: {type(DEFAULT_TEXT_BGR)}  (value = {DEFAULT_TEXT_BGR}'
assert len(DEFAULT_TEXT_BGR) == 3, f'DEFAULT_TEXT_BGR was expected to have 3 elements but ' \
                                   f'instead found: {len(DEFAULT_TEXT_BGR)}'

assert isinstance(DEFAULT_TEXT_BACKGROUND_BGR, tuple), f'DEFAULT_TEXT_BACKGROUND_BGR was expected to be a tuple but ' \
                                            f'instead found type: {type(DEFAULT_TEXT_BACKGROUND_BGR)}  ' \
                                                       f'(value = {DEFAULT_TEXT_BACKGROUND_BGR}'
assert len(DEFAULT_TEXT_BACKGROUND_BGR) == 3, f'DEFAULT_TEXT_BACKGROUND_BGR was expected to have 3 elements but ' \
                                   f'instead found: {len(DEFAULT_TEXT_BACKGROUND_BGR)}'


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

bodyparts = {key: configuration['DLC_FEATURES'][key]
             for key in configuration['DLC_FEATURES']}

###


### LEGACY ###

# App meta-variables which help standardize things
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


### CONFIG FUNCTIONS ###

def get_part(part) -> str:
    """
    For some DLC projects, there are different naming conventions for body parts and their associated
    column names in the DLC output. This function resolves that name aliasing by actively
    checking the configuration file to find the true name that is expected for the given bodypart.
    Get the actual body part name as per the DLC data.
    """
    return configuration['DLC_FEATURES'][part]


def get_config_str() -> str:
    """ Debugging function """
    config_string = ''
    for section in configuration.sections():
        config_string += f'SECTION: {section} // OPTIONS: {configuration.options(section)}\n'
    return config_string.strip()


def get_data_source_from_file_path(file_path: str):
    file_folder, file_name = os.path.split(file_path)
    file_name_without_extension, extension = file_name.split('.')
    return file_name_without_extension


### Debugging efforts below. __main__ not integral to file. Use __main__ to check in on config vars.


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
    # print(f'DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH = {DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH}')
    # print(DEFAULT_TRAIN_DATA_DIR)
    pass

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
from pathlib import Path
from types import FunctionType
from typing import Optional, Tuple, Union
import configparser
import numpy as np
import os
import pandas as pd
import random
import sys
import time

from dibs import logging_enhanced


### Debug options
pd.set_option('display.max_rows', 1_000)
pd.set_option('display.max_columns', 1_000)
pd.set_option('display.width', 1_000)
pd.set_option('display.max_colwidth', 1_000)
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
VIDEO_INPUT_FOLDER_PATH = configuration.get('PATH', 'VIDEOS_INPUT_PATH').strip() or os.path.join(DIBS_BASE_PROJECT_PATH, 'input_videos')
VIDEO_OUTPUT_FOLDER_PATH = configuration.get('PATH', 'VIDEOS_OUTPUT_PATH').strip() or os.path.join(OUTPUT_PATH, 'videos')
GRAPH_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'graphs')
FRAMES_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'frames')
EXAMPLE_VIDEOS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'example_videos')
PIPELINE_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'pipelines')

### PATH asserts
for path in (
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_TRAIN_DATA_DIR,
    OUTPUT_PATH,
    VIDEO_OUTPUT_FOLDER_PATH,
    VIDEO_INPUT_FOLDER_PATH,
    GRAPH_OUTPUT_PATH,
    FRAMES_OUTPUT_PATH,
    EXAMPLE_VIDEOS_OUTPUT_PATH,
    PIPELINE_OUTPUT_PATH
):
    os.path.isdir(path) or os.makedirs(path) # Will fail if a path is provided that is not a directory for any of these arguments.

assert os.path.isdir(OUTPUT_PATH), f'SPECIFIED OUTPUT PATH INVALID/DOES NOT EXIST: {OUTPUT_PATH}'
assert os.path.isdir(VIDEO_OUTPUT_FOLDER_PATH), \
    f'`short_video_output_directory` dir. (value={VIDEO_OUTPUT_FOLDER_PATH}) must exist for runtime but does not.'


### APP #######################################################

FRAMES_OUTPUT_FORMAT: str = configuration.get('APP', 'FRAMES_OUTPUT_FORMAT')  # E.g. png, jpg, svg, etc.
N_JOBS = configuration.getint('APP', 'N_JOBS')  # TODO: low: currently not being used
MODEL_NAME = configuration.get('APP', 'OUTPUT_MODEL_NAME', fallback='DEFAULT_OUTPUT_MODEL_NAME__TODO:DEPRECATE?')  # Machine learning model name?
PLOT_GRAPHS: bool = configuration.getboolean('APP', 'PLOT_GRAPHS')
VIDEO_FPS: float = configuration.getfloat('APP', 'VIDEO_FRAME_RATE')
SAVE_GRAPHS_TO_FILE: bool = configuration.getboolean('APP', 'SAVE_GRAPHS_TO_FILE')
DEFAULT_SAVED_GRAPH_FILE_FORMAT: str = configuration.get('APP', 'DEFAULT_SAVED_GRAPH_FILE_FORMAT')
OUTPUT_VIDEO_FPS = configuration.getint('APP', 'OUTPUT_VIDEO_FPS')
if 'NUMEXPR_MAX_THREADS' not in os.environ and configuration.get('APP', 'NUMEXPR_MAX_THREADS'):
    NUMEXPR_MAX_THREADS = configuration.getint('APP', 'NUMEXPR_MAX_THREADS')
    assert NUMEXPR_MAX_THREADS > 0, f'NUMEXPR_MAX_THREADS must be an integer greater than 0'
    os.environ['NUMEXPR_MAX_THREADS'] = str(NUMEXPR_MAX_THREADS)

### APP asserts
# assert isinstance(N_JOBS, int) and N_JOBS > 0, f'N_JOBS is invalid. Value = `{N_JOBS}`'


### STREAMLIT ############################################################
default_pipeline_file_path = configuration.get('STREAMLIT', 'default_pipeline_location', fallback='')
STREAMLIT_DEFAULT_VIDEOS_FOLDER = configuration.get('STREAMLIT', 'STREAMLIT_DEFAULT_VIDEOS_FOLDER')

### STREAMLIT asserts
if default_pipeline_file_path:
    assert os.path.isfile(default_pipeline_file_path), f'Pipeline location could not be found: {default_pipeline_file_path}'
if STREAMLIT_DEFAULT_VIDEOS_FOLDER:
    assert os.path.isdir(STREAMLIT_DEFAULT_VIDEOS_FOLDER), f'Streamlit config `DEFAULT_VIDEOS_FOLDER` Folder missing: {STREAMLIT_DEFAULT_VIDEOS_FOLDER}'
    assert os.path.isabs(STREAMLIT_DEFAULT_VIDEOS_FOLDER), f'Streamlit config `DEFAULT_VIDEOS_FOLDER` Path is not absolute: {STREAMLIT_DEFAULT_VIDEOS_FOLDER}'

### MODEL ###############################################################
AVERAGE_OVER_N_FRAMES: int = configuration.getint('MODEL', 'AVERAGE_OVER_N_FRAMES')
CROSSVALIDATION_K: int = configuration.getint('MODEL', 'CROSS_VALIDATION_K')
CROSSVALIDATION_N_JOBS: int = configuration.getint('MODEL', 'CROSS_VALIDATION_N_JOBS')
HOLDOUT_PERCENT: float = configuration.getfloat('MODEL', 'HOLDOUT_TEST_PCT')
RANDOM_STATE: int = configuration.getint('MODEL', 'RANDOM_STATE') if configuration.get('MODEL', 'RANDOM_STATE') else random.randint(1, 100_000_000)

### MODEL ASSERTS
assert CROSSVALIDATION_K >= 2, f'CROSSVALIDATION_K must be 2 or greater. Instead, found: {CROSSVALIDATION_K}'
assert CROSSVALIDATION_N_JOBS >= -2, f'CROSSVALIDATION_N_JOBS must be -2 or greater. Instead, found: {CROSSVALIDATION_N_JOBS}'
assert 0. <= HOLDOUT_PERCENT <= 1., f'HOLDOUT_PERCENT must be between 0 and 1. Instead, found: {HOLDOUT_PERCENT}'


### LOGGING ##########################################################
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


log_function_entry_exit: callable = logging_enhanced.log_entry_exit  # Temporary measure to enable logging when entering/exiting functions. Times entry/exit for duration and logs it.

### Logging asserts
assert os.path.isdir(log_file_folder_path), f'Path does not exist: {log_file_folder_path}'


##### TESTING VARIABLES ################################################################################################

# Prime data files
DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE: str = configuration.get('TESTING', 'DEFAULT_PIPELINE_PRIME_CSV_TEST_FILE')
DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH = os.path.join(DIBS_BASE_PROJECT_PATH, 'tests', 'test_data', DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE)

# Mimic data files
TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE: str = configuration.get('TESTING', 'TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE')
TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH = os.path.join(DIBS_BASE_PROJECT_PATH, 'tests', 'test_data', TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE)
TEST_FILE__PipelineMimic__CSV__PREDICT_DATA_FILE: str = configuration.get('TESTING', 'TEST_FILE__PipelineMimic__CSV__PREDICT_DATA_FILE')
TEST_FILE__PipelineMimic__CSV__PREDICT_DATA_FILE_PATH = os.path.join(DIBS_BASE_PROJECT_PATH, 'tests', 'test_data', TEST_FILE__PipelineMimic__CSV__PREDICT_DATA_FILE)

# Howland data files
TEST_FILE__PipelineHowland__CSV__TRAIN_DATA_FILE: str = configuration.get('TESTING', 'TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE')
TEST_FILE__PipelineHowland__CSV__TRAIN_DATA_FILE_PATH = os.path.join(DIBS_BASE_PROJECT_PATH, 'tests', 'test_data', TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE)
TEST_FILE__PipelineHowland__CSV__PREDICT_DATA_FILE: str = configuration.get('TESTING', 'TEST_FILE__PipelineMimic__CSV__PREDICT_DATA_FILE')
TEST_FILE__PipelineHowland__CSV__PREDICT_DATA_FILE_PATH = os.path.join(DIBS_BASE_PROJECT_PATH, 'tests', 'test_data', TEST_FILE__PipelineMimic__CSV__PREDICT_DATA_FILE)

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
assert os.path.isfile(TEST_FILE__PipelineMimic__CSV__PREDICT_DATA_FILE_PATH), f'CSV test file not found: {TEST_FILE__PipelineMimic__CSV__PREDICT_DATA_FILE_PATH}'
# assert os.path.isfile(DEFAULT_H5_TEST_FILE), f'h5 test file was not found: {DEFAULT_H5_TEST_FILE}'  # TODO: low: when h5 format finally figured-out (From an actual DLC project outcome), re-instate this assert


### GENERAL CLASSIFIER VARIABLES ###
DEFAULT_CLASSIFIER: str = configuration.get('CLASSIFIER', 'DEFAULT_CLASSIFIER')
CLASSIFIER_VERBOSE: int = configuration.getint('CLASSIFIER', 'VERBOSE')

### Classifier asserts
valid_classifiers = {'SVM', 'RANDOMFOREST'}

assert DEFAULT_CLASSIFIER in valid_classifiers, f'An invalid classifer was detected: "{DEFAULT_CLASSIFIER}". ' \
                                                f'Valid classifier values include: {valid_classifiers}'
# assert CLASSIFIER_N_JOBS
assert CLASSIFIER_VERBOSE >= 0, f'Invalid verbosity integer submitted. CLASSIFIER_VERBOSE value = {CLASSIFIER_VERBOSE}'


### GMM PARAMS #########################################################################################################

gmm_n_components = configuration.getint('GMM', 'n_components')
gmm_covariance_type = configuration.get('GMM', 'covariance_type')
gmm_tol = configuration.getfloat('GMM', 'tol')
gmm_reg_covar = configuration.getfloat('GMM', 'reg_covar')
gmm_max_iter = configuration.getint('GMM', 'max_iter')
gmm_n_init = configuration.getint('GMM', 'n_init')
gmm_init_params = configuration.get('GMM', 'init_params')
gmm_verbose = configuration.getint('GMM', 'verbose')
gmm_verbose_interval = configuration.getint('GMM', 'verbose_interval') if configuration.get('GMM', 'verbose_interval') else 10  # 10 is a default that can be changed  # TODO: low: address
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
    'hidden_layer_sizes': eval(configuration.get('MLP', 'hidden_layer_sizes')),
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
rf_n_jobs = configuration.getint('RANDOMFOREST', 'n_jobs')
rf_verbose = configuration.getint('RANDOMFOREST', 'verbose')

assert rf_n_estimators > 0, f''
assert rf_n_jobs > 0, f''
assert rf_verbose >= 0, f''

### SVM ################################################################################################################
svm_c = configuration.getfloat('SVM', 'C')
svm_gamma = configuration.getfloat('SVM', 'gamma')
svm_probability = configuration.getboolean('SVM', 'probability')
svm_verbose = configuration.getint('SVM', 'verbose')


### TSNE ################################################################################
# TSNE parameters, can tweak if you are getting undersplit/oversplit behaviors
#   the missing perplexity is scaled with data size (1% of data for nearest neighbors)

TSNE_EARLY_EXAGGERATION: float = configuration.getfloat('TSNE', 'early_exaggeration')
TSNE_IMPLEMENTATION: str = configuration.get('TSNE', 'implementation')
TSNE_INIT: str = configuration.get('TSNE', 'init')
TSNE_LEARNING_RATE: float = configuration.getfloat('TSNE', 'learning_rate')
TSNE_N_COMPONENTS: int = configuration.getint('TSNE', 'n_components')
TSNE_N_ITER: int = configuration.getint('TSNE', 'n_iter')
TSNE_N_JOBS: int = configuration.getint('TSNE', 'n_jobs')
TSNE_PERPLEXITY: Union[str, float] = configuration.get('TSNE', 'perplexity')
try:
    TSNE_PERPLEXITY = float(TSNE_PERPLEXITY)
except ValueError:
    pass
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
# assert isinstance(TSNE_PERPLEXITY, float) \
#     or isinstance(TSNE_PERPLEXITY, int) \
#     or isinstance(TSNE_PERPLEXITY, str), \
#     f'INVALID TYPE FOR PERPLEXITY: {type(TSNE_PERPLEXITY)} (value: {TSNE_PERPLEXITY})'

### UMAP ################################################################################
UMAP_PARAMS = {
    'n_neighbors': configuration.getint('UMAP', 'n_neighbors'),
    'n_components': configuration.getint('UMAP', 'n_components'),
    'min_dist': configuration.getfloat('UMAP', 'min_dist'),
    'random_state': RANDOM_STATE,
}


###### VIDEO PARAMETERS #####
DEFAULT_FONT_SCALE: int = configuration.getint('VIDEO', 'DEFAULT_FONT_SCALE')
# DEFAULT_TEXT_BGR: Tuple[int] = literal_eval(configuration.get('VIDEO', 'DEFAULT_TEXT_BGR'))
DEFAULT_TEXT_BGR: Tuple[int] = eval(configuration.get('VIDEO', 'DEFAULT_TEXT_BGR'))
# DEFAULT_TEXT_BACKGROUND_BGR: Tuple[int] = literal_eval(configuration.get('VIDEO', 'DEFAULT_TEXT_BACKGROUND_BGR'))
DEFAULT_TEXT_BACKGROUND_BGR: Tuple[int] = eval(configuration.get('VIDEO', 'DEFAULT_TEXT_BACKGROUND_BGR'))

map_ext_to_fourcc = {
    'mp4': 'mp4v',
    'avi': 'MJPG',
}

### VIDEO asserts
assert isinstance(DEFAULT_TEXT_BGR, tuple), f'DEFAULT_TEXT_BGR was expected to be a tuple but instead found type: {type(DEFAULT_TEXT_BGR)}  (value = {DEFAULT_TEXT_BGR}'
assert len(DEFAULT_TEXT_BGR) == 3, f'DEFAULT_TEXT_BGR was expected to have 3 elements but instead found: {len(DEFAULT_TEXT_BGR)}'
assert isinstance(DEFAULT_TEXT_BACKGROUND_BGR, tuple), f'DEFAULT_TEXT_BACKGROUND_BGR was expected to be a tuple but instead found type: {type(DEFAULT_TEXT_BACKGROUND_BGR)} (value = {DEFAULT_TEXT_BACKGROUND_BGR}'
assert len(DEFAULT_TEXT_BACKGROUND_BGR) == 3, f'DEFAULT_TEXT_BACKGROUND_BGR was expected to have 3 elements but instead found: {len(DEFAULT_TEXT_BACKGROUND_BGR)}'


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
    # file_name_without_extension, extension = file_name.split('.')  # Old way of doing things
    ext_common_idx = file_name.rfind('.')
    file_name_without_extension, extension = file_name[:ext_common_idx], file_name[ext_common_idx + 1:]

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

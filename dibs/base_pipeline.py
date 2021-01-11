"""

"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle as sklearn_shuffle_dataframe
from typing import Any, Collection, Dict, List, Optional, Tuple  # TODO: med: review all uses of Optional
import inspect
import joblib
import numpy as np
from openTSNE import TSNE as OpenTsneObj
import os
import pandas as pd
import time

import sys
from bhtsne import tsne as TSNE_bhtsne
# from pandas.core.common import SettingWithCopyWarning
# from tqdm import tqdm
# import openTSNE  # openTSNE only supports n_components 2 or less

from dibs.logging_enhanced import get_current_function
from dibs import check_arg, config, io, statistics, videoprocessing, visuals

logger = config.initialize_logger(__file__)


# Base pipeline objects that outline the API
class BasePipeline(object):
    """BasePipeline

    It enumerates the basic functions by which each pipeline should adhere.


    Parameters
    ----------
    name : str
        Name of pipeline. Also is the name of the saved pipeline file.


    kwargs
        Kwargs default to pulling in data from config.ini file unless overtly specified to override. See below.
    ----------

    tsne_source : {'sklearn', 'bhtsne'}
        Specify a TSNE implementation to use for dimensionality reduction.

    clf_type : {'svm', 'rf' }
        Specify a classifier to use.
        Default is 'svm'.
        - 'svm' : Support Vector Machine
        - 'rf' : Random Forest


        # TODO: med: expand on further kwargs
    """

    # Base information
    _name, _description = 'DefaultPipelineName', '(Default pipeline description)'
    # data_ext: str = 'csv'  # Extension which data is read from  # TODO: deprecate, delete line
    # dims_cols_names = None  # Union[List[str], Tuple[str]]
    valid_tsne_sources: set = {'bhtsne', 'sklearn', }
    gmm_assignment_col_name, svm_assignment_col_name, = 'gmm_assignment', 'svm_assignment'
    behaviour_col_name = 'behaviour'

    # Tracking vars
    _is_built = False  # Is False until the classifiers are built then changes to True

    _is_training_data_set_different_from_model_input: bool = False  # Changes to True if new training data is added and classifiers not rebuilt.
    _has_unengineered_predict_data: bool = False  # Changes to True if new predict data is added. Changes to False if features are engineered.
    _has_modified_model_variables: bool = False

    # Data
    default_cols = ['data_source', 'file_source']  # , svm_assignment_col_name, gmm_assignment_col_name]
    df_features_train_raw = pd.DataFrame(columns=default_cols)
    df_features_train = pd.DataFrame(columns=default_cols)
    df_features_train_scaled = pd.DataFrame(columns=default_cols)
    df_features_predict_raw = pd.DataFrame(columns=default_cols)
    df_features_predict = pd.DataFrame(columns=default_cols)
    df_features_predict_scaled = pd.DataFrame(columns=default_cols)

    # Other model vars (Rename this)
    input_videos_fps = config.VIDEO_FPS
    cross_validation_k: int = config.CROSSVALIDATION_K
    cross_validation_n_jobs: int = config.CROSSVALIDATION_N_JOBS
    _random_state: int = None
    average_over_n_frames: int = 3
    test_train_split_pct: float = None

    # Model objects
    _scaler: StandardScaler = None
    _clf_gmm: GaussianMixture = None

    # TSNE
    tsne_implementation: str = config.TSNE_IMPLEMENTATION
    tsne_n_components: int = config.TSNE_N_COMPONENTS
    tsne_n_iter: int = config.TSNE_N_ITER
    tsne_early_exaggeration: float = config.TSNE_EARLY_EXAGGERATION
    tsne_n_jobs: int = config.TSNE_N_JOBS  # n cores used during process
    tsne_verbose: int = config.TSNE_VERBOSE
    tsne_init: str = config.TSNE_INIT
    _tsne_perplexity: Optional[float] = config.TSNE_PERPLEXITY
    tsne_learning_rate: float = config.TSNE_LEARNING_RATE
    # GMM
    gmm_n_components, gmm_covariance_type, gmm_tol, gmm_reg_covar = None, None, None, None
    gmm_max_iter, gmm_n_init, gmm_init_params = None, None, None
    gmm_verbose: int = None
    gmm_verbose_interval: int = None

    # Classifier
    _classifier = None
    clf_type: str = config.DEFAULT_CLASSIFIER
    # Classifier: SVM
    svm_c, svm_gamma, svm_probability, svm_verbose = None, None, None, None
    # Classifier: Random Forest
    rf_n_estimators = config.rf_n_estimators

    # Column names
    features_which_average_by_mean = ['DistFrontPawsTailbaseRelativeBodyLength', 'DistBackPawsBaseTailRelativeBodyLength', 'InterforepawDistance', 'BodyLength', ]
    features_which_average_by_sum = ['SnoutToTailbaseChangeInAngle', 'SnoutSpeed', 'TailbaseSpeed']
    _all_features: Tuple[str] = tuple(features_which_average_by_mean + features_which_average_by_sum)
    test_col_name = 'is_test_data'

    # All label properties for respective assignments instantiated below to ensure no missing properties b/w Pipelines (aka: quick fix, not enough time to debug in full)
    label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9 = ['' for _ in range(10)]
    label_10, label_11, label_12, label_13, label_14, label_15, label_16, label_17, label_18 = ['' for _ in range(9)]
    label_19, label_20, label_21, label_22, label_23, label_24, label_25, label_26, label_27 = ['' for _ in range(9)]
    label_28, label_29, label_30, label_31, label_32, label_33, label_34, label_35, label_36 = ['' for _ in range(9)]

    # Misc attributes
    kwargs: dict = {}
    _last_built: str = None

    # SORT ME
    _acc_score: float = None
    _cross_val_scores: np.ndarray = np.array([])
    seconds_to_engineer_train_features: float = None
    seconds_to_build: float = -1.

    # TODO: med/high: create tests for this func below
    def get_assignment_label(self, assignment: int) -> str:
        """
        Get behavioural label according to assignment value (number).
        If a label does not exist for a given assignment, then return empty string.
        """
        try:
            assignment = int(assignment)
        except ValueError:
            err = f'TODO: elaborate error: invalid assignment submitted: "{assignment}"'
            logger.error(err)
            raise ValueError(err)

        label = getattr(self, f'label_{assignment}', '')

        return label

    def set_label(self, assignment: int, label: str):
        """ Set behavioural label for a given model assignment number/value """
        check_arg.ensure_type(label, str)
        assignment = int(assignment)
        setattr(self, f'label_{assignment}', label)
        return self

    # # # Getters/Properties
    @property
    def is_in_inconsistent_state(self):
        """
        Useful for checking if training data has been added/removed from pipeline
        relative to already-compiled model
        """
        return self._is_training_data_set_different_from_model_input

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def tsne_perplexity(self):  # TODO: <---- REVIEW !!!! It's implciit math!
        return self._tsne_perplexity if self._tsne_perplexity else np.sqrt(len(self.all_features))

    @property
    def clf_gmm(self):
        return self._clf_gmm

    @property
    def clf(self):
        return self._classifier

    @property
    def random_state(self):
        return self._random_state

    def get_desc(self) -> str:
        return self._description

    @property
    def is_built(self):
        return self._is_built

    @property
    def accuracy_score(self):
        return self._acc_score

    @property
    def scaler(self):
        return self._scaler

    @property
    def svm_col(self) -> str:
        return self.svm_assignment_col_name

    @property
    def svm_assignment(self) -> str:
        return self.svm_assignment_col_name

    @property
    def cross_val_scores(self):  # Union[List, np.ndarray]
        return self._cross_val_scores

    @property
    def training_data_sources(self) -> List[str]:
        return list(np.unique(self.df_features_train_raw['data_source'].values))

    @property
    def predict_data_sources(self):  # List[str]
        return list(np.unique(self.df_features_predict_raw['data_source'].values))

    @property
    def raw_assignments(self):  # List[str]
        return self.raw_assignments

    @property
    def unique_assignments(self) -> List[any]:
        if len(self.df_features_train_scaled) > 0:
            return list(np.unique(self.df_features_train_scaled[self.svm_col].values))
        return []

    @property
    def all_features(self) -> Collection[str]:
        return self._all_features

    @property
    def all_features_list(self) -> List[str]:
        return list(self._all_features)

    @property
    def total_build_time(self):
        return self.seconds_to_engineer_train_features

    # Setters
    def set_name(self, name: str):
        # TODO: MED: will this cause problems later with naming convention?
        check_arg.ensure_has_valid_chars_for_path(name)
        self._name = name
        return self

    def set_description(self, description):
        """ Set a description of the pipeline. Include any notes you want to keep regarding the process used. """
        check_arg.ensure_type(description, str)
        self._description = description
        return self

    def __bool__(self):
        return True

    # Init
    def __init__(self, name: str, **kwargs):
        # Pipeline name
        check_arg.ensure_type(name, str)
        self.set_name(name)

        # TSNE source  # TODO: HIGH: move this section to set_params
        tsne_source = kwargs.get('tsne_source', '')
        check_arg.ensure_type(tsne_source, str)
        if tsne_source in self.valid_tsne_sources:
            self.tsne_implementation = tsne_source
        #
        self.kwargs = kwargs
        # Final setup
        self.set_params(read_config_on_missing_param=True, **kwargs)

    def set_params(self, read_config_on_missing_param: bool = False, **kwargs):
        """
        Reads in variables to change for pipeline.

        If optional arg `read_config_on_missing_param` is True, then any parameter NOT mentioned
        explicitly will be read in from the config.ini file and then replace the current value
        for that property in the pipeline.

        Valid Kwargs:
            - input_video_fps
            - random_state
            - tsne_n_components
            - tsne_n_iter
            - tsne_early_exaggeration
            - tsne_n_jobs
            - tsne_verbose
            TODO: low: complete list

        """
        check_arg.ensure_type(read_config_on_missing_param, bool)
        ### General Params ###
        # TODO: MED: ADD KWARGS OPTION FOR OVERRIDING VERBOSE in CONFIG.INI!!!!!!!! ?
        video_fps = kwargs.get('input_videos_fps', config.VIDEO_FPS if read_config_on_missing_param else self.input_videos_fps)
        check_arg.ensure_type(video_fps, int, float)
        self.input_videos_fps = video_fps
        average_over_n_frames = kwargs.get('average_over_n_frames', self.average_over_n_frames)  # TODO: low: add a default option for this in config.ini+config.py
        check_arg.ensure_type(average_over_n_frames, int)
        self.average_over_n_frames = average_over_n_frames
        random_state = kwargs.get('random_state', config.RANDOM_STATE if read_config_on_missing_param else self.random_state)  # TODO: low: ensure random state correct
        check_arg.ensure_type(random_state, int)
        self._random_state = random_state
        ### TSNE ###
        tsne_implementation = kwargs.get('tsne_implementation', config.TSNE_IMPLEMENTATION if read_config_on_missing_param else self.tsne_implementation)
        check_arg.ensure_type(tsne_implementation, str)
        self.tsne_implementation = tsne_implementation
        tsne_init = kwargs.get('tsne_init', config.TSNE_INIT if read_config_on_missing_param else self.tsne_init)
        check_arg.ensure_type(tsne_init, str)
        self.tsne_init = tsne_init
        tsne_early_exaggeration = kwargs.get('tsne_early_exaggeration', config.TSNE_EARLY_EXAGGERATION if read_config_on_missing_param else self.tsne_early_exaggeration)
        check_arg.ensure_type(tsne_early_exaggeration, float)
        self.tsne_early_exaggeration = tsne_early_exaggeration
        tsne_learning_rate = kwargs.get('tsne_learning_rate', config.TSNE_LEARNING_RATE if read_config_on_missing_param else self.tsne_learning_rate)
        check_arg.ensure_type(tsne_learning_rate, float)
        self.tsne_learning_rate = tsne_learning_rate
        tsne_n_components = kwargs.get('tsne_n_components', config.TSNE_N_COMPONENTS if read_config_on_missing_param else self.tsne_n_components)  # TODO: low: shape up kwarg name for n components? See string name
        check_arg.ensure_type(tsne_n_components, int)
        self.tsne_n_components = tsne_n_components
        tsne_n_iter = kwargs.get('tsne_n_iter', config.TSNE_N_ITER if read_config_on_missing_param else self.tsne_n_iter)
        check_arg.ensure_type(tsne_n_iter, int)
        self.tsne_n_iter = tsne_n_iter
        tsne_n_jobs = kwargs.get('tsne_n_jobs', config.TSNE_N_JOBS if read_config_on_missing_param else self.tsne_n_jobs)
        check_arg.ensure_type(tsne_n_jobs, int)
        self.tsne_n_jobs = tsne_n_jobs
        tsne_perplexity = kwargs.get('tsne_perplexity', config.TSNE_PERPLEXITY if read_config_on_missing_param else self._tsne_perplexity)
        check_arg.ensure_type(tsne_perplexity, float)
        self._tsne_perplexity = tsne_perplexity
        tsne_verbose = kwargs.get('tsne_verbose', config.TSNE_VERBOSE if read_config_on_missing_param else self.tsne_verbose)
        check_arg.ensure_type(tsne_verbose, int)
        self.tsne_verbose = tsne_verbose

        # GMM vars
        gmm_n_components = kwargs.get('gmm_n_components', config.gmm_n_components if read_config_on_missing_param else self.gmm_n_components)
        check_arg.ensure_type(gmm_n_components, int)
        self.gmm_n_components = gmm_n_components
        gmm_covariance_type = kwargs.get('gmm_covariance_type', config.gmm_covariance_type if read_config_on_missing_param else self.gmm_covariance_type)
        check_arg.ensure_type(gmm_covariance_type, str)
        self.gmm_covariance_type = gmm_covariance_type
        gmm_tol = kwargs.get('gmm_tol', config.gmm_tol if read_config_on_missing_param else self.gmm_tol)
        check_arg.ensure_type(gmm_tol, float)
        self.gmm_tol = gmm_tol
        gmm_reg_covar = kwargs.get('gmm_reg_covar', config.gmm_reg_covar if read_config_on_missing_param else self.gmm_reg_covar)
        check_arg.ensure_type(gmm_reg_covar, float)
        self.gmm_reg_covar = gmm_reg_covar
        gmm_max_iter = kwargs.get('gmm_max_iter',config.gmm_max_iter if read_config_on_missing_param else self.gmm_max_iter)
        check_arg.ensure_type(gmm_max_iter, int)
        self.gmm_max_iter = gmm_max_iter
        gmm_n_init = kwargs.get('gmm_n_init', config.gmm_n_init if read_config_on_missing_param else self.gmm_n_init)
        check_arg.ensure_type(gmm_n_init, int)
        self.gmm_n_init = gmm_n_init
        gmm_init_params = kwargs.get('gmm_init_params', config.gmm_init_params if read_config_on_missing_param else self.gmm_init_params)
        check_arg.ensure_type(gmm_init_params, str)
        self.gmm_init_params = gmm_init_params
        gmm_verbose = kwargs.get('gmm_verbose', config.gmm_verbose if read_config_on_missing_param else self.gmm_verbose)
        check_arg.ensure_type(gmm_verbose, int)
        self.gmm_verbose = gmm_verbose
        gmm_verbose_interval = kwargs.get('gmm_verbose_interval', config.gmm_verbose_interval if read_config_on_missing_param else self.gmm_verbose_interval)
        check_arg.ensure_type(gmm_verbose_interval, int)
        self.gmm_verbose_interval = gmm_verbose_interval
        # Classifier vars
        clf_type = kwargs.get('clf_type', config.DEFAULT_CLASSIFIER if read_config_on_missing_param else self.clf_type)
        self.clf_type = clf_type
        # Random Forest vars
        rf_n_estimators = kwargs.get('rf_n_estimators', config.rf_n_estimators if read_config_on_missing_param else self.rf_n_estimators)
        check_arg.ensure_type(rf_n_estimators, int)
        self.rf_n_estimators = rf_n_estimators
        # SVM vars
        svm_c = kwargs.get('svm_c', config.svm_c if read_config_on_missing_param else self.svm_c)
        self.svm_c = svm_c
        svm_gamma = kwargs.get('svm_gamma', config.svm_gamma if read_config_on_missing_param else self.svm_gamma)
        self.svm_gamma = svm_gamma
        svm_probability = kwargs.get('svm_probability', config.svm_probability if read_config_on_missing_param else self.svm_probability)
        self.svm_probability = svm_probability
        svm_verbose = kwargs.get('svm_verbose', config.svm_verbose if read_config_on_missing_param else self.svm_verbose)
        self.svm_verbose = svm_verbose
        cross_validation_k = kwargs.get('cross_validation_k', config.CROSSVALIDATION_K if read_config_on_missing_param else self.cross_validation_k)
        check_arg.ensure_type(cross_validation_k, int)
        self.cross_validation_k = cross_validation_k

        # TODO: low/med: add kwargs for parsing test/train split pct
        if self.test_train_split_pct is None:
            self.test_train_split_pct = config.HOLDOUT_PERCENT

        # self.dims_cols_names = [f'dim_{d+1}' for d in range(self.tsne_n_components)]  # TODO: remove this coment, it is now a property

        self._has_modified_model_variables = True
        return self

    @property
    def dims_cols_names(self) -> List[str]:
        return [f'dim_{d+1}' for d in range(self.tsne_n_components)]

    # Functions that should be overwritten by child classes
    def engineer_features(self, data: pd.DataFrame):
        err = f'{get_current_function()}(): Not Implemented for base ' \
              f'Pipeline object {self.__name__}. You must implement this for all child objects.'
        logger.error(err)
        raise NotImplementedError(err)

    # Add & delete data
    def add_train_data_source(self, *train_data_paths_args):
        """
        Reads in new data and saves raw data. A source can be either a directory or a file.
        If the source is a path, then all CSV files directly within that directory will
        be read in as DLC files
        # TODO: add in

        train_data_args: any number of args. Types submitted expected to be of type [str]
            In the case that an arg is List[str], the arg must be a list of strings that
        """
        train_data_paths_args = [path for path in train_data_paths_args
                                 if os.path.split(path)[-1].split('.')[0]  # <- Get file name without extension
                                 not in set(self.df_features_train_raw['data_source'].values)]
        for path in train_data_paths_args:
            if os.path.isfile(path):
                df_new_data = io.read_csv(path)
                self.df_features_train_raw = self.df_features_train_raw.append(df_new_data)
                self._is_training_data_set_different_from_model_input = True
                logger.debug(f'Added file to train data: {path}')
            elif os.path.isdir(path):
                logger.debug(f'Attempting to pull DLC files from {path}')
                data_sources: List[str] = [os.path.join(path, file_name)
                                           for file_name in os.listdir(path)
                                           if file_name.split('.')[-1] in config.valid_dlc_output_extensions
                                           and file_name not in set(self.df_features_train_raw['data_source'].values)]
                for file_path in data_sources:
                    logger.debug(f'Reading in: {file_path}')
                    df_new_data_i = io.read_csv(file_path)
                    self.df_features_train_raw = self.df_features_train_raw.append(df_new_data_i)
                    self._is_training_data_set_different_from_model_input = True
                    logger.debug(f'Added file to train data: {file_path}')
            else:
                unusual_path_err = f'Unusual file/dir path submitted but not found: "{path}". Is not a valid ' \
                                   f'file and not a directory.'
                logger.error(unusual_path_err)
                raise ValueError(unusual_path_err)

        return self

    def add_predict_data_source(self, *predict_data_path_args):
        """
        Reads in new data and saves raw data. A source can be either a directory or a file.

        predict_data_args: any number of args. Types submitted expected to be of type str.
            In the case that an arg is List[str], the arg must be a list of strings that
        """
        predict_data_path_args = [path for path in predict_data_path_args
                                  if os.path.split(path)[-1].split('.')[0]  # <- Get file name without extension
                                  not in set(self.df_features_predict_raw['data_source'].values)]
        for path in predict_data_path_args:
            if os.path.isfile(path):
                df_new_data = io.read_csv(path)
                self.df_features_predict_raw = self.df_features_predict_raw.append(df_new_data)
                self._has_unengineered_predict_data = True
                logger.debug(f'Added file to predict data: {path}')
            elif os.path.isdir(path):
                logger.debug(f'Attempting to pull DLC files from {path}')
                data_sources: List[str] = [os.path.join(path, file_name)
                                           for file_name in os.listdir(path)
                                           if file_name.split('.')[-1] in config.valid_dlc_output_extensions
                                           and file_name not in set(self.df_features_predict_raw['data_source'].values)]
                for file_path in data_sources:
                    df_new_data_i = io.read_csv(file_path)
                    self.df_features_predict_raw = self.df_features_predict_raw.append(df_new_data_i)
                    self._has_unengineered_predict_data = True
                    logger.debug(f'Added file to predict data: {file_path}')
            else:
                unusual_path_err = f'Unusual file/dir path submitted but not found: {path}. Is not a valid ' \
                                   f'file and not a directory.'
                logger.error(unusual_path_err)
                raise ValueError(unusual_path_err)

        return self

    def remove_train_data_source(self, data_source: str):
        """
        Remove a data source from the training data set.
        If the data source specified does not exist, then nothing changes.
        """
        check_arg.ensure_type(data_source, str)
        self.df_features_train_raw = self.df_features_train_raw.loc[
            self.df_features_train_raw['data_source'] != data_source]
        self.df_features_train = self.df_features_train.loc[
            self.df_features_train['data_source'] != data_source]
        self.df_features_train_scaled = self.df_features_train_scaled.loc[
            self.df_features_train_scaled['data_source'] != data_source]

        return self

    def remove_predict_data_source(self, data_source: str):
        """
        Remove data from predicted data set.
        :param data_source: (str) name of a data source
        """
        # TODO: low: ensure function, add tests
        check_arg.ensure_type(data_source, str)
        self.df_features_predict_raw = self.df_features_predict_raw.loc[
            self.df_features_predict_raw['data_source'] != data_source]
        self.df_features_predict = self.df_features_predict.loc[
            self.df_features_predict['data_source'] != data_source]
        self.df_features_predict_scaled = self.df_features_predict_scaled.loc[
            self.df_features_predict_scaled['data_source'] != data_source]
        return self

    # Engineer features
    def engineer_features_all_dfs(self, list_dfs_of_raw_data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        The main function that can build features for BOTH training and prediction data.
        Here we are ensuring that the data processing for both training and prediction
        occurs in the same way.

        Assumptions:
            - Each returned feature is assumed to be a float and is converted as such after
                the child return of self.engineer_features() executes (bug fixing reasons. For whatever
                reason, feature cols kept being returned as Object type and sklearn TSNE handled
                it just fine but bhtsne did not).
        """
        # TODO: MED: these cols really should be saved in
        #  engineer_7_features_dataframe_NOMISSINGDATA(),
        #  but that func can be amended later due to time constraints

        # Reconcile args
        if isinstance(list_dfs_of_raw_data, pd.DataFrame):
            list_dfs_raw_data = [list_dfs_of_raw_data, ]

        check_arg.ensure_type(list_dfs_of_raw_data, list)

        # Execute
        list_dfs_engineered_features: List[pd.DataFrame] = []
        for i, df in enumerate(list_dfs_of_raw_data):
            df = df.copy().astype({'frame': float})
            check_arg.ensure_frame_indices_are_integers(df)
            logger.debug(f'{get_current_function()}(): Engineering df feature set {i+1} of {len(list_dfs_of_raw_data)}')
            df_engineered_features: pd.DataFrame = self.engineer_features(df)
            df_engineered_features = df_engineered_features.astype({feature: float for feature in self.all_features})
            list_dfs_engineered_features.append(df_engineered_features)

        # Aggregate all data into one DataFrame, return
        df_features = pd.concat(list_dfs_engineered_features)

        return df_features

    def engineer_features_train(self):
        """
        Utilizes
        All functions that take the raw data (data retrieved from using dibs.read_csv()) and
        transforms it into classifier-ready data.

        Post-conditions: sets
        Returns self.
        """
        # TODO: low: save feature engineering time for train data
        start = time.perf_counter()
        # Queue up data according to data source
        list_dfs_raw_data = [self.df_features_train_raw.loc[self.df_features_train_raw['data_source'] == src]
                                 .astype({'frame': float}).sort_values('frame').copy()
                             for src in set(self.df_features_train_raw['data_source'].values)]
        # Call engineering function
        logger.debug(f'Start engineering training data features.')
        df_features = self.engineer_features_all_dfs(list_dfs_raw_data)
        logger.debug(f'Done engineering training data features.')
        # Save data
        self.df_features_train = df_features
        # Wrap up
        end = time.perf_counter()
        self._is_training_data_set_different_from_model_input = False
        self.seconds_to_engineer_train_features = round(end - start, 1)
        return self

    def engineer_features_predict(self):
        """ Engineer features for the predicted data"""
        # Queue data
        list_dfs_raw_data = [self.df_features_predict_raw.loc[self.df_features_predict_raw['data_source'] == src]
                                 .sort_values('frame').copy()
                             for src in set(self.df_features_predict_raw['data_source'].values)]
        # Call engineering function
        logger.debug(f'Start engineering predict data features.')
        df_features = self.engineer_features_all_dfs(list_dfs_raw_data)
        logger.debug(f'Done engineering predict data features.')
        # Save data, return
        self.df_features_predict = df_features
        self._has_unengineered_predict_data = False
        return self

    ## Scaling data
    def _create_scaled_data(self, df_data, features, create_new_scaler: bool = False) -> pd.DataFrame:
        """
        A universal data scaling function that is usable for training data as well as new prediction data.
        Scales down features in place and does not keep original data.
        """
        # Check args
        check_arg.ensure_type(features, list, tuple)
        check_arg.ensure_columns_in_DataFrame(df_data, features)
        # Execute
        if create_new_scaler:
            self._scaler = StandardScaler()
            self._scaler.fit(df_data[features])
        arr_data_scaled: np.ndarray = self.scaler.transform(df_data[features])
        df_scaled_data = pd.DataFrame(arr_data_scaled, columns=features)
        # For new DataFrame, replace columns that were not scaled so that data does not go missing
        for col in df_data.columns:
            if col not in set(df_scaled_data.columns):
                df_scaled_data[col] = df_data[col].values
        return df_scaled_data

    def scale_transform_train_data(self, features: Collection[str] = None, create_new_scaler=True):
        """
        Scales training data. By default, creates new scaler according to train
        data and stores it in pipeline

        :param features:
        :param create_new_scaler:

        :return: self
        """
        # Queue up data to use
        if features is None:  # TODO: low: remove his if statement as a default feature
            features = self.all_features
        df_features_train = self.df_features_train
        # Check args
        check_arg.ensure_type(features, list, tuple)
        check_arg.ensure_columns_in_DataFrame(df_features_train, features)
        # Get scaled data
        df_scaled_data = self._create_scaled_data(df_features_train, list(features), create_new_scaler=create_new_scaler)
        check_arg.ensure_type(df_scaled_data, pd.DataFrame)  # Debugging effort. Remove later.
        # Save data. Return.
        self.df_features_train_scaled = df_scaled_data
        return self

    def scale_transform_predict_data(self, features: List[str] = None):
        """
        Scales prediction data. Utilizes existing scaler.
        If no feature set is explicitly specified, then the default features set in the Pipeline are used.
        :param features:
        :return:
        """
        # Queue up data to use
        if features is None:
            features = self.all_features
        features = list(features)
        df_features_predict = self.df_features_predict

        # Check args before execution
        check_arg.ensure_type(features, list, tuple)
        check_arg.ensure_type(df_features_predict, pd.DataFrame)
        check_arg.ensure_columns_in_DataFrame(df_features_predict, features)

        # Get scaled data
        df_scaled_data: pd.DataFrame = self._create_scaled_data(df_features_predict, features, create_new_scaler=False)
        check_arg.ensure_type(df_scaled_data, pd.DataFrame)  # Debugging effort. Remove later.

        # Save data. Return.
        self.df_features_predict_scaled = df_scaled_data
        return self

    # TSNE Transformations
    def _train_tsne_get_dimension_reduced_data(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        TODO: elaborate
        TODO: ensure that TSNE obj can be saved and used later for new data? *** Important ***
        :param data:
        :param kwargs:
        :return:
        """
        # Check args
        check_arg.ensure_type(data, pd.DataFrame)
        check_arg.ensure_columns_in_DataFrame(data, self.all_features_list)
        # Execute
        if self.tsne_implementation == 'sklearn':
            logger.debug(f'Now reducing data with SKLEARN implementation...')
            arr_result = TSNE_sklearn(
                perplexity=self.tsne_perplexity,
                learning_rate=self.tsne_learning_rate,  # alpha*eta = n  # TODO: encapsulate this later                     !!!
                n_components=self.tsne_n_components,
                random_state=self.random_state,
                n_iter=self.tsne_n_iter,
                early_exaggeration=self.tsne_early_exaggeration,
                n_jobs=self.tsne_n_jobs,
                verbose=self.tsne_verbose,
                init=self.tsne_init,
            ).fit_transform(data[list(self.all_features_list)])
        elif self.tsne_implementation == 'bhtsne':
            logger.debug(f'Now reducing data with bhtsne implementation...')
            arr_result = TSNE_bhtsne(
                # TODO: low: investigate: ValueError: Expected n_neighbors > 0. Got -2
                data[list(self.all_features)],
                dimensions=self.tsne_n_components,
                perplexity=self.tsne_perplexity,  # TODO: implement math somewhere else
                rand_seed=self.random_state,
                theta=0.5,
            )
        elif self.tsne_implementation == 'opentsne':
            logger.debug(f'Now reducing data with OpenTSNE implementation...')
            tsne = OpenTsneObj(
                n_components=self.tsne_n_components,
                perplexity=self.tsne_perplexity,
                learning_rate='auto',  # TODO: med: review
                early_exaggeration=self.tsne_early_exaggeration,
                n_iter=self.tsne_n_iter,
                n_jobs=self.tsne_n_jobs,
                negative_gradient_method='bh',  # Note: default 'fft' does not work with dims >2
                random_state=self.random_state,
                verbose=bool(self.tsne_verbose),
                metric="euclidean",  # TODO: med: review
                # early_exaggeration_iter=250,
                # n_iter=500,
                # exaggeration=None,
                # dof=1,
                # theta=0.5,
                # n_interpolation_points=3,
                # min_num_intervals=50,
                # ints_in_interval=1,
                # initialization="pca",
                # metric_params=None,
                # initial_momentum=0.5,
                # final_momentum=0.8,
                # max_grad_norm=None,
                # max_step_norm=5,
                # affinities=None,
                # neighbors="auto",
                # callbacks=None,
                # callbacks_every_iters=50,
            )
            arr_result = tsne.fit(data[list(self.all_features_list)].values)
        else:
            err = f'Invalid TSNE source type fell through the cracks: {self.tsne_implementation}'
            logger.error(err)
            raise RuntimeError(err)
        return arr_result

    def train_GMM(self, data: pd.DataFrame):
        """"""
        self._clf_gmm = GaussianMixture(
            n_components=self.gmm_n_components,
            covariance_type=self.gmm_covariance_type,
            tol=self.gmm_tol,
            reg_covar=self.gmm_reg_covar,
            max_iter=self.gmm_max_iter,
            n_init=self.gmm_n_init,
            init_params=self.gmm_init_params,
            verbose=self.gmm_verbose,
            verbose_interval=self.gmm_verbose_interval,
            random_state=self.random_state,
        ).fit(data)
        return self

    def gmm_predict(self, data):  # TODO: low: remove func?
        assignments = self.clf_gmm.predict(data)
        return assignments

    def train_classifier(self):
        # TODO: HIGH: finish this function!
        df = self.df_features_train_scaled

        if self.clf_type == 'SVM':
            clf = SVC(
                C=self.svm_c,
                gamma=self.svm_gamma,
                probability=self.svm_probability,
                verbose=self.svm_verbose,
                random_state=self.random_state,
            )
        elif self.clf_type == 'RANDOMFOREST':
            clf = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                # criterion='gini',
                # max_depth=None,
                # min_samples_split=2,
                # min_samples_leaf=1,
                # min_weight_fraction_leaf=0,
                # max_features="auto",
                # max_leaf_nodes=None,
                # min_impurity_decrease=0.,
                # min_impurity_split=None,
                # bootstrap=True,
                # oob_score=False,
                # n_jobs=None,
                # random_state=None,
                # verbose=0,
                # warm_start=False,
                # class_weight=None,
                # ccp_alpha=0.0,
                # max_samples=None,
            )
        else:
            err = f'TODO: elaborate: an invalid classifier type was detected: {self.clf_type}'
            logger.error(err)
            raise KeyError(err)
        # Fit classifier to non-test data
        clf.fit(
            X=df.loc[~df[self.test_col_name]][list(self.all_features)],  # TODO: too specific ??? review the veracity of this TODO
            y=df.loc[~df[self.test_col_name]][self.gmm_assignment_col_name],
        )
        # Save classifier
        self._classifier = clf

    # Higher level data processing functions
    def tsne_reduce_df_features_train(self):
        arr_tsne_result = self._train_tsne_get_dimension_reduced_data(self.df_features_train)
        self.df_features_train_scaled = pd.concat([
            self.df_features_train_scaled,
            pd.DataFrame(arr_tsne_result, columns=self.dims_cols_names),
        ], axis=1)

        return self

    # Model building
    def build_model(self, reengineer_train_features: bool = False):
        """
        Builds the model for predicting behaviours.
        :param reengineer_train_features: (bool) If True, forces the training data to be re-engineered.
        """
        # Engineer features
        if reengineer_train_features or self._is_training_data_set_different_from_model_input:
            logger.debug(f'{inspect.stack()[0][3]}(): Start engineering features...')
            self.engineer_features_train()

        # Scale data
        logger.debug(f'Scaling data now...')
        self.scale_transform_train_data(features=self.all_features, create_new_scaler=True)

        # TSNE -- create new dimensionally reduced data
        logger.debug(f'TSNE reducing features now...')
        self.tsne_reduce_df_features_train()

        # Train GMM, get assignments
        logger.debug(f'Training GMM now...')
        self.train_GMM(self.df_features_train_scaled[self.dims_cols_names])
        self.df_features_train_scaled[self.gmm_assignment_col_name] = self.clf_gmm.predict(
            self.df_features_train_scaled[self.dims_cols_names].values)

        # Test-train split
        self.add_test_data_column_to_scaled_train_data()

        # # Train Classifier
        logger.debug(f'Training classifier now...')
        self.train_classifier()  # self.train_SVM()

        # Set predictions
        self.df_features_train_scaled[self.svm_assignment_col_name] = self.clf.predict(
            self.df_features_train_scaled[list(self.all_features)].values)  # Get predictions
        self.df_features_train_scaled[self.svm_assignment_col_name] = self.df_features_train_scaled[
            self.svm_assignment_col_name].astype(int)  # Coerce into int

        logger.debug(f'Generating cross-validation scores...')
        # # Get cross-val accuracy scores
        self._cross_val_scores = cross_val_score(
            self.clf,
            self.df_features_train_scaled[list(self.all_features)],
            self.df_features_train_scaled[self.svm_assignment_col_name],
            cv=self.cross_validation_k,
        )

        logger.debug(f'Generating accuracy score with a test split % of: {self.test_train_split_pct*100}%')
        df_features_train_scaled_test_data = self.df_features_train_scaled.loc[
            ~self.df_features_train_scaled[self.test_col_name]]
        self._acc_score = accuracy_score(
            y_pred=self.clf.predict(df_features_train_scaled_test_data[list(self.all_features)]),
            y_true=df_features_train_scaled_test_data[self.svm_assignment_col_name].values)
        logger.debug(f'Pipeline train accuracy: {self.accuracy_score}')
        # TODO: low: save the confusion matrix after accuracy score too?

        # Final touches. Save state of pipeline.
        self._is_built = True
        self._is_training_data_set_different_from_model_input = False  # TODO: med: review these 3 variables
        self._has_modified_model_variables = False
        self._last_built = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
        logger.debug(f'All done with building classifiers/model!')

        return self

    def build_classifier(self, reengineer_train_features: bool = False):
        """ This is the legacy naming. Method kept for backwards compatibility. This function will be deleted later. """
        warn = f'Pipeline.build_classifier(): was called, but this is the ' \
               f'legacy name. Instead, use Pipeline.build_model() from now on.'
        logger.warning(warn)
        return self.build_model(reengineer_train_features=reengineer_train_features)

    def generate_predict_data_assignments(self, reengineer_train_data_features: bool = False, reengineer_predict_features=False):  # TODO: low: rename?
        """
        Runs after build(). Using terminology from old implementation. TODO: purpose
        """
        # TODO: add arg checking for empty predict data?

        # Check that classifiers are built on the training data
        if reengineer_train_data_features or not self.is_built or self.is_in_inconsistent_state:
            self.build_model()

        # TODO: temp exit early for zero test data found
        if len(self.df_features_predict_raw) == 0:
            warn = f'Zero test data poiknts found. exiting early. predict features not built.'
            logger.warning(warn)
            return self
        # Check if predict features have been engineered
        if reengineer_predict_features or self._has_unengineered_predict_data:
            self.engineer_features_predict()
            self.scale_transform_predict_data()

        # Add prediction labels
        if len(self.df_features_predict_scaled) > 0:
            self.df_features_predict_scaled[self.svm_assignment_col_name] = self.clf.predict(
                self.df_features_predict_scaled[list(self.all_features_list)].values)
        else:
            logger.debug(f'{get_current_function()}(): 0 records were detected '
                         f'for PREDICT data. No data was predicted with model.')

        return self

    def build(self, reengineer_train_features=False, reengineer_predict_features=False):
        """
        Build all classifiers and get predictions from predict data
        """
        start = time.perf_counter()
        # Build model
        self.build_model(reengineer_train_features=reengineer_train_features)
        # Get predict data
        self.generate_predict_data_assignments(reengineer_predict_features=reengineer_predict_features)
        end = time.perf_counter()
        self.seconds_to_build = round(end - start, 3)
        return self

    # More data transformations
    def add_test_data_column_to_scaled_train_data(self):
        """
        Add boolean column to scaled training data DataFrame to assign train/test data
        """
        test_data_col_name = self.test_col_name
        check_arg.ensure_type(test_data_col_name, str)

        df = self.df_features_train_scaled
        df_shuffled = sklearn_shuffle_dataframe(df)  # Shuffles data, loses none in the process. Assign bool accordingly

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
        df_shuffled[test_data_col_name] = False  # TODO: med: Setting with copy warning occurs on this exact line. is this not how to instantiate it? https://realpython.com/pandas-settingwithcopywarning/
        df_shuffled.loc[:int(len(df) * self.test_train_split_pct), test_data_col_name] = True

        df_shuffled = df_shuffled.reset_index()
        self.df_features_train_scaled = df_shuffled

        return self

    # Saving and stuff
    def save(self, output_path_dir=config.OUTPUT_PATH):
        """
        Defaults to config.ini OUTPUT_PATH variable if a save path not specified beforehand.
        :param output_path_dir: (str) an absolute path to a DIRECTORY where the pipeline will be saved.
        """
        warn = f'This function, "{get_current_function()}()", will be deprecated ' \
               f'in the future since naming is too vague. Instead, use save_to_folder()'
        logger.warning(warn)
        return self.save_to_folder(output_path_dir)

    def save_to_folder(self, output_path_dir=config.OUTPUT_PATH):
        """
        Defaults to config.ini OUTPUT_PATH variable if a save path not specified beforehand.
        :param output_path_dir: (str) an absolute path to a DIRECTORY where the pipeline will be saved.
        """
        logger.debug(
            f'{inspect.stack()[0][3]}(): Attempting to save pipeline to the following folder: {output_path_dir}.')

        # Check if valid directory
        check_arg.ensure_is_dir(output_path_dir)

        # Execute
        final_out_path = os.path.join(output_path_dir, generate_pipeline_filename(self._name))
        # Check if valid final path to be saved
        check_arg.ensure_is_valid_path(final_out_path)
        if not check_arg.is_pathname_valid(final_out_path):  # TODO: low: review
            invalid_path_err = f'Invalid output path save: {final_out_path}'
            logger.error(invalid_path_err)
            raise ValueError(invalid_path_err)

        logger.debug(f'{inspect.stack()[0][3]}(): Attempting to save pipeline to file: {final_out_path}.')

        # Write to file
        try:
            with open(final_out_path, 'wb') as model_file:
                joblib.dump(self, model_file)
        except Exception as e:
            err = f'{get_current_function()}(): An unexpected error occurred: {repr(e)}'
            logger.error(err)
            raise e

        logger.debug(f'{inspect.stack()[0][3]}(): Pipeline ({self.name}) saved to: {final_out_path}')
        return io.read_pipeline(final_out_path)

    # Video stuff

    def make_video(self, video_to_be_labeled_path: str, data_source: str, video_name: str, output_dir: str,
                   output_fps: float = config.OUTPUT_VIDEO_FPS):
        """

        :param video_to_be_labeled_path: (str) Path to a video that will be labeled
            e.g.: "/home/videos/Vid1.mp4"
        :param data_source: (str) Name of data source set that corresponds to the video which
            is going to be labeled. If you input the wrong data source relative to corresponding video,
            the behaviours associated with the video will not properly line up and the output
            video will have labels that make no sense
            e.g.: "Vid1DLC_sourceABC"
        :param video_name:  (str) Name of the output video. Do NOT include the extension in the name.
            e.g.: "Vid1Labeled"
        :param output_dir: (str) Path to a directory into which the labeled video will be saved.
        :param output_fps: (either: int OR float)The FPS of the labeled video output
        :return:
        """

        # Arg checking
        check_arg.ensure_is_file(video_to_be_labeled_path)
        check_arg.ensure_type(data_source, str)
        # check_arg.ensure_has_valid_chars_for_path(video_name)  # TODO: low/med: review this line then implement
        check_arg.ensure_is_dir(output_dir)
        if not self.is_built:
            err = f'Model is not built so cannot make labeled video'
            logger.error(err)
            raise Exception(err)

        ### Execute
        # Get corresponding data
        if data_source in self.training_data_sources:
            df_data = self.df_features_train_scaled.loc[self.df_features_train_scaled['data_source'] == data_source]
        elif data_source in self.predict_data_sources:
            df_data = self.df_features_predict_scaled.loc[self.df_features_predict_scaled['data_source'] == data_source]
        else:
            err = f'{get_current_function()}(): Data source not found: "{data_source}"'
            logger.error(err)
            raise ValueError(err)

        df_data = df_data.astype({'frame': int}).sort_values('frame')

        svm_assignment_values_array = df_data[self.svm_assignment].values
        labels = list(map(self.get_assignment_label, svm_assignment_values_array))
        frames = list(df_data['frame'].astype(int).values)

        # Generate video with variables
        logger.debug(f'{get_current_function()}(): labels[:5] example = {labels[:5]}')
        logger.debug(f'frames[:5] example: {frames[:5]}')
        videoprocessing.make_labeled_video_according_to_frame(
            labels,
            frames,
            video_name,
            video_to_be_labeled_path,
            output_fps=output_fps,
            output_dir=output_dir,
        )

        return self

    def make_behaviour_example_videos(self, data_source: str, video_file_path: str, file_name_prefix=None,
                                      min_rows_of_behaviour=1, max_examples=3, num_frames_buffer=0, output_fps=15):
        """
        Create video clips of behaviours

        :param data_source: (str)
        :param video_file_path: (str)
        :param file_name_prefix:
        :param min_rows_of_behaviour:
        :param max_examples:
        :return:
        """
        # Args checking
        check_arg.ensure_type(num_frames_buffer, int)
        check_arg.ensure_is_file(video_file_path)
        check_arg.ensure_type(output_fps, int, float)
        # Solve kwargs
        if file_name_prefix is None:
            file_name_prefix = ''
        else:
            check_arg.ensure_type(file_name_prefix, str)
            check_arg.ensure_has_valid_chars_for_path(file_name_prefix)
            file_name_prefix += '__'  # TODO: low: remove hidden formatting here?

        # Get data from data source name
        if data_source in self.training_data_sources:
            df = self.df_features_train_scaled
        elif data_source in self.predict_data_sources:
            df = self.df_features_predict_scaled
        else:
            err = f'Data source not found: {data_source}'
            logger.error(err)
            raise KeyError(err)
        logger.debug(f'{get_current_function()}(): Total records: {len(df)}')

        ### Execute
        # Get DataFrame of the data
        df = df.loc[df["data_source"] == data_source].astype({'frame': int}).sort_values('frame').copy()

        # Get Run-Length Encoding of assignments
        assignments = df[self.svm_assignment_col_name].values
        rle: Tuple[List, List, List] = statistics.augmented_runlength_encoding(assignments)

        # Zip RLE according to order
        # First index is value, second is index, third is *additional* length of value occurrence in sequence.
        # EXAMPLE RESULT: Zip'd RLE according to order: [[15, 0, 0], [4, 1, 1], [14, 3, 0], [15, 4, 0], ... ]
        rle_zipped_by_entry = []
        for row__assignment_idx_addedLength in zip(*rle):
            rle_zipped_by_entry.append(list(row__assignment_idx_addedLength))

        # Roll up assignments into a dict. Keys are labels, values are lists of [index, additional length]
        rle_by_assignment: Dict[Any: List[
            int, int]] = {}  # Dict[Any: List[int, int]] // First element in list is the frame index, second element is the additional length duration of behaviour
        for label, frame_idx, additional_length in rle_zipped_by_entry:
            if label not in rle_by_assignment:
                rle_by_assignment[label] = []
            if additional_length >= min_rows_of_behaviour - 1:
                rle_by_assignment[label].append([frame_idx, additional_length])
        # Sort from longest additional length (ostensibly the duration of behaviour) to least
        for assignment_val in rle_by_assignment.keys():
            rle_by_assignment[assignment_val] = sorted(rle_by_assignment[assignment_val], key=lambda x: x[1],
                                                       reverse=True)

        ### Finally: make video clips
        # Loop over assignments
        for assignment_val, values_list in rle_by_assignment.items():
            # Loop over examples
            num_examples = min(max_examples, len(values_list))
            for i in range(
                    num_examples):  # TODO: HIGH: this part dumbly loops over first n examples...In the future, it would be better to ensure that at least one of the examples has a long runtime for analysis
                output_file_name = f'{file_name_prefix}{time.strftime("%y-%m-%d_%Hh%Mm")}_' \
                                   f'BehaviourExample__assignment_{assignment_val}__example_{i + 1}_of_{num_examples}'
                frame_text_prefix = f'Target assignment: {assignment_val} / '  # TODO: med/high: magic variable

                frame_idx, additional_length_i = values_list[
                    i]  # Recall: first elem is frame idx, second elem is additional length

                lower_bound_row_idx: int = max(0, int(frame_idx) - num_frames_buffer)
                upper_bound_row_idx: int = min(len(df) - 1, frame_idx + additional_length_i - 1 + num_frames_buffer)
                df_frames_selection = df.iloc[lower_bound_row_idx:upper_bound_row_idx, :]

                # Compile labels list via SVM assignment for now...Later, we should get the actual behavioural labels instead of the numerical assignments
                logger.debug(f'df_frames_selection["frame"].dypes.dtypes: {df_frames_selection["frame"].dtypes}')
                assignments_list = list(df_frames_selection[self.svm_assignment_col_name].values)
                current_behaviour_list = [self.get_assignment_label(a) for a in assignments_list]
                frames_indices_list = list(df_frames_selection['frame'].astype(int).values)
                color_map_array = visuals.generate_color_map(len(self.unique_assignments))
                text_colors_list: List[Tuple[float]] = [tuple(float(min(255. * x, 255.))  # Multiply the 3 values by 255 since existing values are on a 0 to 1 scale
                                                              for x in tuple(color_map_array[a][:3]))  # Takes only the first 3 elements since the 4th appears to be brightness value?
                                                        for a in assignments_list]

                #
                videoprocessing.make_labeled_video_according_to_frame(
                    assignments_list,
                    frames_indices_list,
                    output_file_name,
                    video_file_path,
                    current_behaviour_list=current_behaviour_list,
                    text_prefix=frame_text_prefix,
                    output_fps=output_fps,
                    output_dir=config.EXAMPLE_VIDEOS_OUTPUT_PATH,
                    text_colors_list=text_colors_list,
                )

        return self

    # Diagnostics and graphs
    def get_plot_svm_assignments_distribution(self) -> Tuple[object, object]:
        """
        Get a histogram of assignments in order to review their distribution in the TRAINING data
        """
        fig, ax = visuals.plot_assignment_distribution_histogram(
            self.df_features_train_scaled[self.svm_assignment_col_name])
        return fig, ax

    def plot_assignments_in_3d(self, show_now=False, save_to_file=False, azim_elev=(70, 135), **kwargs) -> Tuple[object, object]:
        """
        TODO: expand
        :param show_now:
        :param save_to_file:
        :param azim_elev:
        :return:
        """
        # TODO: low: check for other runtime vars
        if not self.is_built:  # or not self._has_unused_raw_data:
            e = f'{get_current_function()}(): The model has not been built. There is nothing to graph.'
            logger.warning(e)
            raise ValueError(e)

        fig, ax = visuals.plot_GM_assignments_in_3d_tuple(
            self.df_features_train_scaled[self.dims_cols_names].values,
            self.df_features_train_scaled[self.gmm_assignment_col_name].values,
            save_to_file,
            show_now=show_now,
            azim_elev=azim_elev,
            **kwargs
        )
        return fig, ax

    def get_plot_cross_val_scoring(self) -> Tuple[object, object]:
        # TODO: med: confirm that this works as expected
        return visuals.plot_cross_validation_scores(self._cross_val_scores)

    def diagnostics(self) -> str:
        """ Function for displaying current state of pipeline. Useful for diagnostics. """
        diag = f"""
self.is_built: {self.is_built}
unique sources in df_train GMM ASSIGNMENTS: {len(np.unique(self.df_features_train[self.gmm_assignment_col_name].values))}
unique sources in df_train SVM ASSIGNMENTS: {len(np.unique(self.df_features_train[self.svm_assignment_col_name].values))}
self._is_training_data_set_different_from_model_input: {self._is_training_data_set_different_from_model_input}
""".strip()
        return diag

    #
    def __repr__(self) -> str:
        # TODO: low: flesh out how these are usually built. Add a last updated info?
        return f'{self.name}'


def generate_pipeline_filename(name: str):
    """
    Generates a pipeline file name given its name.

    This is an effort to standardize naming for saving pipelines.
    """
    file_name = f'{name}.pipeline'
    return file_name


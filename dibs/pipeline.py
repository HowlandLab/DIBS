"""

For ever new pipeline implementation by the user, make sure you do the following:
    - Use BasicPipeline as the parent object
    - Implement `engineer_features()` using the given interface


Notes
    - the OpenTSNE implementation does not allow more than 2 components
    - GMM's "reg covar" == "regularization covariance"
TODO:
    med/high: review use of UMAP -- potential alternative to SVC?
    med/high: review use of HDBSCAN -- possible replacement for GMM clustering?
    low: implement ACTUAL random state s.t. all random state property calls beget a truly random integer
    low: review "theta"(/angle) for TSNE

Add attrib checking for engineer_features? https://duckduckgo.com/?t=ffab&q=get+all+classes+within+a+file+python&ia=web&iax=qa


"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle as sklearn_shuffle_dataframe
from typing import Any, Collection, Dict, List, Tuple  # TODO: med: review all uses of Optional
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import sys
import time

# from bhtsne import tsne as TSNE_bhtsne
# from pandas.core.common import SettingWithCopyWarning
# from tqdm import tqdm
# import openTSNE  # openTSNE only supports n_components 2 or less
# import warnings

from dibs.logging_dibs import get_current_function
from dibs import check_arg, config, feature_engineering, io, statistics, videoprocessing, visuals

logger = config.initialize_logger(__file__)


# Base pipeline objects that outline the API

class PipelineAttributeHolder(object):
    """
    Helps hide params from base Pipeline object for API clarity
    Implement setters and getters.
    """
    # Base information
    _name, _description = 'DefaultPipelineName', '(Default pipeline description)'
    # data_ext: str = 'csv'  # Extension which data is read from  # TODO: deprecate, delete line
    dims_cols_names = None  # Union[List[str], Tuple[str]]
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
    input_videos_fps = config.VIDEO_FPS  # TODO: remove default as config?
    cross_validation_k: int = config.CROSSVALIDATION_K  # TODO remove default as config?
    _random_state: int = None
    average_over_n_frames: int = 3  # TODO: low: add to kwargs? Address later.  TODO: change to `n_rows_to_integrate_by`
    test_train_split_pct: float = None

    # Model objects
    _scaler: StandardScaler = None
    _clf_gmm: GaussianMixture = None

    # TSNE
    tsne_source: str = 'sklearn'
    tsne_n_components: int = 3
    tsne_n_iter: int = None
    tsne_early_exaggeration: float = None
    tsne_n_jobs: int = None  # n cores used during process
    tsne_verbose: int = None
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
    features_which_average_by_mean = ['DistFrontPawsTailbaseRelativeBodyLength',
                                      'DistBackPawsBaseTailRelativeBodyLength', 'InterforepawDistance', 'BodyLength', ]
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


class BasePipeline(PipelineAttributeHolder):
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

    # Init
    def __init__(self, name: str, **kwargs):
        # Pipeline name
        check_arg.ensure_type(name, str)
        self.set_name(name)

        # TSNE source  # TODO: HIGH: move this section to set_params
        tsne_source = kwargs.get('tsne_source', '')
        check_arg.ensure_type(tsne_source, str)
        if tsne_source in self.valid_tsne_sources:
            self.tsne_source = tsne_source
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
        video_fps = kwargs.get('input_videos_fps',
                               config.VIDEO_FPS if read_config_on_missing_param else self.input_videos_fps)
        check_arg.ensure_type(video_fps, int, float)
        self.input_videos_fps = video_fps
        average_over_n_frames = kwargs.get('average_over_n_frames',
                                           self.average_over_n_frames)  # TODO: low: add a default option for this in config.ini+config.py
        check_arg.ensure_type(average_over_n_frames, int)
        self.average_over_n_frames = average_over_n_frames
        # TODO: low ensure random state correct
        random_state = kwargs.get('random_state',
                                  config.RANDOM_STATE if read_config_on_missing_param else self.random_state)
        check_arg.ensure_type(random_state, int)
        self._random_state = random_state
        ### TSNE ###
        # TODO: add `tsne_source`?
        tsne_n_components = kwargs.get('tsne_n_components',
                                       config.TSNE_N_COMPONENTS if read_config_on_missing_param else self.tsne_n_components)  # TODO: low: shape up kwarg name for n components? See string name
        check_arg.ensure_type(tsne_n_components, int)
        self.tsne_n_components = tsne_n_components
        tsne_n_iter = kwargs.get('tsne_n_iter',
                                 config.TSNE_N_ITER if read_config_on_missing_param else self.tsne_n_iter)
        check_arg.ensure_type(tsne_n_iter, int)
        self.tsne_n_iter = tsne_n_iter
        tsne_early_exaggeration = kwargs.get('tsne_early_exaggeration',
                                             config.TSNE_EARLY_EXAGGERATION if read_config_on_missing_param else self.tsne_early_exaggeration)
        check_arg.ensure_type(tsne_early_exaggeration, float)
        self.tsne_early_exaggeration = tsne_early_exaggeration
        n_jobs = kwargs.get('tsne_n_jobs', config.TSNE_N_JOBS if read_config_on_missing_param else self.tsne_n_jobs)
        check_arg.ensure_type(n_jobs, int)
        self.tsne_n_jobs = n_jobs
        tsne_verbose = kwargs.get('tsne_verbose',
                                  config.TSNE_VERBOSE if read_config_on_missing_param else self.tsne_verbose)
        check_arg.ensure_type(tsne_verbose, int)
        self.tsne_verbose = tsne_verbose
        # GMM vars
        gmm_n_components = kwargs.get('gmm_n_components',
                                      config.gmm_n_components if read_config_on_missing_param else self.gmm_n_components)
        check_arg.ensure_type(gmm_n_components, int)
        self.gmm_n_components = gmm_n_components
        gmm_covariance_type = kwargs.get('gmm_covariance_type',
                                         config.gmm_covariance_type if read_config_on_missing_param else self.gmm_covariance_type)
        check_arg.ensure_type(gmm_covariance_type, str)
        self.gmm_covariance_type = gmm_covariance_type
        gmm_tol = kwargs.get('gmm_tol', config.gmm_tol if read_config_on_missing_param else self.gmm_tol)
        check_arg.ensure_type(gmm_tol, float)
        self.gmm_tol = gmm_tol
        gmm_reg_covar = kwargs.get('gmm_reg_covar',
                                   config.gmm_reg_covar if read_config_on_missing_param else self.gmm_reg_covar)
        check_arg.ensure_type(gmm_reg_covar, float)
        self.gmm_reg_covar = gmm_reg_covar
        gmm_max_iter = kwargs.get('gmm_max_iter',
                                  config.gmm_max_iter if read_config_on_missing_param else self.gmm_max_iter)
        check_arg.ensure_type(gmm_max_iter, int)
        self.gmm_max_iter = gmm_max_iter
        gmm_n_init = kwargs.get('gmm_n_init', config.gmm_n_init if read_config_on_missing_param else self.gmm_n_init)
        check_arg.ensure_type(gmm_n_init, int)
        self.gmm_n_init = gmm_n_init
        gmm_init_params = kwargs.get('gmm_init_params',
                                     config.gmm_init_params if read_config_on_missing_param else self.gmm_init_params)
        check_arg.ensure_type(gmm_init_params, str)
        self.gmm_init_params = gmm_init_params
        gmm_verbose = kwargs.get('gmm_verbose',
                                 config.gmm_verbose if read_config_on_missing_param else self.gmm_verbose)
        check_arg.ensure_type(gmm_verbose, int)
        self.gmm_verbose = gmm_verbose
        gmm_verbose_interval = kwargs.get('gmm_verbose_interval',
                                          config.gmm_verbose_interval if read_config_on_missing_param else self.gmm_verbose_interval)
        check_arg.ensure_type(gmm_verbose_interval, int)
        self.gmm_verbose_interval = gmm_verbose_interval
        # Classifier vars
        clf_type = kwargs.get('clf_type', config.DEFAULT_CLASSIFIER if read_config_on_missing_param else self.clf_type)
        self.clf_type = clf_type
        # Random Forest vars
        rf_n_estimators = kwargs.get('rf_n_estimators',
                                     config.rf_n_estimators if read_config_on_missing_param else self.rf_n_estimators)
        check_arg.ensure_type(rf_n_estimators, int)
        self.rf_n_estimators = rf_n_estimators
        # SVM vars
        svm_c = kwargs.get('svm_c', config.svm_c if read_config_on_missing_param else self.svm_c)
        self.svm_c = svm_c
        svm_gamma = kwargs.get('svm_gamma', config.svm_gamma if read_config_on_missing_param else self.svm_gamma)
        self.svm_gamma = svm_gamma
        svm_probability = kwargs.get('svm_probability',
                                     config.svm_probability if read_config_on_missing_param else self.svm_probability)
        self.svm_probability = svm_probability
        svm_verbose = kwargs.get('svm_verbose',
                                 config.svm_verbose if read_config_on_missing_param else self.svm_verbose)
        self.svm_verbose = svm_verbose
        cross_validation_k = kwargs.get('cross_validation_k',
                                        config.CROSSVALIDATION_K if read_config_on_missing_param else self.cross_validation_k)
        check_arg.ensure_type(cross_validation_k, int)
        self.cross_validation_k = cross_validation_k

        # TODO: low/med: add kwargs for parsing test/train split pct
        if self.test_train_split_pct is None:
            self.test_train_split_pct = config.HOLDOUT_PERCENT

        self.dims_cols_names = [f'dim_{d + 1}' for d in
                                range(self.tsne_n_components)]  # TODO: low: encapsulate elsewhere

        self._has_modified_model_variables = True
        return self

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
        """"""
        # TODO: low: ensure function, add tests
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
        Here we are ensuring that the data processing for both training and prediction occurs in the same way.
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
            logger.debug(
                f'{get_current_function()}(): Engineering df feature set {i + 1} of {len(list_dfs_of_raw_data)}')
            df_engineered_features: pd.DataFrame = self.engineer_features(df)
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
        """ TODO
        """
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
        df_scaled_data = self._create_scaled_data(df_features_train, list(features),
                                                  create_new_scaler=create_new_scaler)
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
    def train_tsne_get_dimension_reduced_data(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
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
        # Do
        # TODO: uncommment this bhtsne later since it was originally commented-out because VCC distrib. problems
        # if self.tsne_source == 'bhtsne':
        #     arr_result = TSNE_bhtsne(
        #         data[self.features_names_7],
        #         dimensions=self.tsne_n_components,
        #         perplexity=np.sqrt(len(self.features_names_7)),  # TODO: implement math somewhere else
        #         rand_seed=self.random_state,
        #     )

        if self.tsne_source == 'sklearn':
            # TODO: high: Save the TSNE object
            arr_result = TSNE_sklearn(
                perplexity=np.sqrt(len(data.columns)),
                # Perplexity scales with sqrt, power law  # TODO: encapsulate this later
                learning_rate=max(200, len(data.columns) // 16),  # alpha*eta = n  # TODO: encapsulate this later
                n_components=self.tsne_n_components,
                random_state=self.random_state,
                n_iter=self.tsne_n_iter,
                early_exaggeration=self.tsne_early_exaggeration,
                n_jobs=self.tsne_n_jobs,
                verbose=self.tsne_verbose,
            ).fit_transform(data[list(self.all_features_list)])
        else:
            err = f'Invalid TSNE source type fell through the cracks: {self.tsne_source}'
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

    def train_SVM(self):
        # TODO: HIGH: DEPRECATED. ENSURE EVERYTING WORKS THEN DELETE
        """ Use scaled training data to train SVM classifier """
        df = self.df_features_train_scaled
        # Instantiate SVM object
        self._clf_svm = SVC(
            C=self.svm_c,
            gamma=self.svm_gamma,
            probability=self.svm_probability,
            verbose=self.svm_verbose,
            random_state=self.random_state,
        )
        # Fit SVM to non-test data
        self._clf_svm.fit(
            X=df.loc[~df[self.test_col_name]][list(self.all_features)],  # TODO: too specific
            y=df.loc[~df[self.test_col_name]][self.gmm_assignment_col_name],
        )
        return self

    def train_classifier(self):
        # TODO: finish this function!
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
            X=df.loc[~df[self.test_col_name]][list(self.all_features)],  # TODO: too specific
            y=df.loc[~df[self.test_col_name]][self.gmm_assignment_col_name],
        )
        # Save classifier
        self._classifier = clf

    # Higher level data processing functions
    def tsne_reduce_df_features_train(self):
        arr_tsne_result = self.train_tsne_get_dimension_reduced_data(self.df_features_train)
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
        logger.debug(f'{inspect.stack()[0][3]}(): Start engineering features')
        if reengineer_train_features or self._is_training_data_set_different_from_model_input:
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
        self.train_classifier()  # # self.train_SVM()

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

        df_features_train_scaled_test_data = self.df_features_train_scaled.loc[
            ~self.df_features_train_scaled[self.test_col_name]]
        self._acc_score = accuracy_score(
            y_pred=self.clf.predict(df_features_train_scaled_test_data[list(self.all_features)]),
            y_true=df_features_train_scaled_test_data[self.svm_assignment_col_name].values)
        logger.debug(f'Pipeline train accuracy: {self.accuracy_score}')
        # TODO: low: save the confusion matrix after accuracy score too?

        # Final touches. Save state of pipeline.
        self._is_built = True
        self._is_training_data_set_different_from_model_input = False  # TODO: review these 3 variables
        self._has_modified_model_variables = False
        self._last_built = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
        logger.debug(f'All done with building classifiers/model!')

        return self

    def build_classifier(self, reengineer_train_features: bool = False):
        """ This is the legacy naming. Method kept for backwards compatibility. This function will be deleted later. """
        warn = f'Pipeline.build_classifier(): was called, but this is the legacy name. Instead, use Pipeline.build_model() from now on.'
        logger.warning(warn)
        return self.build_model(reengineer_train_features=reengineer_train_features)

    def generate_predict_data_assignments(self, reengineer_train_data_features: bool = False,
                                          reengineer_predict_features=False):  # TODO: low: rename?
        """ Runs after build(). Using terminology from old implementation. TODO: purpose """
        # TODO: add arg checking for empty predict data?

        # Check that classifiers are built on the training data
        if reengineer_train_data_features or not self.is_built or self.is_in_inconsistent_state:
            self.build_model()

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
        df_shuffled[
            test_data_col_name] = False  # TODO: med: Setting with copy warning occurs on this exact line. is this not how to instantiate it? https://realpython.com/pandas-settingwithcopywarning/
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
        # if output_path_dir is None:
        #     output_path_dir = config.OUTPUT_PATH
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

    def save_to_folder(self, dir: str):  # TODO: med: review
        """  """
        # Arg checking
        if not os.path.isdir(dir):
            not_a_dir_err = f'Argument `dir` = "{dir}" is not a valid directory. Saving pipeline to dir failed.'
            logger.error(not_a_dir_err)
            raise NotADirectoryError(not_a_dir_err)
        # Execute
        # TODO: low: implement execution
        return

    def save_as(self, file_path):  # TODO: med: review
        """

        :param file_path:
        :return:
        """
        # Arg check
        # TODO: implement arg checking
        # Execute
        # TODO: implement
        return

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
        # if not os.path.isfile(video_to_be_labeled_path):
        #     not_a_video_err = f'The video to be labeled is not a valid file/path. ' \
        #                       f'Submitted video path: {video_to_be_labeled_path}. '
        #     logger.error(not_a_video_err)
        #     raise FileNotFoundError(not_a_video_err)

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
        logger.debug(f'{get_current_function()}(): labels indices example: {labels[:5]}')
        logger.debug(f'Frame indices example: {frames[:5]}')
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

        :param data_source:
        :param video_file_path:
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
                text_colors_list: List[Tuple[float]] = [tuple(float(min(255. * x, 255.))
                                                              # Multiply the 3 values by 255 since existing values are on a 0 to 1 scale
                                                              for x in tuple(color_map_array[a][:3]))
                                                        # Takes only the first 3 elements since the 4th appears to be brightness value?
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

    def plot_assignments_in_3d(self, show_now=False, save_to_file=False, azim_elev=(70, 135), **kwargs) -> Tuple[
        object, object]:
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


# Concrete pipeline implementations

class DemoPipeline(BasePipeline):
    """ Demo pipeline used for demonstration on Pipeline usage. Do not implement this into any real projects. """

    def engineer_features(self, data: pd.DataFrame):
        """
        Sample feature engineering function since all
        implementations of BasePipeline must implement this single function.
        """

        logger.debug(f'Engineering features for one data set...')
        logger.debug(f'Done engineering features.')
        return data


class PipelinePrime(BasePipeline):
    """
    First implementation of a full pipeline. Utilizes the 7 original features from the B-SOiD paper.

    """

    def engineer_features(self, in_df) -> pd.DataFrame:
        """

        """
        columns_to_save = ['scorer', 'source', 'file_source', 'data_source', 'frame']
        df = in_df.sort_values('frame').copy()

        # Filter
        df_filtered, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        df_features: pd.DataFrame = feature_engineering.engineer_7_features_dataframe(
            df_filtered, features_names_7=list(self.all_features))

        # Ensure columns don't get dropped by accident
        for col in columns_to_save:
            if col in in_df.columns and col not in df_features.columns:
                df_features[col] = df[col].values

        # Smooth over n-frame windows
        for feature in self.features_which_average_by_mean:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'avg', self.average_over_n_frames)
        # Sum
        for feature in self.features_which_average_by_sum:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'sum', self.average_over_n_frames)

        return df_features

    def engineer_features_all_dfs(self, list_dfs_of_raw_data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        The main function that can build features for BOTH training and prediction data.
        Here we are ensuring that the data processing for both training and prediction occurs in the same way.
        """
        # TODO: MED: these cols really should be saved in
        #  engineer_7_features_dataframe_NOMISSINGDATA(),
        #  but that func can be amended later due to time constraints

        list_dfs_raw_data = list_dfs_of_raw_data

        # Reconcile args
        if isinstance(list_dfs_raw_data, pd.DataFrame):
            list_dfs_raw_data = [list_dfs_raw_data, ]

        check_arg.ensure_type(list_dfs_raw_data, list)

        list_dfs_engineered_features: List[pd.DataFrame] = []
        for df in list_dfs_raw_data:
            df_engineered_features: pd.DataFrame = self.engineer_features(df)
            list_dfs_engineered_features.append(df_engineered_features)

        # # Adaptively filter features
        # dfs_list_adaptively_filtered: List[Tuple[pd.DataFrame, List[float]]] = [feature_engineering.adaptively_filter_dlc_output(df) for df in list_dfs_raw_data]
        #
        # # Engineer features as necessary
        # dfs_features: List[pd.DataFrame] = []
        # for df_i, _ in tqdm(dfs_list_adaptively_filtered, desc='Engineering features...'):
        #     # Save scorer, source values because the current way of engineering features strips out that info.
        #     df_features_i = feature_engineering.engineer_7_features_dataframe_NOMISSINGDATA(df_i, features_names_7=self.features_names_7)
        #     for col in columns_to_save:
        #         if col not in df_features_i.columns and col in df_i.columns:
        #             df_features_i[col] = df_i[col].values
        #     dfs_features.append(df_features_i)
        #
        # # Smooth over n-frame windows
        # for i, df in tqdm(enumerate(dfs_features), desc='Smoothing values over frames...'):
        #     # Mean
        #     for feature in self.features_which_average_by_mean:
        #         dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
        #             df[feature].values, 'avg', self.average_over_n_frames)
        #     # Sum
        #     for feature in self.features_which_average_by_sum:
        #         dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
        #             df[feature].values, 'sum', self.average_over_n_frames)

        # # Aggregate all data
        df_features = pd.concat(list_dfs_engineered_features)

        return df_features


class PipelineEPM(BasePipeline):
    """
    First try implementation for [EPM] (what does the acronym stand for again?) which matches B-SOID specs
    """

    def engineer_features(self, in_df) -> pd.DataFrame:
        """

        """
        check_arg.ensure_type(in_df, pd.DataFrame)
        map_mouse_point_to_config_name = {
            'Head': 'NOSETIP',
            'ForepawLeft': 'FOREPAW_LEFT',
            'ForepawRight': 'FOREPAW_RIGHT',
            'HindpawLeft': 'HINDPAW_LEFT',
            'HindpawRight': 'HINDPAW_RIGHT',
            'Tailbase': 'TAILBASE',
        }

        columns_to_save = ['scorer', 'source', 'file_source', 'data_source', 'frame']
        df = in_df.sort_values('frame').copy()

        # Filter
        df_filtered, _ = feature_engineering.adaptively_filter_dlc_output(df)

        # Engineer features
        df_features: pd.DataFrame = feature_engineering.engineer_7_features_dataframe(
            df_filtered,
            features_names_7=list(self.all_features),
            map_names=map_mouse_point_to_config_name,
        )
        # Ensure columns don't get dropped by accident
        for col in columns_to_save:
            if col in in_df.columns and col not in df_features.columns:
                df_features[col] = df[col].values

        # Smooth over n-frame windows
        for feature in self.features_which_average_by_mean:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'avg', self.average_over_n_frames)
        # Sum
        for feature in self.features_which_average_by_sum:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'sum', self.average_over_n_frames)

        # except Exception as e:
        #     logger.error(f'{df_features.columns} // fail on feature: {feature} // {df_features.head(10).to_string()} //{repr(e)}')
        #     raise e

        return df_features


class PipelineFlex(BasePipeline):

    # TODO: WIP: creating a flexible class to use with streamlit that allows for flexible feature selection

    def engineer_features(self, data: pd.DataFrame):
        # TODO
        return data


class PipelineTim(BasePipeline):
    """

    """
    # Feature names
    intermediate_avg_forepaw = ''
    intermediate_avg_hindpaw = ''
    feat_name_dist_forepawleft_nosetip = 'distForepawLeftNosetip'
    feat_name_dist_forepawright_nosetip = 'distForepawRightNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'distForepawLeftHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'distForepawRightHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'distAvgHindpawNoseTip'
    feat_name_dist_AvgForepaw_NoseTip = 'distAvgForepawNoseTip'
    feat_name_velocity_AvgForepaw = 'velocAvgForepaw'
    _all_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
    )
    n_rows_to_integrate_by: int = 3  # 3 => 3 frames = 100ms capture in a 30fps video. 100ms was used in original paper.

    def engineer_features(self, in_df: pd.DataFrame):
        # TODO: WIP
        """
        # Head dips
        1. d(forepaw left to nose)
        2. d(forepaw right to nose)
        # Rears
        3. d(forepaw left to hindpaw left)
        4. d(forepaw right to hindpaw right)
        5. d(nose to avg hindpaw)
        # Stretch attends
        6. d(avg hindpaw to nose) - same as #5
        7. d(avg forepaw to nose)
        8. v(avgForepaw)

        """
        # Arg Checking
        check_arg.ensure_type(in_df, pd.DataFrame)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = in_df.sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_LEFT', 'NOSETIP',
                                                                             self.feat_name_dist_forepawleft_nosetip,
                                                                             resolve_bodyparts_with_config_ini=True)
        # 2
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_RIGHT', 'NOSETIP',
                                                                             self.feat_name_dist_forepawright_nosetip,
                                                                             resolve_bodyparts_with_config_ini=True)
        # 3
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_LEFT', 'HINDPAW_LEFT',
                                                                             self.feat_name_dist_forepawLeft_hindpawLeft,
                                                                             resolve_bodyparts_with_config_ini=True)
        # 4
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_RIGHT', 'HINDPAW_RIGHT',
                                                                             self.feat_name_dist_forepawRight_hindpawRight,
                                                                             resolve_bodyparts_with_config_ini=True)
        # 5 & 6
        # Get avg forepaw
        # df = feature_engineering.attach_average_forepaw_xy(df)  # TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, 'FOREPAW_LEFT', 'FOREPAW_RIGHT',
                                                            self.intermediate_avg_forepaw,
                                                            resolve_bodyparts_with_config_ini=True)
        # Get avg hindpaw
        # df = feature_engineering.attach_average_hindpaw_xy(df)
        df = feature_engineering.attach_average_bodypart_xy(df, 'HINDPAW_LEFT', 'HINDPAW_RIGHT',
                                                            self.intermediate_avg_hindpaw,
                                                            resolve_bodyparts_with_config_ini=True)

        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_avg_hindpaw,
                                                                             config.get_part('NOSETIP'),
                                                                             self.feat_name_dist_AvgHindpaw_Nosetip)
        # 7
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_avg_forepaw,
                                                                             config.get_part('NOSETIP'),
                                                                             self.feat_name_dist_AvgForepaw_NoseTip)
        # 8
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_avg_forepaw,
                                                                     1 / config.VIDEO_FPS,
                                                                     output_feature_name=self.feat_name_velocity_AvgForepaw)

        # Binning
        map_feature_to_integrate_method = {
            self.feat_name_dist_forepawleft_nosetip: 'avg',
            self.feat_name_dist_forepawright_nosetip: 'avg',
            self.feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.feat_name_velocity_AvgForepaw: 'sum',
        }
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method,
                                                                self.n_rows_to_integrate_by)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        # # Debug effort/check: ensure columns don't get dropped by accident
        # for col in in_df.columns:
        #     if col not in list(df.columns):
        #         err_missing_col = f'Missing col should not have been lost in feature engineering but was. ' \
        #                           f'Column = {col}. (df={df.head().to_string()})'  # TODO: low: improve err message
        #         logger.error(err_missing_col)
        #         raise KeyError(err_missing_col)

        logger.debug(f'Done engineering features.')
        return df


class PipelineCHBO(BasePipeline):
    """

    """
    # Feature names
    intermediate_bodypart_avgForepaw = 'AvgForepaw'
    intermediate_bodypart_avgHindpaw = 'AvgHindpaw'
    feat_name_dist_forepawleft_nosetip = 'distForepawLeftNosetip'
    feat_name_dist_forepawright_nosetip = 'distForepawRightNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'distForepawLeftHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'distForepawRightHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'distAvgHindpawNoseTip'
    feat_name_dist_AvgForepaw_NoseTip = 'distAvgForepawNoseTip'
    feat_name_velocity_AvgForepaw = 'velocAvgForepaw'
    _all_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
    )

    def engineer_features(self, in_df: pd.DataFrame):
        # TODO: WIP
        """
        # Head dips
        1. d(forepaw left to nose)
        2. d(forepaw right to nose)
        # Rears
        3. d(forepaw left to hindpaw left)
        4. d(forepaw right to hindpaw right)
        5. d(nose to avg hindpaw)
        # Stretch attends
        6. d(avg hindpaw to nose) - same as #5
        7. d(avg forepaw to nose)
        8. v(avgForepaw)

        """
        # Arg Checking
        check_arg.ensure_type(in_df, pd.DataFrame)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = in_df.sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_LEFT', 'NOSETIP',
                                                                             self.feat_name_dist_forepawleft_nosetip,
                                                                             resolve_bodyparts_with_config_ini=True)
        # 2
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_RIGHT', 'NOSETIP',
                                                                             self.feat_name_dist_forepawright_nosetip,
                                                                             resolve_bodyparts_with_config_ini=True)
        # 3
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_LEFT', 'HINDPAW_LEFT',
                                                                             self.feat_name_dist_forepawLeft_hindpawLeft,
                                                                             resolve_bodyparts_with_config_ini=True)
        # 4
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_RIGHT', 'HINDPAW_RIGHT',
                                                                             self.feat_name_dist_forepawRight_hindpawRight,
                                                                             resolve_bodyparts_with_config_ini=True)
        # 5, 6
        # df = feature_engineering.attach_average_forepaw_xy(df)  # BELOW SOLVES TODO: TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, 'FOREPAW_LEFT', 'FOREPAW_RIGHT',
                                                            output_bodypart=self.intermediate_bodypart_avgForepaw,
                                                            resolve_bodyparts_with_config_ini=True)

        # df = feature_engineering.attach_average_hindpaw_xy(df)  # BELO SOLVES TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, 'HINDPAW_LEFT', 'HINDPAW_RIGHT',
                                                            output_bodypart=self.intermediate_bodypart_avgHindpaw,
                                                            resolve_bodyparts_with_config_ini=True)

        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw,
                                                                             config.get_part('NOSETIP'),
                                                                             self.feat_name_dist_AvgHindpaw_Nosetip)

        # 7
        # df = feature_engineering.attach_distance_between_2_feats(df, 'AvgForepaw', config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw,
                                                                             config.get_part('NOSETIP'),
                                                                             self.feat_name_dist_AvgForepaw_NoseTip)

        # 8
        # df = feature_engineering.attach_velocity_of_feature(df, 'AvgForepaw', 1/config.VIDEO_FPS, self.feat_name_velocity_AvgForepaw)
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_bodypart_avgForepaw,
                                                                     1 / config.VIDEO_FPS,
                                                                     self.feat_name_velocity_AvgForepaw)

        # Binning
        map_feature_to_integrate_method = {
            self.feat_name_dist_forepawleft_nosetip: 'avg',
            self.feat_name_dist_forepawright_nosetip: 'avg',
            self.feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.feat_name_velocity_AvgForepaw: 'sum',
        }
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method,
                                                                self.average_over_n_frames)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        # # Debug effort/check: ensure columns don't get dropped by accident
        # for col in in_df.columns:
        #     if col not in list(df.columns):
        #         err_missing_col = f'Missing col should not have been lost in feature engineering but was. ' \
        #                           f'Column = {col}. (df={df.head().to_string()})'  # TODO: low: improve err message
        #         logger.error(err_missing_col)
        #         raise KeyError(err_missing_col)

        logger.debug(f'Done engineering features.')
        return df


class PipelineMimic(BasePipeline):
    """
    A pipeline implementation for mimicking the B-SOID implementation

    7 Features listed in paper (terms in brackets are cursive and were written in math format. See paper page 12/13):
    1. body length (or "[d_ST]"): distance from snout to base of tail
    2. [d_SF]: distance of front paws to base of tail relative to body length (formally: [d_SF] = [d_ST] - [d_FT],
        where [d_FT] is the distance between front paws and base of tail
    3. [d_SB]: distance of back paws to base of tail relative to body length (formally: [d_SB] = [d_ST] - [d_BT]
    4. Inter-forepaw distance (or "[d_FP]"): the distance between the two front paws

    5. snout speed (or "[v_s]"): the displacement of the snout location over a period of 16ms
    6. base-of-tail speed (or ["v_T"]): the displacement of the base of the tail over a period of 16ms
    7. snout to base-of-tail change in angle:
    """

    # Feature names
    feat_body_length = 'bodyLength'  # 1
    intermediate_bodypart_avgForepaw = 'AvgForepaw'
    intermediate_bodypart_avgHindpaw = 'AvgHindpaw'
    intermediate_dist_avgForepaw_to_tailbase = 'dist_avgForepaw_to_tailbase'  # TODO: rename
    feat_dist_front_paws_to_tailbase_relative_to_body_length = 'dist_front_paws_to_tailbase_relative_to_body_length'  # 2  # TODO: rename
    intermediate_dist_avgHindpaw_to_tailbase = 'dist_avgHindpaw_to_tailbase'  # TODO: rename
    feat_dist_hind_paws_to_tailbase_relative_to_body_length = 'distHindpawsToTailBaseRelativetobodylength'  # 3  # TODO: rename str
    feat_dist_bw_front_paws = 'distBetweenFrontPaws'  # 4
    feat_snout_speed = 'snoutSpeed'  # 5
    feat_tail_base_speed = 'tailSpeed'  # 6
    feat_snout_tail_delta_angle = 'snoutTailAngle'  # 7

    _all_features = (
        feat_body_length,
        feat_dist_front_paws_to_tailbase_relative_to_body_length,
        feat_dist_hind_paws_to_tailbase_relative_to_body_length,
        feat_dist_bw_front_paws,
        feat_snout_speed,
        feat_tail_base_speed,
        feat_snout_tail_delta_angle,
    )

    def engineer_features(self, df: pd.DataFrame):
        """
            7 Features listed in paper (terms in brackets are cursive and were written in math format. See paper page 12/13):
        1. body length (or "[d_ST]"): distance from snout to base of tail
        2. [d_SF]: distance of front paws to base of tail relative to body length (formally: [d_SF] = [d_ST] - [d_FT], where [d_FT] is the distance between front paws and base of tail
        3. [d_SB]: distance of back paws to base of tail relative to body length (formally: [d_SB] = [d_ST] - [d_BT]
        4. Inter-forepaw distance (or "[d_FP]"): the distance between the two front paws

        5. snout speed (or "[v_s]"): the displacement of the snout location over a period of 16ms
        6. base-of-tail speed (or ["v_T"]): the displacement of the base of the tail over a period of 16ms
        7. snout to base-of-tail change in angle:
        """

        check_arg.ensure_type(df, pd.DataFrame)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = df.sort_values('frame').copy()

        # 1 dist snout to tail
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('TAILBASE'),
                                                                             config.get_part('NOSETIP'),
                                                                             self.feat_body_length)

        # 2: Dist FrontPaws to tail relative to body length
        ## 1/3: Get AvgForepaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'),
                                                            config.get_part('FOREPAW_RIGHT'),
                                                            output_bodypart=self.intermediate_bodypart_avgForepaw)
        ## 2/3: Get dist from forepaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw,
                                                                             config.get_part('TAILBASE'),
                                                                             self.intermediate_dist_avgForepaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_front_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[
            self.intermediate_dist_avgForepaw_to_tailbase]

        # 3 Dist back paws to base of tail relative to body length
        ## 1/3: Get AvgHindpaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'),
                                                            config.get_part('HINDPAW_RIGHT'),
                                                            output_bodypart=self.intermediate_bodypart_avgHindpaw)
        ## 2/3: Get dist from hindpaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw,
                                                                             config.get_part('TAILBASE'),
                                                                             output_feature_name=self.intermediate_dist_avgHindpaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_hind_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[
            self.intermediate_dist_avgHindpaw_to_tailbase]

        # 4: distance between 2 front paws
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'),
                                                                             config.get_part('FOREPAW_RIGHT'),
                                                                             self.feat_dist_bw_front_paws)

        # 5: snout speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('NOSETIP'),
                                                                     action_duration=1 / self.input_videos_fps,
                                                                     output_feature_name=self.feat_snout_speed)

        # 6 tail speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('TAILBASE'),
                                                                     action_duration=1 / self.input_videos_fps,
                                                                     output_feature_name=self.feat_tail_base_speed)

        # 7: snout to base of tail change in angle
        df = feature_engineering.attach_angle_between_bodyparts(df, config.get_part('NOSETIP'),
                                                                config.get_part('TAILBASE'),
                                                                self.feat_snout_tail_delta_angle)

        # BINNING #
        map_feature_to_integrate_method = {
            self.feat_body_length: 'avg',
            self.feat_dist_front_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_hind_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_bw_front_paws: 'avg',
            self.feat_snout_speed: 'sum',
            self.feat_tail_base_speed: 'sum',
            self.feat_snout_tail_delta_angle: 'sum',
        }

        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method,
                                                                self.average_over_n_frames)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        return df


### Accessory functions ###

def generate_pipeline_filename(name: str):
    """
    Generates a pipeline file name given its name.

    This is an effort to standardize naming for saving pipelines.
    """
    file_name = f'{name}.pipeline'
    return file_name


def generate_pipeline_filename_from_pipeline(pipeline_obj: BasePipeline) -> str:
    return generate_pipeline_filename(pipeline_obj.name)

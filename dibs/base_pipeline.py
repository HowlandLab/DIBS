"""

If a function or property begins with an underscore, it is likely not meant for a user to be manipulating.
Try finding it's corresponding property/function sans underscore.

Dev notes:

Currently only CSVs are being read-in. Fix this later.

If you see any variables that are hard-coded, it is likely that it was kept
that way in the code here to make explicit which variables are available but not in use.
If there is a need in future to add it as a variable config variable, it can be
implemented as such later.
"""
from bhtsne import tsne as TSNE_bhtsne
from openTSNE import TSNE as OpenTsneObj
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle as sklearn_shuffle_dataframe
from typing import Any, Collection, Dict, List, Optional, Tuple, Union  # TODO: med: review all uses of Optional
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import time
from types import FunctionType


# from pandas.core.common import SettingWithCopyWarning
# from tqdm import tqdm
# import openTSNE  # openTSNE only supports n_components 2 or less
import sys

from dibs.logging_enhanced import get_current_function
from dibs import check_arg, config, io, logging_enhanced, statistics, videoprocessing, visuals

logger = config.initialize_logger(__name__)


# Base pipeline objects that outline the Pipeline API

class BasePipeline(object):
    """
    BasePipeline:
    This class enumerates the basic functions by which each pipeline should adhere.


    Parameters
    ----------
    name : str
        Name of pipeline. Also is the name of the saved pipeline file.


    kwargs
        Kwargs default to pulling in data from config.ini file unless overtly specified to override. See below.
        Nearly every single parameter property is settable by the user with set_params().
    ----------

    tsne_source : { 'SKLEARN', 'BHTSNE', 'OPENTSNE' }
        Specify a TSNE implementation to use for dimensionality reduction.
        Note: the SKLEARN implementation seems to be quite slow despite it's widespread use. If
        time is a limiting factor, try using an alternative.

    classifier_type : {'SVM', 'RANDOMFOREST' }
        Specify a classifier to use.
        Default is 'svm'.
        - 'svm' : Support Vector Machine
        - 'rf' : Random Forest


        # TODO: med: expand on further kwargs
    """

    # Base information
    _name, _description = 'DefaultPipelineName', '(Default pipeline description)'
    # valid_tsne_sources: set = {'bhtsne', 'sklearn', }  # TODO: low: remove this since not used anymore
    # Column names
    gmm_assignment_col_name, clf_assignment_col_name, = 'gmm_assignment', 'classifier_assignment'
    behaviour_col_name = 'behaviour'

    # Pipeline state-tracking variables
    _is_built = False  # Is False until the classifiers are built then changes to True
    _is_training_data_set_different_from_model_input: bool = False  # Changes to True if new training data is added and classifiers not rebuilt.
    _has_unengineered_predict_data: bool = False  # Changes to True if new predict data is added. Changes to False if features are engineered.
    _has_modified_model_variables: bool = False

    # Data
    default_cols = ['frame', 'data_source', 'file_source', ]  # ,  clf_assignment_col_name, gmm_assignment_col_name]
    _df_features_train_raw = pd.DataFrame(columns=default_cols)
    _df_features_train = pd.DataFrame(columns=default_cols)
    _df_features_train_scaled = pd.DataFrame(columns=default_cols)
    _df_features_predict_raw = pd.DataFrame(columns=default_cols)
    _df_features_predict = pd.DataFrame(columns=default_cols)
    _df_features_predict_scaled = pd.DataFrame(columns=default_cols)

    # Other model vars (Rename this)
    video_fps: float = config.VIDEO_FPS
    cross_validation_k: int = config.CROSSVALIDATION_K
    cross_validation_n_jobs: int = config.CROSSVALIDATION_N_JOBS
    _random_state: int = None
    average_over_n_frames: int = 3
    test_train_split_pct: float = config.HOLDOUT_PERCENT

    # Model objects
    _scaler = None
    _clf_gmm: GaussianMixture = None

    # TSNE
    tsne_implementation: str = config.TSNE_IMPLEMENTATION
    tsne_n_components: int = config.TSNE_N_COMPONENTS
    tsne_n_iter: int = config.TSNE_N_ITER
    tsne_early_exaggeration: float = config.TSNE_EARLY_EXAGGERATION
    tsne_n_jobs: int = config.TSNE_N_JOBS  # n cores used during process
    tsne_verbose: int = config.TSNE_VERBOSE
    tsne_init: str = config.TSNE_INIT
    _tsne_perplexity: Union[float, str] = None  # config.TSNE_PERPLEXITY
    tsne_learning_rate: float = config.TSNE_LEARNING_RATE
    # GMM
    gmm_n_components, gmm_covariance_type, gmm_tol, gmm_reg_covar = None, None, None, None
    gmm_max_iter, gmm_n_init, gmm_init_params = None, None, None
    gmm_verbose: int = config.gmm_verbose
    gmm_verbose_interval: int = config.gmm_verbose_interval

    # Classifier
    classifier_type: str = config.DEFAULT_CLASSIFIER
    classifier_verbose: int = config.CLASSIFIER_VERBOSE
    _classifier = None
    # Classifier: SVM
    svm_c, svm_gamma = config.svm_c, config.svm_gamma
    svm_probability, svm_verbose = config.svm_probability, config.svm_verbose
    # Classifier: Random Forest
    rf_n_estimators = config.rf_n_estimators
    rf_n_jobs = config.rf_n_jobs
    rf_verbose = config.rf_verbose

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
    _acc_score: float = -1.
    _cross_val_scores: np.ndarray = np.array([])

    seconds_to_engineer_train_features: float = None
    seconds_to_build: float = -1.

    # TODO: low: create tests for this func below
    def get_assignment_label(self, assignment: int) -> str:
        """
        Get behavioural label according to assignment value (number).
        If a label does not exist for a given assignment, then return empty string.
        """
        try:
            assignment = int(assignment)
        except ValueError:
            err = f'TODO: elaborate error: invalid assignment submitted: "{assignment}"'
            logging_enhanced.log_then_raise(err, logger, ValueError)

        label = getattr(self, f'label_{assignment}', '')

        return label

    def set_label(self, assignment: int, label: str):
        """ Set behavioural label for a given model assignment number/value """
        check_arg.ensure_type(label, str)
        assignment = int(assignment)
        setattr(self, f'label_{assignment}', label)
        return self

    def convert_types(self, df):
        return df.astype({'frame': int, })

    ### Properties & Getters ###
    @property
    def df_features_train_raw(self): return self.convert_types(self._df_features_train_raw)
    @property
    def df_features_train(self): return self.convert_types(self._df_features_train)
    @property
    def df_features_train_scaled(self): return self.convert_types(self._df_features_train_scaled)
    @property
    def df_features_predict_raw(self): return self.convert_types(self._df_features_predict_raw)
    @property
    def df_features_predict(self): return self.convert_types(self._df_features_predict)
    @property
    def df_features_predict_scaled(self): return self.convert_types(self._df_features_predict_scaled)

    @property
    def is_in_inconsistent_state(self):
        """
        Useful for checking if training data has been added/removed from pipeline
        relative to already-compiled model
        """
        return self._is_training_data_set_different_from_model_input \
            or self._has_unengineered_predict_data \
            or self._has_modified_model_variables

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def num_training_data_points(self) -> int:
        df_train = self.df_features_train_scaled
        if len(df_train) == 0:
            return 0
        df_train = df_train.loc[~df_train[list(self.all_features)].isnull().any(axis=1)]
        if self.test_col_name in set(df_train.columns):
            df_train = df_train.loc[~df_train[self.test_col_name]]
        return len(df_train)

    @property
    def tsne_perplexity(self) -> float:
        """
        TODO:
        """
        perplexity = self._tsne_perplexity
        # logger.debug(f'{get_current_function()}(): Perplexity starts as "{perplexity}"')
        if isinstance(perplexity, str):
            check_arg.ensure_valid_perplexity_lambda(perplexity)
            perplexity = eval(perplexity)(self)
            # logger.debug(f'{get_current_function()}(): after perp eval, perplexity is: {perplexity}')
        check_arg.ensure_type(perplexity, float)
        # logger.debug(f'@property.{get_current_function()} returns: {perplexity}')
        return perplexity

    @property
    def tsne_perplexity_relative_to_num_features(self) -> float:
        """
        # TODO: med: evaluate
        :return: (float) perplexity_value_used / number of features
        """
        return self.tsne_perplexity / len(self.all_features)

    @property
    def tsne_perplexity_relative_to_num_data_points(self) -> float:
        """
        # TODO: med: evaluate
        :return: perplexity / number of data points for training
        """
        if self.num_training_data_points == 0:
            logger.warning(f'{logging_enhanced.get_caller_function()}() is calling to get perplexity, '
                           f'but there are zero data points. Returning 0 for TSNE perplexity!')
            return 0
        return self.tsne_perplexity / self.num_training_data_points

    @property
    def clf_gmm(self): return self._clf_gmm

    @property
    def clf(self): return self._classifier

    @property
    def random_state(self): return self._random_state

    @property
    def is_built(self): return self._is_built

    @property
    def has_train_data(self) -> bool: return bool(len(self.df_features_train_raw))

    @property
    def has_test_data(self) -> bool: return bool(len(self.df_features_predict_raw))

    @property
    def accuracy_score(self): return self._acc_score

    @property
    def scaler(self): return self._scaler

    @property
    def svm_col(self) -> str: return self.clf_assignment_col_name

    @property
    def svm_assignment(self) -> str: return self.clf_assignment_col_name

    @property
    def cross_val_scores(self): return self._cross_val_scores

    @property
    def training_data_sources(self) -> List[str]:
        return list(np.unique(self.df_features_train_raw['data_source'].values))

    @property
    def predict_data_sources(self):
        return list(np.unique(self.df_features_predict_raw['data_source'].values))

    @property
    def raw_assignments(self): return self.raw_assignments

    @property
    def unique_assignments(self) -> List[any]:
        if len(self._df_features_train_scaled) > 0:
            return list(np.unique(self._df_features_train_scaled[self.svm_col].values))
        return []

    @property
    def all_features(self) -> Tuple[str]: return self._all_features

    @property
    def all_features_list(self) -> List[str]: return list(self._all_features)

    @property
    def total_build_time(self): return self.seconds_to_engineer_train_features

    @property
    def dims_cols_names(self) -> List[str]:
        """
        Automatically creates a list of consistent column names, relative to the number of
        TSNE components, that labels the columns of reduced data after the TSNE operation.
        """
        return [f'dim_{d+1}' for d in range(self.tsne_n_components)]

    # Init
    def __init__(self, name: str, **kwargs):
        # Pipeline name
        check_arg.ensure_type(name, str)
        self.set_name(name)
        #
        self.kwargs = kwargs
        # Final setup
        self.set_params(read_config_on_missing_param=True, **kwargs)

    # Setters
    def set_name(self, name: str):
        check_arg.ensure_has_valid_chars_for_path(name)
        self._name = name
        return self

    def set_description(self, description):
        """ Set a description of the pipeline. Include any notes you want to keep regarding the process used. """
        check_arg.ensure_type(description, str)
        self._description = description
        return self

    def set_params(self, read_config_on_missing_param: bool = False, **kwargs):
        """
        Reads in variables to change for pipeline.

        If optional arg `read_config_on_missing_param` is True, then any parameter NOT mentioned
        explicitly will be read in from the config.ini file and then replace the current value
        for that property in the pipeline.

        Valid Kwargs include any properties in the BasePipeline that do not begin with an underscore.

        Kwargs
        ----------
        classifier_type : str
            Must be one of { 'SVM', 'RANDOMFOREST' }

        video_fps : int
            Explanation goes here

        random_state : int
            Explanation __

        tsne_n_components : int

        tsne_n_iter : int

        tsne_early_exaggeration : float

        tsne_n_jobs : int

        tsne_verbose : int

        TODO: low: complete list

        """
        check_arg.ensure_type(read_config_on_missing_param, bool)
        ### General Params ###
        # Test/train split %
        test_train_split_pct = kwargs.get('test_train_split_pct', config.HOLDOUT_PERCENT if read_config_on_missing_param else self.test_train_split_pct)
        check_arg.ensure_type(test_train_split_pct, float)
        self.test_train_split_pct = test_train_split_pct
        # Classifier model
        classifier_type = kwargs.get('classifier_type', config.DEFAULT_CLASSIFIER if read_config_on_missing_param else self.classifier_type)
        check_arg.ensure_type(classifier_type, str)
        if classifier_type not in config.valid_classifiers:
            err = f'Input classifier type is not valid. Value = {classifier_type}'
            logger.error(err)
            raise ValueError(err)
        self.classifier_type = classifier_type
        # TODO: MED: ADD KWARGS OPTION FOR OVERRIDING VERBOSE in CONFIG.INI!!!!!!!! ?
        # Source video FPS
        video_fps = kwargs.get('video_fps', config.VIDEO_FPS if read_config_on_missing_param else self.video_fps)
        check_arg.ensure_type(video_fps, int, float)
        self.video_fps = float(video_fps)
        # Window averaging
        average_over_n_frames = kwargs.get('average_over_n_frames', self.average_over_n_frames)  # TODO: low: add a default option for this in config.ini+config.py
        check_arg.ensure_type(average_over_n_frames, int)
        self.average_over_n_frames = average_over_n_frames
        # Random state value
        random_state = kwargs.get('random_state', config.RANDOM_STATE if read_config_on_missing_param else self.random_state)  # TODO: low: ensure random state correct
        check_arg.ensure_type(random_state, int)
        self._random_state = random_state
        ### TSNE ###
        # TSNE implementation type
        tsne_implementation = kwargs.get('tsne_implementation', config.TSNE_IMPLEMENTATION if read_config_on_missing_param else self.tsne_implementation)
        check_arg.ensure_type(tsne_implementation, str)
        self.tsne_implementation = tsne_implementation
        # TSNE Initialization
        tsne_init = kwargs.get('tsne_init', config.TSNE_INIT if read_config_on_missing_param else self.tsne_init)
        check_arg.ensure_type(tsne_init, str)
        self.tsne_init = tsne_init
        # TSNE early exaggeration
        tsne_early_exaggeration = kwargs.get('tsne_early_exaggeration', config.TSNE_EARLY_EXAGGERATION if read_config_on_missing_param else self.tsne_early_exaggeration)
        check_arg.ensure_type(tsne_early_exaggeration, float, int)
        self.tsne_early_exaggeration = float(tsne_early_exaggeration)
        # TSNE learning rate
        tsne_learning_rate = kwargs.get('tsne_learning_rate', config.TSNE_LEARNING_RATE if read_config_on_missing_param else self.tsne_learning_rate)
        check_arg.ensure_type(tsne_learning_rate, float, int)
        self.tsne_learning_rate = float(tsne_learning_rate)
        # TSNE n components
        tsne_n_components = kwargs.get('tsne_n_components', config.TSNE_N_COMPONENTS if read_config_on_missing_param else self.tsne_n_components)  # TODO: low: shape up kwarg name for n components? See string name
        check_arg.ensure_type(tsne_n_components, int)
        self.tsne_n_components = tsne_n_components
        # TSNE n iterations
        tsne_n_iter = kwargs.get('tsne_n_iter', config.TSNE_N_ITER if read_config_on_missing_param else self.tsne_n_iter)
        check_arg.ensure_type(tsne_n_iter, int)
        self.tsne_n_iter = tsne_n_iter
        # TSNE n jobs
        tsne_n_jobs = kwargs.get('tsne_n_jobs', config.TSNE_N_JOBS if read_config_on_missing_param else self.tsne_n_jobs)
        check_arg.ensure_type(tsne_n_jobs, int)
        self.tsne_n_jobs = tsne_n_jobs
        # TSNE perplexity
        tsne_perplexity: Union[float, int, str] = kwargs.get('tsne_perplexity', config.TSNE_PERPLEXITY if read_config_on_missing_param else self._tsne_perplexity)
        if isinstance(tsne_perplexity, str):
            check_arg.ensure_valid_perplexity_lambda(tsne_perplexity)
        check_arg.ensure_type(tsne_perplexity, float, int, str)
        self._tsne_perplexity = float(tsne_perplexity) if isinstance(tsne_perplexity, int) else tsne_perplexity
        # TSNE verbosity
        tsne_verbose = kwargs.get('tsne_verbose', config.TSNE_VERBOSE if read_config_on_missing_param else self.tsne_verbose)
        check_arg.ensure_type(tsne_verbose, int)
        self.tsne_verbose = tsne_verbose
        ### GMM parameters
        # GMM n components
        gmm_n_components = kwargs.get('gmm_n_components', config.gmm_n_components if read_config_on_missing_param else self.gmm_n_components)
        check_arg.ensure_type(gmm_n_components, int)
        self.gmm_n_components = gmm_n_components
        # GMM covariance type
        gmm_covariance_type = kwargs.get('gmm_covariance_type', config.gmm_covariance_type if read_config_on_missing_param else self.gmm_covariance_type)
        check_arg.ensure_type(gmm_covariance_type, str)
        self.gmm_covariance_type = gmm_covariance_type
        # GMM tolerance
        gmm_tol = kwargs.get('gmm_tol', config.gmm_tol if read_config_on_missing_param else self.gmm_tol)
        check_arg.ensure_type(gmm_tol, float)
        self.gmm_tol = gmm_tol
        # TODO: what does reg cover stand for?
        gmm_reg_covar = kwargs.get('gmm_reg_covar', config.gmm_reg_covar if read_config_on_missing_param else self.gmm_reg_covar)
        check_arg.ensure_type(gmm_reg_covar, float, int)
        self.gmm_reg_covar = float(gmm_reg_covar)
        # GMM maximum iterations
        gmm_max_iter = kwargs.get('gmm_max_iter', config.gmm_max_iter if read_config_on_missing_param else self.gmm_max_iter)
        check_arg.ensure_type(gmm_max_iter, int)
        self.gmm_max_iter = gmm_max_iter
        # GMM n init TODO
        gmm_n_init = kwargs.get('gmm_n_init', config.gmm_n_init if read_config_on_missing_param else self.gmm_n_init)
        check_arg.ensure_type(gmm_n_init, int)
        self.gmm_n_init = gmm_n_init
        # GMM initialization parameters
        gmm_init_params = kwargs.get('gmm_init_params', config.gmm_init_params if read_config_on_missing_param else self.gmm_init_params)
        check_arg.ensure_type(gmm_init_params, str)
        self.gmm_init_params = gmm_init_params
        # GMM verbosity
        gmm_verbose = kwargs.get('gmm_verbose', config.gmm_verbose if read_config_on_missing_param else self.gmm_verbose)
        check_arg.ensure_type(gmm_verbose, int)
        self.gmm_verbose = gmm_verbose
        # GMM verbosity interval
        gmm_verbose_interval = kwargs.get('gmm_verbose_interval', config.gmm_verbose_interval if read_config_on_missing_param else self.gmm_verbose_interval)
        check_arg.ensure_type(gmm_verbose_interval, int)
        self.gmm_verbose_interval = gmm_verbose_interval
        # Classifiers
        classifier_verbose = kwargs.get('classifier_verbose', config.CLASSIFIER_VERBOSE if read_config_on_missing_param else self.classifier_verbose)
        check_arg.ensure_type(classifier_verbose, int)
        self.classifier_verbose = classifier_verbose
        ### Random Forest vars
        # Random Forest n estimators
        rf_n_estimators = kwargs.get('rf_n_estimators', config.rf_n_estimators if read_config_on_missing_param else self.rf_n_estimators)
        check_arg.ensure_type(rf_n_estimators, int)
        self.rf_n_estimators = rf_n_estimators
        # Random Forest n jobs
        rf_n_jobs = kwargs.get('rf_n_jobs', config.rf_n_jobs if read_config_on_missing_param else self.rf_n_jobs)
        check_arg.ensure_type(rf_n_jobs, int)
        self.rf_n_jobs = rf_n_jobs
        ### SVM vars
        # SVM C
        svm_c = kwargs.get('svm_c', config.svm_c if read_config_on_missing_param else self.svm_c)
        self.svm_c = svm_c
        # SVM gamma
        svm_gamma = kwargs.get('svm_gamma', config.svm_gamma if read_config_on_missing_param else self.svm_gamma)
        self.svm_gamma = svm_gamma
        # SVM probability
        svm_probability = kwargs.get('svm_probability', config.svm_probability if read_config_on_missing_param else self.svm_probability)
        self.svm_probability = svm_probability
        # SVM verbosity
        svm_verbose = kwargs.get('svm_verbose', config.svm_verbose if read_config_on_missing_param else self.svm_verbose)
        self.svm_verbose = svm_verbose
        # Cross-validation K
        cross_validation_k = kwargs.get('cross_validation_k', config.CROSSVALIDATION_K if read_config_on_missing_param else self.cross_validation_k)
        check_arg.ensure_type(cross_validation_k, int)
        self.cross_validation_k = cross_validation_k
        # Cross-validation n jobs
        cross_validation_n_jobs = kwargs.get('cross_validation_n_jobs', config.CROSSVALIDATION_N_JOBS if read_config_on_missing_param else self.cross_validation_n_jobs)
        check_arg.ensure_type(cross_validation_n_jobs, int)
        self.cross_validation_n_jobs = cross_validation_n_jobs

        self._has_modified_model_variables = True
        return self

    def set_tsne_perplexity_as_fraction_of_training_data(self, fraction: float):
        check_arg.ensure_type(fraction, float)
        if not 0. < fraction <= 1.:
            err = f'TSNE perplexity fraction is not between 0 and 1, and thus is invalid. ' \
                  f'Fraction detected: {fraction} (type: {type(fraction)}).'
            raise ValueError(err)
        self._tsne_perplexity = f'lambda self: self.num_training_data_points * {fraction}'
        check_arg.ensure_valid_perplexity_lambda(self._tsne_perplexity)  # TODO: delete this line later. it's a sanity check.
        return self

    # Important functions that should be overwritten by child classes
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
                self._df_features_train_raw = self.df_features_train_raw.append(df_new_data)
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
                    self._df_features_train_raw = self.df_features_train_raw.append(df_new_data_i)
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
                self._df_features_predict_raw = self.df_features_predict_raw.append(df_new_data)
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
                    self._df_features_predict_raw = self.df_features_predict_raw.append(df_new_data_i)
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
        self._df_features_train_raw = self.df_features_train_raw.loc[
            self.df_features_train_raw['data_source'] != data_source]
        self._df_features_train = self.df_features_train.loc[
            self.df_features_train['data_source'] != data_source]
        self._df_features_train_scaled = self.df_features_train_scaled.loc[
            self.df_features_train_scaled['data_source'] != data_source]

        return self

    def remove_predict_data_source(self, data_source: str):
        """
        Remove data from predicted data set.
        :param data_source: (str) name of a data source
        """
        # TODO: low: ensure function, add tests
        check_arg.ensure_type(data_source, str)
        self._df_features_predict_raw = self.df_features_predict_raw.loc[
            self.df_features_predict_raw['data_source'] != data_source]
        self._df_features_predict = self.df_features_predict.loc[
            self.df_features_predict['data_source'] != data_source]
        self._df_features_predict_scaled = self.df_features_predict_scaled.loc[
            self.df_features_predict_scaled['data_source'] != data_source]
        return self

    # Engineer features
    def _engineer_features_all_dfs(self, list_dfs_of_raw_data: List[pd.DataFrame]) -> pd.DataFrame:
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

        # Reconcile args
        check_arg.ensure_type(list_dfs_of_raw_data, list)

        # Execute
        # TODO: MED: implement multiprocessing
        list_dfs_engineered_features: List[pd.DataFrame] = []
        for i, df_i in enumerate(list_dfs_of_raw_data):
            df_i = df_i.copy().astype({'frame': float}).astype({'frame': int})
            check_arg.ensure_frame_indices_are_integers(df_i)
            logger.debug(f'{get_current_function()}(): Engineering df feature set {i+1} of {len(list_dfs_of_raw_data)}')
            df_engineered_features: pd.DataFrame = self.engineer_features(df_i)
            df_engineered_features = df_engineered_features.astype({feature: float for feature in self.all_features})
            list_dfs_engineered_features.append(df_engineered_features)

        # Aggregate all data into one DataFrame, return
        df_features = pd.concat(list_dfs_engineered_features)

        return df_features

    def _engineer_features_train(self):
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
        df_features = self._engineer_features_all_dfs(list_dfs_raw_data)
        logger.debug(f'Done engineering training data features.')
        # Save data
        self._df_features_train = df_features
        # Wrap up
        end = time.perf_counter()
        self._is_training_data_set_different_from_model_input = False
        self.seconds_to_engineer_train_features = round(end - start, 1)
        return self

    def _engineer_features_predict(self):
        """ Engineer features for the predicted data"""
        # Queue data
        list_dfs_raw_data = [self.df_features_predict_raw.loc[self.df_features_predict_raw['data_source'] == src]
                                 .sort_values('frame').copy()
                             for src in set(self.df_features_predict_raw['data_source'].values)]
        # Call engineering function
        logger.debug(f'Start engineering predict data features.')
        df_features = self._engineer_features_all_dfs(list_dfs_raw_data)
        logger.debug(f'Done engineering predict data features.')
        # Save data, return
        self._df_features_predict = df_features
        self._has_unengineered_predict_data = False
        return self

    ## Scaling data
    def _create_scaled_data(self, df_data, features, create_new_scaler: bool = False) -> pd.DataFrame:
        """
        A universal data scaling function that is usable for training data as well as new prediction data.
        Scales down features in place and does not keep original data.

        More notes on scaling, choosing b/w Standardization and Normalization: https://kharshit.github.io/blog/2018/03/23/scaling-vs-normalization
        """
        # Check args
        check_arg.ensure_type(features, list, tuple)
        check_arg.ensure_columns_in_DataFrame(df_data, features)
        # Execute
        if create_new_scaler:
            # self._scaler = StandardScaler()
            self._scaler = MinMaxScaler()
            self._scaler.fit(df_data[features])
        arr_data_scaled: np.ndarray = self.scaler.transform(df_data[features])
        df_scaled_data = pd.DataFrame(arr_data_scaled, columns=features)

        # For new DataFrame, replace columns that were not scaled so that data does not go missing
        for col in df_data.columns:
            if col not in set(df_scaled_data.columns):
                df_scaled_data[col] = df_data[col].values

        return df_scaled_data

    def _scale_transform_train_data(self, features: Collection[str] = None, create_new_scaler=True):
        """
        Scales training data. By default, creates new scaler according to train
        data and stores it in pipeline
        Note: this function implicitly filters out NAN results that may be present in the features set.
        The final result is that all scaled training data will be valid to put into the classifier
        :param features: (List[str]) List of feature names (column names)
        :param create_new_scaler: (bool)
        :return: self
        """
        # Queue up data to use
        if features is None:  # TODO: low: remove his if statement as a default feature?
            features = self.all_features
        features = list(features)
        df_features_train = self.df_features_train.copy()
        df_features_train = df_features_train.loc[~df_features_train[features].isnull().any(axis=1)]
        # Check args
        check_arg.ensure_type(features, list)
        check_arg.ensure_columns_in_DataFrame(df_features_train, features)
        # Get scaled data
        df_features_train_scaled = self._create_scaled_data(df_features_train, features, create_new_scaler=create_new_scaler)
        check_arg.ensure_type(df_features_train_scaled, pd.DataFrame)  # Debugging effort. Remove later.
        # Save data. Return.
        self._df_features_train_scaled = df_features_train_scaled
        return self

    def _scale_transform_predict_data(self, features: List[str] = None):
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
        check_arg.ensure_type(features, list)
        check_arg.ensure_type(df_features_predict, pd.DataFrame)
        check_arg.ensure_columns_in_DataFrame(df_features_predict, features)

        # Get scaled data
        df_scaled_data: pd.DataFrame = self._create_scaled_data(df_features_predict, features, create_new_scaler=False)

        # Save data. Return.
        self._df_features_predict_scaled = df_scaled_data
        return self

    # TSNE Transformations
    def _train_tsne_get_dimension_reduced_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        TODO: elaborate
        TODO: ensure that TSNE obj can be saved and used later for new data? *** Important ***
        :param data:
        :return:
        """
        # Check args
        check_arg.ensure_type(data, pd.DataFrame)
        check_arg.ensure_columns_in_DataFrame(data, self.all_features_list)
        logger.info(f'pre-TSNE info: Perplexity={self.tsne_perplexity} / numtrainingdatapoints={self.num_training_data_points} / number of df_features_train: {len(self.df_features_train)} / number of df_features_train_scaled={len(self.df_features_train_scaled)}')
        # Execute
        start = time.perf_counter()
        logger.debug(f'Now reducing data with {self.tsne_implementation} implementation...')
        if self.tsne_implementation == 'SKLEARN':
            arr_result = TSNE_sklearn(
                perplexity=self.tsne_perplexity,
                learning_rate=self.tsne_learning_rate,  # alpha*eta = n  # TODO: low: encapsulate this later                     !!!
                n_components=self.tsne_n_components,
                random_state=self.random_state,
                n_iter=self.tsne_n_iter,
                early_exaggeration=self.tsne_early_exaggeration,
                n_jobs=self.tsne_n_jobs,
                verbose=self.tsne_verbose,
                init=self.tsne_init,
            ).fit_transform(data[list(self.all_features_list)])
        elif self.tsne_implementation == 'BHTSNE':
            arr_result = TSNE_bhtsne(
                data[list(self.all_features)],
                dimensions=self.tsne_n_components,
                perplexity=self.tsne_perplexity,
                theta=0.5,
                rand_seed=self.random_state,
            )
        elif self.tsne_implementation == 'OPENTSNE':
            tsne = OpenTsneObj(
                n_components=self.tsne_n_components,
                perplexity=self.tsne_perplexity,
                learning_rate=self.tsne_learning_rate,
                early_exaggeration=self.tsne_early_exaggeration,
                early_exaggeration_iter=250,  # TODO: med: review
                n_iter=self.tsne_n_iter,
                # exaggeration=None,
                # dof=1,
                # theta=0.5,
                # n_interpolation_points=3,
                # min_num_intervals=50,
                # ints_in_interval=1,
                # initialization="pca",
                metric="euclidean",  # TODO: med: review
                # metric_params=None,
                # initial_momentum=0.5,
                # final_momentum=0.8,
                # max_grad_norm=None,
                # max_step_norm=5,
                n_jobs=self.tsne_n_jobs,
                # affinities=None,
                # neighbors="auto",
                negative_gradient_method='bh',  # Note: default 'fft' does not work with dims >2
                # callbacks=None,
                # callbacks_every_iters=50,
                random_state=self.random_state,
                verbose=bool(self.tsne_verbose),
            )
            arr_result = tsne.fit(data[list(self.all_features)].values)
        else:
            err = f'Invalid TSNE source type fell through the cracks: {self.tsne_implementation}'
            logger.error(err)
            raise RuntimeError(err)
        logger.debug(f'Number of seconds it took to train TSNE: {round(time.perf_counter() - start, 1)}')
        return arr_result

    def _tsne_reduce_df_features_train(self):
        """
        Attach new reduced dimension columns to existing (scaled) features DataFrame
        :return: self
        """
        # arr_tsne_result: np.ndarray = self._train_tsne_get_dimension_reduced_data(self.df_features_train)  # TODO: review this and below lines
        arr_tsne_result: np.ndarray = self._train_tsne_get_dimension_reduced_data(self.df_features_train_scaled)
        # Attach dimensionally reduced data
        self._df_features_train_scaled = pd.concat([
            self.df_features_train_scaled,
            pd.DataFrame(arr_tsne_result, columns=self.dims_cols_names),
        ], axis=1)

        return self

    # GMM
    def _train_gmm_and_classifier(self, n_clusters: int = None):
        """

        :return:
        """
        if n_clusters is not None:
            self.set_params(gmm_n_components=n_clusters)

        # Train GMM, get assignments
        logger.debug(f'Training GMM now...')
        data = self.df_features_train_scaled[self.dims_cols_names].values
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
        self._df_features_train_scaled[self.gmm_assignment_col_name] = self.clf_gmm.predict(self.df_features_train_scaled[self.dims_cols_names].values)

        # Test-train split
        self._add_test_data_column_to_scaled_train_data()

        # # Train Classifier
        self._train_classifier()

        # Set predictions
        # self._df_features_train_scaled[self.clf_assignment_col_name] = self.clf.predict(self.df_features_train_scaled[list(self.all_features)].values)  # Get predictions
        self._df_features_train_scaled[self.clf_assignment_col_name] = self.clf_predict(self.df_features_train_scaled[list(self.all_features)].values)  # Get predictions
        self._df_features_train_scaled[self.clf_assignment_col_name] = self.df_features_train_scaled[self.clf_assignment_col_name].astype(int)  # Coerce into int

        return self

    def clf_predict(self, arr: np.ndarray):
        # TODO: low/med: add checks for NULL values in array
        return self.clf.predict(arr)

    def recolor_gmm_and_retrain_classifier(self, n_components: int):
        self._train_gmm_and_classifier(n_components)
        return self

    # Classifier
    def _train_classifier(self):
        """
        Train classifier on non-test-assigned data from the training data set.
        For any kwarg that does not take it's value from the it's own Pipeline config., then that
        variable
        """
        df = self.df_features_train_scaled
        df = df.loc[(~df[self.test_col_name]) & (~df[list(self.all_features)].isnull().any(axis=1))]
        if self.classifier_type == 'SVM':
            clf = SVC(
                C=self.svm_c,
                gamma=self.svm_gamma,
                probability=self.svm_probability,
                verbose=bool(self.classifier_verbose),
                random_state=self.random_state,
                cache_size=200,  # TODO: LOW: add variable to CONFIG.INI later? Measured in MB.
                max_iter=-1,
            )
        elif self.classifier_type == 'RANDOMFOREST':
            clf = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                bootstrap=True,
                oob_score=False,
                n_jobs=self.rf_n_jobs,
                random_state=self.random_state,
                verbose=self.rf_verbose,
                warm_start=False,
                class_weight=None,
                ccp_alpha=0.0,
                max_samples=None,
            )
        else:
            err = f'TODO: elaborate: an unexpected classifier type was detected: {self.classifier_type}'
            logging_enhanced.log_then_raise(err, logger, KeyError)

        # Fit classifier to non-test data
        logger.debug(f'Training {self.classifier_type} classifier now...')
        clf.fit(
            X=df[list(self.all_features)],
            y=df[self.gmm_assignment_col_name],
        )
        # Save classifier
        self._classifier = clf

    def _generate_accuracy_scores(self):
        """

        :return:
        """
        df = self.df_features_train_scaled
        logger.debug(f'Generating cross-validation scores...')
        # # Get cross-val accuracy scores
        self._cross_val_scores = cross_val_score(
            self.clf,
            df[self.all_features_list].values,
            df[self.clf_assignment_col_name].values,
            cv=self.cross_validation_k,
            n_jobs=self.cross_validation_n_jobs,
            pre_dispatch=self.cross_validation_n_jobs,
        )

        # logger.debug(f'Generating accuracy score with a test split % of: {self.test_train_split_pct*100}%')
        df_features_train_scaled_test_data = df.loc[df[self.test_col_name]]
        self._acc_score = accuracy_score(
            y_pred=self.clf_predict(df_features_train_scaled_test_data[list(self.all_features)]),
            y_true=df_features_train_scaled_test_data[self.clf_assignment_col_name].values)
        logger.debug(f'Pipeline train accuracy: {self.accuracy_score}')
        return self

    def _generate_predict_data_assignments(self, reengineer_train_data_features: bool = False, reengineer_predict_features=False):  # TODO: low: rename?
        """
        Runs after build(). Using terminology from old implementation. TODO: purpose
        """

        # Check that classifiers are built on the training data
        if reengineer_train_data_features or self._is_training_data_set_different_from_model_input or self._has_modified_model_variables:
            self._build_pipeline()

        # TODO: temp exit early for zero test data found
        if len(self.df_features_predict_raw) == 0:
            warn = f'Zero test data points found. Exiting early. predict features not built.'
            logger.warning(warn)
            return self

        # Check if predict features have been engineered
        # if reengineer_predict_features or self._has_unengineered_predict_data:
        self._engineer_features_predict()
        self._scale_transform_predict_data()

        # Add prediction labels
        if len(self.df_features_predict_scaled) > 0:
            self.df_features_predict_scaled[self.clf_assignment_col_name] = self.clf_predict(self.df_features_predict_scaled[list(self.all_features)].values)
        else:
            logger.debug(f'{get_current_function()}(): 0 records were detected '
                         f'for PREDICT data. No data was predicted with model.')

        return self

    # Pipeline building
    def _build_pipeline(self, force_reengineer_train_features: bool = False, skip_cross_val_scoring: bool = False):
        """
        Builds the model for predicting behaviours.
        :param force_reengineer_train_features: (bool) If True, forces the training data to be re-engineered.
        """
        # Engineer features
        if force_reengineer_train_features or self._is_training_data_set_different_from_model_input:
            logger.debug(f'{inspect.stack()[0][3]}(): Start engineering features...')
            self._engineer_features_train()

        # Scale data
        logger.debug(f'Scaling data now...')
        self._scale_transform_train_data(create_new_scaler=True)

        # TSNE -- create new dimensionally reduced data
        logger.debug(f'TSNE reducing features now...')
        self._tsne_reduce_df_features_train()

        # GMM + Classifier
        self._train_gmm_and_classifier()

        if skip_cross_val_scoring:
            logger.debug(f'Skipping cross-validation scoring...')
        else:
            self._generate_accuracy_scores()

        # Final touches. Save state of pipeline.
        self._is_built = True
        self._is_training_data_set_different_from_model_input = False
        self._has_modified_model_variables = False
        self._last_built = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
        logger.debug(f'{get_current_function()}(): All done with building classifiers/model!')

        return self

    def build(self, force_reengineer_train_features=False, reengineer_predict_features=False):
        """
        Encapsulate entire build process from front to back.
        This included transforming training data, predict data, training classifiers, and getting all results.
        """
        start = time.perf_counter()
        # Build model
        self._build_pipeline(force_reengineer_train_features=force_reengineer_train_features)
        # Get predict data
        self._generate_predict_data_assignments(reengineer_predict_features=reengineer_predict_features)
        # Wrap up
        end = time.perf_counter()
        self.seconds_to_build = round(end - start, 2)
        logger.info(f'Total build time: {self.seconds_to_build} seconds. Rows of data: {len(self._df_features_train_scaled)} / tsne_n_jobs={self.tsne_n_jobs} / cross_validation_n_jobs = {self.cross_validation_n_jobs}')  # TODO: med: amend this line later. Has extra info for debugging purposes.
        return self

    # More data transformations
    def _add_test_data_column_to_scaled_train_data(self):
        """
        Add boolean column to scaled training data DataFrame to assign train/test data
        """
        test_data_col_name = self.test_col_name
        check_arg.ensure_type(test_data_col_name, str)

        df = self.df_features_train_scaled
        df[self.test_col_name] = False
        df_shuffled = sklearn_shuffle_dataframe(df)  # Shuffles data, loses none in the process. Assign bool according to random assortment.
        # TODO: med: fix setting with copy warning
        df_shuffled.iloc[:round(len(df_shuffled) * self.test_train_split_pct), :][test_data_col_name] = True  # Setting copy with warning: https://realpython.com/pandas-settingwithcopywarning/

        df_shuffled = df_shuffled.sort_values(['data_source', 'frame'])

        actual_split = round(len(df_shuffled.loc[df_shuffled[self.test_col_name]]) / len(df_shuffled), 3)
        logger.debug(f"Final test/train split is calculated to be: {actual_split}")

        self._df_features_train_scaled = df_shuffled
        return self

    # Saving and stuff
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

    def save_as(self, out_path):
        """

        :param out_path:
        :return:
        """
        # Arg check
        check_arg.ensure_is_valid_path(out_path)
        check_arg.ensure_is_dir(os.path.split(out_path))
        # Execute
        # TODO: MED
        # Return pipeline that was saved-as
        return io.read_pipeline(out_path)

    # Video creation
    def make_video(self, video_to_be_labeled_path: str, data_source: str, video_name: str, output_dir: str, output_fps: float = config.OUTPUT_VIDEO_FPS):
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

        df_data = df_data.astype({'frame': float}).astype({'frame': int}).sort_values('frame')

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

    def make_behaviour_example_videos(self, data_source: str, video_file_path: str, file_name_prefix=None, min_rows_of_behaviour=1, max_examples=3, num_frames_buffer=0, output_fps=15):
        """
        Create video clips of behaviours

        :param data_source: (str)
        :param video_file_path: (str)
        :param file_name_prefix:
        :param min_rows_of_behaviour: (int) The number of frames that precede and succeed the points of interest
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
        df = df.loc[df["data_source"] == data_source].astype({'frame': float}).astype({'frame': int}).sort_values('frame').copy()

        # Get Run-Length Encoding of assignments
        assignments = df[self.clf_assignment_col_name].values
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
            for i in range(num_examples):  # TODO: HIGH: this part dumbly loops over first n examples...In the future, it would be better to ensure that at least one of the examples has a long runtime for analysis
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
                assignments_list = list(df_frames_selection[self.clf_assignment_col_name].values)
                current_behaviour_list: List[str] = [self.get_assignment_label(a) for a in assignments_list]
                frames_indices_list = list(df_frames_selection['frame'].astype(int).values)
                color_map_array = visuals.generate_color_map(len(self.unique_assignments))
                text_colors_list: List[Tuple[float]] = [tuple(float(min(255. * x, 255.))  # Multiply the 3 values by 255 since existing values are on a 0 to 1 scale
                                                              for x in tuple(color_map_array[a][:3]))  # Takes only the first 3 elements since the 4th appears to be brightness value (?)
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
    def get_plot_figure_of_classifier_assignments_distribution(self) -> Tuple[object, object]:
        """
        Get a histogram of assignments in order to review their distribution in the TRAINING data
        """
        fig, ax = visuals.plot_assignment_distribution_histogram(
            self.df_features_train_scaled[self.clf_assignment_col_name])
        return fig, ax

    def plot_clusters_by_assignments(self, title='', show_now=False, save_to_file=False, azim_elev: tuple = (70, 135), draw_now=False, **kwargs) -> Tuple[object, object]:
        """
        # TODO: rename function as plot assignments by cluster
        Get plot of clusters colored by GMM assignment
        NOTE: all kwargs expected by the plotting function are passed from this function down to that plotting function.
        :param title: (str) Title of graph
        :param show_now: (bool)
        :param save_to_file: (bool)
        :param azim_elev:
        :param kwargs: See kwargs of plotting function for more usage.
        Some notable kwargs include:
            azim_elev : Tuple[int, int]
        :return:
        """
        # Hard coded args to be fixed later  # TODO: HIGH
        if not title:
            title = f'Perplexity={self.tsne_perplexity} / EarlyExaggeration={self.tsne_early_exaggeration} / LearnRate={self.tsne_learning_rate}'
        # Arg checking
        check_arg.ensure_type(azim_elev, tuple)
        # TODO: low: check for other runtime vars
        if not self.is_built:  # or not self._has_unused_raw_data:
            e = f'{get_current_function()}(): The model has not been built. There is nothing to graph.'
            logger.warning(e)
            raise ValueError(e)
        # Resolve kwargs
        # fig_file_prefix = kwargs.get('fig_file_prefix', f'{self.name}__train_assignments_and_clustering__')
        # Execute
        fig, ax = visuals.plot_clusters_by_assignment(
            self.df_features_train_scaled[self.dims_cols_names].values,
            self.df_features_train_scaled[self.gmm_assignment_col_name].values,
            # fig_file_prefix=fig_file_prefix,
            save_fig_to_file=save_to_file,
            show_now=show_now,
            draw_now=draw_now,

            azim_elev=azim_elev,
            title=title,
            **kwargs
        )
        return fig, ax

    def get_plot_cross_val_scoring(self) -> Tuple[object, object]:
        # TODO: med: confirm that this works as expected
        return visuals.plot_cross_validation_scores(self._cross_val_scores)

    def generate_confusion_matrix(self) -> np.ndarray:
        # TODO: high: implement this function in stats then put it into here
        df_features_train_scaled_test_data = self.df_features_train_scaled.loc[self.df_features_train_scaled[self.test_col_name]]
        y_pred = self.clf_predict(df_features_train_scaled_test_data[list(self.all_features)])
        y_true = df_features_train_scaled_test_data[self.clf_assignment_col_name].values
        cnf_matrix = statistics.confusion_matrix(y_true, y_pred)
        return cnf_matrix

    def diagnostics(self) -> str:
        """
        Function for displaying current state of pipeline. Useful for diagnosing problems.
        Changes often and should not be used in final build for anything important.
        """
        diag = f"""
self.is_built: {self.is_built}
unique sources in df_train GMM ASSIGNMENTS: {len(np.unique(self.df_features_train[self.gmm_assignment_col_name].values))}
unique sources in df_train SVM ASSIGNMENTS: {len(np.unique(self.df_features_train[self.clf_assignment_col_name].values))}
self._is_training_data_set_different_from_model_input: {self._is_training_data_set_different_from_model_input}
""".strip()
        return diag

    # built-ins
    def __bool__(self):
        return True

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

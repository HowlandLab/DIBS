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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle as sklearn_shuffle_dataframe
from typing import Any, Collection, Dict, List, Tuple, Union  # TODO: med: review all uses of Optional
from collections import defaultdict
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from types import FunctionType

from dibs.pipeline_pieces import FeatureEngineerer, Embedder, Clusterer, CLF
from dibs import pipeline_pieces

from dibs.logging_enhanced import get_current_function
from dibs import check_arg, config, feature_engineering, io, logging_enhanced, statistics, videoprocessing, visuals

logger = config.initialize_logger(__name__)


# Base pipeline objects that outline the Pipeline API
class BasePipelineAttributeHolder(object):
    """ Philosophy: Read config once at start up ever.
    Hence, all objects should always be viable and editing the config at runtime will have no effect.
    config.ini is meant to act as default parameters, and these will be valid for the entire lifetime
    of the process. If interactive modification of parameters at runtime is desired, either edit the
    source code directly, or use streamlit.
    """

    # Base information
    _name, _description = 'DefaultPipelineName', '(Default pipeline description)'

    # TODO: Scalar??  Not sure what that is used for yet.
    _scaler = None

    # Column names7 # TODO: Rename, and/or load defaults based on config.
    clusterer_assignment_col_name, clf_assignment_col_name, = 'clusterer_assignment', 'classifier_assignment'
    test_col_name = 'is_test_data'

    def __init__(self):
        """ IMPORTANT: Nested objects seem to have to be strict instance variables, or at least not be
        class variables with default initialization expressions...
        Pickling failed when these were outside of __init__.  Could not find any explaination or
        documentation. Could not reproduce with a smaller test case. """
        # Model objects.  Modular, so that we can swap out different parts of the algorithm independently.
        self._feature_engineerer: FeatureEngineerer = getattr(pipeline_pieces, config.FEATURE_ENGINEERER.DEFAULT)()
        self._embedder: Embedder = getattr(pipeline_pieces, config.EMBEDDER.DEFAULT)()
        self._clusterer: Clusterer = getattr(pipeline_pieces, config.CLUSTERER.DEFAULT)()
        self._clf: CLF = getattr(pipeline_pieces, config.CLASSIFIER.DEFAULT)()
        # Data
        default_cols = ['frame', 'data_source', 'file_source', self.clusterer_assignment_col_name, self.clf_assignment_col_name]
        self._df_features_train_raw = pd.DataFrame(columns=default_cols)
        self._df_features_train = pd.DataFrame(columns=default_cols)
        self._df_features_train_scaled = pd.DataFrame(columns=default_cols)
        self._df_features_predict_raw = pd.DataFrame(columns=default_cols)
        self._df_features_predict = pd.DataFrame(columns=default_cols)
        self._df_features_predict_scaled = pd.DataFrame(columns=default_cols)

    _embedder_is_built = False
    _clusterer_is_built = False
    _clf_is_built = False

    cross_validation_k = 10
    cross_validation_n_jobs = 2

    # Pipeline state-tracking variables
    _is_training_data_set_different_from_model_input: bool = False  # Changes to True if new training data is added and classifiers not rebuilt.
    _has_unengineered_predict_data: bool = False  # Changes to True if new predict data is added. Changes to False if features are engineered.

    null_classifier_label = null_clusterer_label = -1

    # Other model vars (Rename this)
    video_fps: float = config.VIDEO_FPS
    average_over_n_frames: int = config.AVERAGE_OVER_N_FRAMES
    test_train_split_pct: float = config.HOLDOUT_PERCENT

    # All label properties for respective assignments instantiated below to ensure no missing properties b/w Pipelines (aka: quick fix, not enough time to debug in full)
    label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9 = ['' for _ in range(10)]
    label_10, label_11, label_12, label_13, label_14, label_15, label_16, label_17, label_18 = ['' for _ in range(9)]
    label_19, label_20, label_21, label_22, label_23, label_24, label_25, label_26, label_27 = ['' for _ in range(9)]
    label_28, label_29, label_30, label_31, label_32, label_33, label_34, label_35, label_36 = ['' for _ in range(9)]
    label_999 = 'DATA_MISSING'

    # SORT ME
    _acc_score: float = -1.
    _cross_val_scores: np.ndarray = np.array([])

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
            err = f'TODO: elaborate error: invalid assignment submitted: "{assignment}".  Must be interger'
            logger.error(err)
            raise ValueError()

        label = getattr(self, f'label_{assignment}', '')

        return label

    def set_label(self, assignment: int, label: str):
        """ Set behavioural label for a given model assignment number """
        check_arg.ensure_type(label, str)
        assignment = int(assignment)
        setattr(self, f'label_{assignment}', label)
        return self

    def convert_types(self, df):
        """ An attempt at standardizing data when called for use """
        return df.astype({'frame': float, }).astype({'frame': int, })

    @property
    def scaler(self): return self._scaler

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
    def df_clusterer_assignments(self):
        return self._df_features_train_scaled[self.clusterer_assignment_col_name].loc[
            self._df_features_train_scaled[self.clusterer_assignment_col_name] != self.null_clusterer_label
        ]
    @property
    def df_embedder_embedding(self):
        """ Allows user to extract reduced data from pipeline. """
        return self._df_features_train_scaled.loc[self.dims_cols_names].copy()

    @property
    def transition_matrix(self):
        gmm_a = self.df_clusterer_assignments
        logger.debug(gmm_a)
        n = len(gmm_a.unique())
        mat = np.zeros((n, n), dtype=int)
        for b1,b2 in zip(gmm_a[0:-2], gmm_a[1:-1]):
            mat[b1,b2] += 1 # TODO: med; There is a bug here at the intersection of datasets, since we are measuring
            #                       the transition from one behavoiur to the next when really the video changed.
            #                       Since this accounts for much less than 0.1% of our overall data, it is not critical to fix at this time.
        np.fill_diagonal(mat, 0)
        return mat

    def plot_transition_matrix_heatmap(self, cmap=sns.color_palette("Blues", as_cmap=True),
                                       save_name='',
                                       save_format=config.DEFAULT_SAVED_GRAPH_FILE_FORMAT, **kwargs):
        # TODO: Add these options too stuff and things?
        # TODO: Add axis labels.  Change labels to reflect user labels
        # sns.set(rc={'figure.figsize':(11.7, 8.27)})
        plt.clf()
        hm=sns.heatmap(
            self.transition_matrix,
            cmap=cmap,
            annot=True,
            fmt='d',
            **kwargs
        )
        fig = hm.get_figure()
        if save_name:
            fig.savefig(os.path.join(config.GRAPH_OUTPUT_PATH, f'{save_name}.{save_format}'))
        return fig

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def _is_built(self):
        return self._clf_is_built or self._clusterer_is_built or self._embedder_is_built

    @property
    def is_in_inconsistent_state(self):
        """
        Useful for checking if training data has been added/removed from pipeline
        relative to already-compiled model
        """
        return self._is_training_data_set_different_from_model_input \
            or self._has_unengineered_predict_data \
            or not self._is_built

    def get_inconsistent_state_repr(self) -> str:
        """ https://github.com/streamlit/streamlit/issues/868 require extra white space to make multi line """
        return f'_is_training_data_set_different_from_model_input: {self._is_training_data_set_different_from_model_input}      \n' \
               f'_has_unengineered_predict_data: {self._has_unengineered_predict_data}     \n' \
               f'_embedder_is_built: {self._embedder_is_built}    \n' \
               f'_clusterer_is_built: {self._clusterer_is_built}    \n' \
               f'_clf_is_built: {self._clf_is_built}    \n'

    @property
    def is_built(self): return self._is_built

    @property
    def has_train_data(self) -> bool: return bool(len(self.df_features_train_raw))

    @property
    def has_test_data(self) -> bool: return bool(len(self.df_features_predict_raw))

    @property
    def accuracy_score(self): return self._acc_score

    @property
    def cross_val_scores(self): return self._cross_val_scores

    @property
    def training_data_sources(self) -> List[str]:
        return list(np.unique(self.df_features_train_raw['data_source'].values))

    @staticmethod
    def _video_path_finder(data_sources: list) -> Dict[str, str]:
        # Use name matching on video path to find videos
        video_paths = sorted(os.listdir(config.VIDEO_INPUT_FOLDER_PATH))
        sources = sorted(data_sources)
        sources_to_video_paths = dict()
        while len(video_paths) > 0:
            path = video_paths.pop()
            logger.debug(f'Just popped path: {path}')
            temp_sources = sources.copy()
            while len(temp_sources) > 0:
                source = temp_sources.pop()
                # HACK: The 'DLC.*' suffix is appended to all data sources, and we assume that this will be present,
                #       and that the string prior to this suffix will be part of the path
                if source.split('DLC')[0] in os.path.splitext(os.path.basename(path))[0]:
                    if source in sources_to_video_paths:
                        # TODO: Come up with more robust logic, or actionable solution for the user to take.  Possible solution: Have the split pattern specified in config.ini
                        raise RuntimeError(f'Double matched source {source} to multiple paths: {path}, {sources_to_video_paths[source]}')
                    sources_to_video_paths[source] = os.path.join(config.VIDEO_INPUT_FOLDER_PATH, path)
                    # HACK: Have to remove from the sources list... so there is no double matching
                    sources.remove(source)
                    break
            else:
                logger.warn(f'Video path not matched to any data source: {path}; This is probably undesirable.')
        logger.info(f'Matched videos and paths as follows:')
        for k,v in sources_to_video_paths.items():
            logger.info(f'{k}: {v}')
        if unmatched_datasources := sorted(set(data_sources) - set(sources_to_video_paths.keys())):
            logger.warn(f'Sources not matched to a video path:')
            for source in unmatched_datasources:
                logger.warn(f'{source}')
        return sources_to_video_paths

    @property
    def training_video_sources(self) -> Dict[str, str]:
        return self._video_path_finder(self.training_data_sources)

    @property
    def predict_video_sources(self) -> Dict[str, str]:
        return self._video_path_finder(self.predict_data_sources)

    @property
    def predict_data_sources(self):
        return list(np.unique(self.df_features_predict_raw['data_source'].values))

    @property
    def unique_assignments(self) -> List[any]:
        if len(self._df_features_train_scaled) > 0:
            return list(np.unique(self._df_features_train_scaled.loc[self._df_features_train_scaled[self.clf_assignment_col_name] != self.null_classifier_label][self.clf_assignment_col_name].values))
        return []

    @property
    def all_engineered_features(self) -> Tuple[str]: return self._feature_engineerer._all_engineered_features

    @property
    def all_engineered_features_list(self) -> List[str]: return list(self.all_engineered_features)

    @property
    def dims_cols_names(self) -> List[str]:
        """
        Automatically creates a list of consistent column names, relative to the number of
        TSNE components, that labels the columns of reduced data after the TSNE operation.
        """
        return [f'dim_{d}' for d in range(1, self._embedder.n_components+1)]


class BasePipeline(BasePipelineAttributeHolder):
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

    # Init
    def __init__(self, name: str, **kwargs):
        # Pipeline name
        super().__init__()
        logger.info(f'Received following params when initializing pipeline: {kwargs}')
        check_arg.ensure_type(name, str)
        self.set_name(name)
        # Set parameters for Pipeline according to kwargs. If kwarg is missing, use default from config.ini.
        self.set_params(**kwargs)

    # Setters
    def set_name(self, name: str):
        check_arg.ensure_has_valid_chars_for_path(name)
        self._name = name
        return self

    def set_description(self, description: str):
        """ Set a description of the pipeline. Include any notes you want to keep regarding the process used. """
        check_arg.ensure_type(description, str)
        self._description = description
        return self

    def set_params(self, **kwargs):
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
        """

        ### General Params ###
        # Test/train split %
        test_train_split_pct = kwargs.get('test_train_split_pct', self.test_train_split_pct)
        check_arg.ensure_type(test_train_split_pct, float)
        self.test_train_split_pct = test_train_split_pct

        def _dict_is_subset_of(d1, d2):
            """ d1 should be a subset of d2 """
            return len(set(d1.items()) - set(d2.items())) == 0


        ### MODEL PARAMS ###
        # TODO: Use a stringified "hash" representation of the feature engineering in some way to ensure we don't redundantly redo?
        if new_feature_engineerer_tuple := kwargs.get('FEATURE_ENGINEERER'):
            new_feature_engineerer_name, new_feature_engineerer_params = new_feature_engineerer_tuple # params usually just RANDOM_STATE
            if new_feature_engineerer_name != self._feature_engineerer.__class__.__name__:
                self._feature_engineerer = getattr(pipeline_pieces, new_feature_engineerer_name)()
            # self._feature_engineerer.set_params(new_feature_engineerer_params) # TODO: Let feature engineerer take args, for example tau
            self._is_training_data_set_different_from_model_input = True # Forces re-generation of engineered training datasets
            self._has_unengineered_predict_data = True # Forces re-generation of engineered predict datasets

        if new_embedder_tuple := kwargs.get('EMBEDDER'):
            new_embedder_name, new_embedder_params = new_embedder_tuple
            if new_embedder_name != self._embedder.__class__.__name__:
                logger.debug('Embedder name changed')
                self._embedder = getattr(pipeline_pieces, new_embedder_name)(new_embedder_params)
                self._embedder_is_built = False
            else:
                if not _dict_is_subset_of(new_embedder_params, self._embedder.get_params()):
                    logger.debug('Embedder params changed')
                    self._embedder.set_params(new_embedder_params)
                    self._embedder_is_built = False

        if new_clusterer_tuple := kwargs.get('CLUSTERER'):
            new_clusterer_name, new_clusterer_params = new_clusterer_tuple
            if new_clusterer_name != self._clusterer.__class__.__name__:
                logger.debug('Clusterer name changed')
                self._clusterer = getattr(pipeline_pieces, new_clusterer_name)(new_clusterer_params)
                self._clusterer_is_built = False
            else:
                if not _dict_is_subset_of(new_clusterer_params, self._clusterer.get_params()):
                    logger.debug('Clusterer params changed')
                    self._clusterer.set_params(new_clusterer_params)
                    self._clusterer_is_built = False

        if new_clf_tuple := kwargs.get('CLF'):
            new_clf_name, new_clf_params = new_clf_tuple
            if new_clf_name != self._clf.__class__.__name__:
                logger.debug('CLF name changed')
                self._clf = getattr(pipeline_pieces, new_clf_name)(new_clf_params)
                self._clf_is_built = False
            else:
                if not _dict_is_subset_of(new_clf_params, self._clf.get_params()):
                    logger.debug('CLF params changed')
                    self._clf.set_params(new_clf_params)
                    self._clf_is_built = False

        # TODO: MED: ADD KWARGS OPTION FOR OVERRIDING VERBOSE in CONFIG.INI!!!!!!!! ?
        # Source video FPS # TODO: NOT SURE IF VIDEO FPS SHOULD INVALIDATE MODELS OR NOT?? Probably not but I don't know for sure
        video_fps = kwargs.get('video_fps', self.video_fps)
        check_arg.ensure_type(video_fps, int, float)
        self.video_fps = float(video_fps)
        # Window averaging
        average_over_n_frames = kwargs.get('average_over_n_frames', self.average_over_n_frames)  # TODO: low: add a default option for this in config.ini+config.py
        check_arg.ensure_type(average_over_n_frames, int)
        self.average_over_n_frames = average_over_n_frames

        return self

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
        :param data_source: (str) name of a data source (not necessary to include file type extension).
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
        :param data_source: (str) name of a data source (not necessary to include file type extension).
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

    ## Engineer features
    def engineer_features(self, data: pd.DataFrame):
        """
        This function takes one data that is continuous in time and engineers all necessary features.
        It *must* be overridden by all child classes for the respective child class to be considered valid.
        """
        return self._feature_engineerer.engineer_features(data, self.average_over_n_frames)

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

        # Reconcile args
        check_arg.ensure_type(list_dfs_of_raw_data, list)

        # Execute
        # TODO: MED: implement multiprocessing
        list_dfs_engineered_features: List[pd.DataFrame] = []
        for i, df_i in enumerate(list_dfs_of_raw_data):
            df_i = self.convert_types(df_i.copy())
            check_arg.ensure_frame_indices_are_integers(df_i)
            logger.debug(f'{get_current_function()}(): Engineering df feature set {i+1} of {len(list_dfs_of_raw_data)}')
            df_engineered_features: pd.DataFrame = self.engineer_features(df_i)
            df_engineered_features = df_engineered_features.astype({feature: float for feature in self.all_engineered_features})
            list_dfs_engineered_features.append(df_engineered_features)

        # Aggregate all data into one DataFrame, return
        df_features = pd.concat(list_dfs_engineered_features)

        return df_features

    def _engineer_features_train(self):
        """
        TODO: Why not engineer all features at once?  Speed?
        Utilizes
        All functions that take the raw data (data retrieved from using dibs.read_csv()) and
        transforms it into classifier-ready data.

        Splits data based on data_source, then sends the list of DataFrames by source to engineer_features().
        Post-conditions: sets TODO: med
        Returns self.
        """
        # Queue up data according to data source
        list_dfs_raw_data = [self.df_features_train_raw.loc[self.df_features_train_raw['data_source'] == src]
                                 .astype({'frame': float}).sort_values('frame').copy()
                             for src in set(self.df_features_train_raw['data_source'].values)]
        # Call engineering function
        logger.debug(f'Start engineering training data features.')
        df_features = self.engineer_features_all_dfs(list_dfs_raw_data)
        logger.debug(f'Done engineering training data features.')
        # Save data
        self._df_features_train = df_features
        # Wrap up
        self._is_training_data_set_different_from_model_input = False
        return self

    def _engineer_features_predict(self):
        """
        Engineer features for the predicted data
        Splits data based on data_source, then sends the list of DataFrames by source to engineer_features().
        """
        logger.debug(f'Engineering features on prediction data.')
        # Queue data
        list_dfs_raw_data = [self.df_features_predict_raw.loc[self.df_features_predict_raw['data_source'] == src]
                                 .sort_values('frame').copy()
                             for src in set(self.df_features_predict_raw['data_source'].values)]
        # Call engineering function
        logger.debug(f'Start engineering predict data features.')
        df_features = self.engineer_features_all_dfs(list_dfs_raw_data)
        logger.debug(f'Done engineering predict data features.')
        # Save data, return
        self._df_features_predict = df_features
        self._has_unengineered_predict_data = False
        return self

    ## Scaling data
    def _create_scaled_data(self, df_data, features, create_new_scaler) -> pd.DataFrame:
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
            # TODO: AARONT: Analyze or produce metrics to analyze the impacts of scaling
            self._scaler = StandardScaler()
            # self._scaler = MinMaxScaler()
            self._scaler.fit(df_data[features])
        assert self._scaler is not None, 'Strange... you have not intialized a scalar yet'
        arr_data_scaled: np.ndarray = self.scaler.transform(df_data[features])
        df_scaled_data = pd.DataFrame(arr_data_scaled, columns=features)

        # For new DataFrame, replace columns that were not scaled so that data does not go missing
        for col in df_data.columns:
            if col not in set(df_scaled_data.columns):
                df_scaled_data[col] = df_data[col].values

        return df_scaled_data

    def _scale_training_data_and_add_train_test_split(self, features: Collection[str] = None, create_new_scaler: bool = True):
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
        if features is None:
            features = self.all_engineered_features
        features = list(features)
        df_features_train = self.df_features_train.copy()

        logger.debug(f'Scaling data now...')
        # Add train/test assignment col
        df_features_train = feature_engineering.attach_train_test_split_col(
            df_features_train,
            test_col=self.test_col_name,
            test_pct=self.test_train_split_pct,
            sort_results_by=['data_source', 'frame'],
            random_state=self._feature_engineerer.random_state,
        )

        # df_features_train_scaled = df_features_train_scaled.loc[~df_features_train_scaled[features].isnull().any(axis=1)]

        # Get scaled data
        df_features_train_scaled = self._create_scaled_data(df_features_train, features, create_new_scaler=create_new_scaler)

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
        logger.debug('Applying scale transform to predict data')
        if features is None:
            features = self.all_engineered_features
        features = list(features)
        df_features_predict = self.df_features_predict

        # Check args before execution
        check_arg.ensure_type(df_features_predict, pd.DataFrame)
        check_arg.ensure_columns_in_DataFrame(df_features_predict, features)

        # Get scaled data
        df_scaled_data: pd.DataFrame = self._create_scaled_data(df_features_predict, features, create_new_scaler=False)

        # Save data. Return.
        self._df_features_predict_scaled = df_scaled_data
        return self

    def recolor_with_clusterer(self):
        """ Replacement for below. """
        raise NotImplementedError()
    # def recolor_gmm_and_retrain_classifier(self, n_components: int):
    #     self._train_gmm_and_classifier(n_components)
    #     return self

    # Pipeline building
    def build(self, force_reengineer_train_features=False, reengineer_predict_features=False,
              skip_accuracy_score: bool = False, pipeline_file_path=None):
        """
        Encapsulate entire build process from front to back.
        This included transforming training data, predict data, training classifiers, and getting all results.
        Dev note: the skipping of accuracy scoring is mainly meant for debug purposes.
        """
        if not self.has_train_data:
            err = 'No training data has been added to the pipeline yet.  Please add training data before building the model.'
            logger.error(err)
            raise RuntimeError(err)
        start = time.perf_counter()
        # Build model

        # # HACK: TODO: Remove after implementing dataset managment
        # force_reengineer_train_features=True
        # reengineer_predict_features=True
        # self._embedder_is_built = False
        # self._clusterer_is_built = False
        # self._clf_is_built = False

        # Build it
        # Engineer features
        if force_reengineer_train_features or self._is_training_data_set_different_from_model_input:
            logger.debug(f'{get_current_function()}(): Start engineering features...')
            self._engineer_features_train()
            if pipeline_file_path: io.save_to_folder(self, pipeline_file_path)

        # TODO: Time each step and log for the user.
        # Scale data
        self._scale_training_data_and_add_train_test_split(create_new_scaler=True)

        # Rebuild any parts that require updating.
        # These flags are set to False on initialization, when setting parameters,
        # and if any previous piece of the pipeline has been changed.
        if not self._embedder_is_built:
            self._build_embedder()
            if pipeline_file_path: io.save_to_folder(self, pipeline_file_path)

        if not self._clusterer_is_built:
            self._build_clusterer()
            if pipeline_file_path: io.save_to_folder(self, pipeline_file_path)

        if not self._clf_is_built:
            self._build_classifier()
            if pipeline_file_path: io.save_to_folder(self, pipeline_file_path)

        self._last_built = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
        logger.debug(f'{get_current_function()}(): All done with building classifiers/model!')

        # Circle back and apply labels to all data, train & test alike
        # TODO: Record if the labels are up to date
        self._label_data_with_classifier()

        # Accuracy scoring
        if skip_accuracy_score:
            logger.debug(f'Accuracy/cross-validation scoring is being skipped.')
        else:
            self.generate_accuracy_scores()

        # Get predict data
        self._generate_predict_data_assignments(reengineer_predict_features=reengineer_predict_features)
        # Wrap up
        end = time.perf_counter()
        self.seconds_to_build = round(end - start, 2)
        io.save_to_folder(self, pipeline_file_path)
        # logger.info(f'{get_current_function()}(): Total build time: {self.seconds_to_build} seconds. Rows of data: {len(self._df_features_train_scaled)} / tsne_n_jobs={self._embedder.n_jobs} / cross_validation_n_jobs = {self.cross_validation_n_jobs}')  # TODO: med: amend this line later. Has extra info for debugging purposes.
        return self

    # dimensionality reduction (embedding) TODO: low; Rename
    def _build_embedder(self):
        # TODO: HIGH: make sure that grabbing the data for training is standardized <----------------------------
        # Grab train data
        df_train_data_for_tsne = self.df_features_train_scaled.loc[
                (~self.df_features_train_scaled[self.all_engineered_features_list].isnull().any(axis=1))  # Non-null features only
            ].copy()

        check_arg.ensure_columns_in_DataFrame(df_train_data_for_tsne, self.all_engineered_features_list)
        logger.debug(f'Reducing feature dimensions now with {self._embedder.__class__.__name__}')
        logger.debug(f'Params: {self._embedder.get_params()}')

        arr_tsne_result: np.ndarray = self._embedder.embed(df_train_data_for_tsne[list(self.all_engineered_features)])
        check_arg.ensure_type(arr_tsne_result, np.ndarray)  # TODO: low: remove, debuggin effort for dim reduc approaches

        # Attach dimensionally reduced data, save
        # TODO: Here, we have to handle the data differently
        # self._df_features_train_scaled_train_split_only = pd.concat([
        #     df_train_data_for_tsne,
        #     pd.DataFrame(arr_tsne_result, columns=self.dims_cols_names),
        # ], axis=1)
        # TODO: This doesn't seem to be saving
        self._df_features_train_scaled[self.dims_cols_names] = arr_tsne_result

        self._embedder_is_built = True

        self._clusterer_is_built = False
        self._clf_is_built = False

        return self

    def _build_clusterer(self):
        """"""
        df = self._df_features_train_scaled
        # TODO: What if we drop some? How do we line up the rows later?
        df = df.loc[~df[self.dims_cols_names].isnull().any(axis=1)][self.dims_cols_names]
        assert np.all(np.isfinite(df.values)), 'Where did the embedder cols go?'

        logger.debug(f'Training {self._clusterer.__class__.__name__} now...')
        logger.debug(f'Params: {self._clusterer.get_params()}')
        labels = self._clusterer.train(df)
        assert np.all(np.isfinite(labels))
        self._df_features_train_scaled[self.clusterer_assignment_col_name] = labels

        self._clusterer_is_built = True
        self._clf_is_built = False

        return self

    def _build_classifier(self):
        """ Train classifier, TODO: And apply labels to something? """
        # TODO: Here we must handle the thing... This is not working
        df_train = self._df_features_train_scaled[self._df_features_train_scaled[self.test_col_name] == 0]
        a = df_train.copy()
        # TODO: TEST!
        # assert len(df) == count of times true in thing
        # Select only
        df_train = df_train.loc[
            (~df_train[list(self.all_engineered_features)].isnull().any(axis=1)) &
            (~df_train[self.clusterer_assignment_col_name].isnull()) &
            (df_train[self.clusterer_assignment_col_name] != self.null_clusterer_label)
            ]

        # classifier is trained in high dimensional feature space, where prediction of new data will occur
        X=df_train[list(self.all_engineered_features)]
        y=df_train[self.clusterer_assignment_col_name]

        logger.debug(f'Training {self._clf.__class__.__name__} classifier now...')
        logger.debug(f'Params: {self._clf.get_params()}')
        self._clf.train(X, y)
        logger.debug(f'Finished training {self._clf.__class__.__name__} classifier!')

        self._clf_is_built = True

        return self

    # AARONT: TODO: Where do these fit in exactly?
    def generate_accuracy_scores(self):
        """

        :return:
        """
        df = self.df_features_train_scaled
        df = df.loc[
            (~df[list(self.all_engineered_features) + [self.clf_assignment_col_name]].isnull().any(axis=1))
            # & (df[] != self.label_999)
        ]
        logger.debug(f'Generating cross-validation scores...')
        # # Get cross-val accuracy scores
        try:
            self._cross_val_scores = cross_val_score(
                self._clf._model,
                df[self.all_engineered_features_list].values,
                df[self.clusterer_assignment_col_name].values,
                cv=self.cross_validation_k,
                n_jobs=self.cross_validation_n_jobs,
                pre_dispatch=self.cross_validation_n_jobs,
            )
        except ValueError as ve:
            cross_val_failure_warning = f'Cross-validation could not be computed in {self.name}. See the following error: {repr(ve)}'
            logger.warning(cross_val_failure_warning)

        df_features_train_scaled_test_data = df.loc[df[self.test_col_name] == 1]

        # TODO: AARONT: Looks like we are using the same input data here as we did when training...?
        self._acc_score = accuracy_score(
            y_pred=self.clf_predict(df_features_train_scaled_test_data[list(self.all_engineered_features)].values),
            y_true=df_features_train_scaled_test_data[self.clf_assignment_col_name].values)
        logger.debug('Finished generating cross-validation scores')
        logger.debug(f'Pipeline train accuracy: {self.accuracy_score}')
        return self

    def clf_predict(self, data: np.array):
        check_arg.ensure_type(data, np.ndarray)
        d = np.copy(data) # TODO: Do we need to make a copy? Why?
        d[~np.isfinite(d)] = 0 # TODO: HACK: Could handle NaNs better...
        return np.array([int(x) if x == x else int(self.null_classifier_label) for x in self._clf.predict(d)])

    def _label_data_with_classifier(self):
        """
        :return:
        """
        logger.debug('Labelling data with classifier now')
        # Set classifier predictions
        # TODO: Remove after debugging
        a = self._df_features_train_scaled[self.all_engineered_features_list].values
        # a[~np.isfinite(a)] = 0 # HACKS: Should probably do something smarter with nans. Also: Why doesn't np.isnan work?
        clf_labels = self.clf_predict(a)
        self._df_features_train_scaled[self.clf_assignment_col_name] = clf_labels
        assert self.clf_assignment_col_name in self._df_features_train_scaled.columns
        assert self.clf_assignment_col_name in self.df_features_train_scaled.columns
        # TODO: True passing all the things
        # self._df_features_train_scaled[self.clf_assignment_col_name] = self._df_features_train_scaled[self.all_engineered_features_list].\
        #     apply(lambda series: self.clf_predict(series.values.reshape(1, len(self.all_engineered_features))), axis=1)  # self.clf_predict(self.df_features_train_scaled[list(self.all_features)].values)  # Get predictions
        # logger.debug('Finished generating labels with classifier')
        # self._df_features_train_scaled[self.clf_assignment_col_name] = self._df_features_train_scaled[self.clf_assignment_col_name].map(lambda x: self.null_classifier_label if x != x else x)
        # self._df_features_train_scaled[self.clf_assignment_col_name] = self.df_features_train_scaled[self.clf_assignment_col_name].astype(int)  # Coerce into int

        return self

    # Predictions
    def _generate_predict_data_assignments(self, reengineer_predict_features=False):  # TODO: low: rename?
        """
        Runs after build(). Using terminology from old implementation. TODO: purpose
        """
        logger.debug(f'Generating prediction data assignments')

        # Check that classifiers are built on the training data
        if self._is_training_data_set_different_from_model_input or not self._is_built:
            # AARONT: TODO: Improve warning message and logging and match Killians style or remove this check all together.
            raise RuntimeError('Calling this funtion is not a good idea in the way that you are doing it.')

        # TODO: temp exit early for zero test data found
        if len(self.df_features_predict_raw) == 0:
            warn = f'Zero test data points found. Exiting early. predict features not built.'
            logger.warning(warn)
            return self

        # Check if predict features have been engineered
        if reengineer_predict_features or self._has_unengineered_predict_data:
            self._engineer_features_predict()
            self._scale_transform_predict_data()

        # Add prediction labels
        if len(self.df_features_predict_scaled) > 0:
            self.df_features_predict_scaled[self.clf_assignment_col_name] = self.clf_predict(self.df_features_predict_scaled[list(self.all_engineered_features)].values)
        else:
            logger.debug(f'{get_current_function()}(): 0 records were detected '
                         f'for PREDICT data. No data was predicted with model.')

        logger.debug('Finished generating prediction data assignments')
        return self

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

        svm_assignment_values_array = df_data[self.clf_assignment_col_name].values
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

    def make_behaviour_example_videos(self, data_source_type: str, ex_video_dir_name=None,
                                      min_rows_of_behaviour=1, max_examples=1, num_frames_buffer=0, output_fps=15, # AARONT: TODO: High fidelity model augment here
                                      max_frames_per_video=500):
        """
        Create video clips of behaviours

        :param ex_video_dir_name:
        :param min_rows_of_behaviour: (int) The number of frames that precede and succeed the points of interest
        :param max_examples:
        :return:
        """
        # Args checking
        check_arg.ensure_type(num_frames_buffer, int)
        check_arg.ensure_type(output_fps, int, float)
        # Solve kwargs
        if ex_video_dir_name is None:
            raise RuntimeError('Must provide a prefix for now; Dont want to drop videos in default location.')
        else:
            check_arg.ensure_type(ex_video_dir_name, str)
            check_arg.ensure_has_valid_chars_for_path(ex_video_dir_name)
            ex_video_output_dir = os.path.join(config.EXAMPLE_VIDEOS_OUTPUT_PATH, ex_video_dir_name)
            logger.debug(f'Creating example videos output dir: {ex_video_output_dir}')
            os.makedirs(ex_video_output_dir, exist_ok=True)

        # Get data from data source name
        if data_source_type == 'training':
            full_df = self.df_features_train_scaled
            data_source_to_video_path: Dict[str, str] = self.training_video_sources
        elif data_source_type == 'predict':
            full_df = self.df_features_predict_scaled
            data_source_to_video_path: Dict[str, str] = self.predict_video_sources
        else:
            err = f'Data source type invalid: {data_source_type}'
            logger.error(err)
            raise ValueError(err)

        valid_data_sources = data_source_to_video_path.keys()
        logger.debug(f'WIP: Now we make the vids...  Video sources: {data_source_to_video_path}')
        for path in data_source_to_video_path.values():
            check_arg.ensure_is_file(path)

        logger.debug(f'{get_current_function()}(): Total records: {len(full_df)}')

        # cache all the required dataframes for quick lookup
        data_source_to_data_frame = {
            data_source: full_df.loc[full_df['data_source'] == data_source].
            astype({'frame': float}).astype({'frame': int}).sort_values('frame') # TODO: Why astype?  It is making a copy at least.
            for data_source in valid_data_sources
        }

        ### Execute
        # Get DataFrame of the data
        all_rle_by_assignment: Dict[Any: Tuple[str, int, int]] = defaultdict(list) # will be labels to list
        for data_source in valid_data_sources:
            # TODO: NOTE: astype creates a copy.  Why are we using astype?
            df = data_source_to_data_frame[data_source]

            # Get Run-Length Encoding of assignments
            assignments = df[self.clf_assignment_col_name].values
            rle: Tuple[List, List, List] = statistics.augmented_runlength_encoding(assignments)
            # TODO: add data source to each tuple.

            # Zip RLE according to order
            # First index is value, second is index, third is *additional* length of value occurrence in sequence.
            # EXAMPLE RESULT: Zip'd RLE according to order: [[15, 0, 0], [4, 1, 1], [14, 3, 0], [15, 4, 0], ... ]
            rle_zipped_by_entry = []
            for row__assignment_idx_addedLength in zip(*rle):
                rle_zipped_by_entry.append(list(row__assignment_idx_addedLength))

            for label, frame_idx, additional_length in rle_zipped_by_entry:
                if additional_length >= min_rows_of_behaviour - 1:
                    all_rle_by_assignment[label].append([data_source, frame_idx, additional_length])

        # TODO: Sort them afterwards.  Then take the ones we want, and convert into the proper format.
        # Sort from longest additional length (ostensibly the duration of behaviour) to least
        for assignment_val in (key for key in all_rle_by_assignment.keys() if key != self.null_classifier_label):
            num_examples = min(max_examples, len(all_rle_by_assignment[assignment_val]))
            all_rle_by_assignment[assignment_val] = sorted(all_rle_by_assignment[assignment_val],
                                                       key=lambda x: x[2], # sort by length
                                                       reverse=True  # True means sort largest to smallest
                                                       )[:num_examples-1]
            # TODO: Now map function over to convert to proper format per label
            #       Desired format is:

        ## TODO: Make list of all labels, assign a color to each.

        unique_assignments = self.df_features_train_scaled[self.clf_assignment_col_name].unique()
        logger.debug(f'*** {unique_assignments} *** UNIQUE ASSIGNMENTS HERE')
        unique_assignments_index_dict = {assignment: i for i, assignment in enumerate(unique_assignments)}
        color_map_array: np.ndarray = visuals.generate_color_map(len(unique_assignments))

        def expand_rle(df, frame_idx, additional_length):
            """
            Return list of tuples representing a clip from a video, all from the same source.
            list[tuple(label, user_assignment_label, color, frame_idx), ...]
            """
            lower_bound_row_idx: int = max(0, int(frame_idx) - num_frames_buffer)
            upper_bound_row_idx: int = min(len(df) - 1, frame_idx + additional_length - 1 + num_frames_buffer)
            df_frames_selection = df.iloc[lower_bound_row_idx:upper_bound_row_idx, :]
            labels = df_frames_selection[self.clf_assignment_col_name].values
            return list(zip(
                [label for label in labels],  # just a copy
                [self.get_assignment_label(label) for label in labels],
                # # Iterate over all assignments, get their respective index, get the colour (3-tuple) for that index, normalize for 255 and set at same index position so that length of colours is same as length of assignments
                # Multiply the 3 values by 255 since existing values are on a 0 to 1 scale
                # Takes only the first 3 elements since the 4th appears to be brightness value (?)
                [tuple(float(min(255. * x, 255.))
                       for x in tuple(color_map_array[unique_assignments_index_dict[label]][:3]))
                       for label in labels],
                df_frames_selection['frame'].values,
            ))

        # Now expand into format that can be used to make videos. 1 per label.
        ## Dict of lable to list of (source, [list of frame specs]) tuples
        label_to_frames = defaultdict(list)
        for label in all_rle_by_assignment:
            for data_source, frame_idx, additional_length in all_rle_by_assignment[label]:
                label_to_frames[label].append((data_source,
                                               expand_rle(
                                                   data_source_to_data_frame[data_source], frame_idx, additional_length))
                                              )
        ### Finally: make video clips
        # Loop over assignments
        time_prefix = time.strftime("%y-%m-%d_%Hh%Mm")
        for assignment_val, data_source_with_video_clip_tuples in ((k, v) for (k, v) in label_to_frames.items() if k != self.null_classifier_label):
            output_file_name = f'{time_prefix}_BehaviourExample__assignment_{assignment_val:02d}'
            frame_text_prefix = f'Target assignment: {assignment_val} / '  # TODO: med/high: magic variable

            videoprocessing.make_video_from_multiple_sources(
                data_source_with_video_clip_tuples, # has clips in order to be read
                data_source_to_video_path, # for opening video files to read
                output_file_name,
                text_prefix=frame_text_prefix,
                output_fps=output_fps,
                output_dir=ex_video_output_dir,
            )

        return self

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
        data = self.df_features_train_scaled
        # TODO: We want to see the null classifier labels!
        # data = data.loc[data[self.clusterer_assignment_col_name] != self.null_clusterer_label]
        if np.any(data[self.clusterer_assignment_col_name] == self.null_classifier_label):
            logger.warn(f'We have null classifier labels, visualization might fail')
        fig, ax = visuals.plot_clusters_by_assignment(
            data[self.dims_cols_names].values,
            data[self.clusterer_assignment_col_name].values,
            # fig_file_prefix=fig_file_prefix,
            save_fig_to_file=save_to_file,
            show_now=show_now,
            draw_now=draw_now,

            azim_elev=azim_elev,
            title=title,
            **kwargs
        )
        return fig, ax

    def plot_confusion_matrix(self) -> np.ndarray:
        fig = sns.heatmap(
            self.generate_confusion_matrix(),
            annot=True,
            fmt='d'
        ).get_figure()
        return fig

    def generate_confusion_matrix(self) -> np.ndarray:
        """
        Generate confusion matrix for test data
        """
        # Select test data, filter out NULLs
        df_data = self.df_features_train_scaled
        df_data = df_data.loc[
            (df_data[self.test_col_name]) &
            (df_data[self.clf_assignment_col_name] != self.null_classifier_label)
        ]

        y_true = df_data[self.clusterer_assignment_col_name].values
        y_pred = self.clf_predict(df_data[self.all_engineered_features_list].values)

        # Generate confusion matrix
        # df_features_train_scaled_test_data = self.df_features_train_scaled_train_split_only.loc[self.df_features_train_scaled_train_split_only[self.test_col_name]];y_pred = self.clf_predict(df_features_train_scaled_test_data[list(self.all_features)]);y_true = df_features_train_scaled_test_data[self.clf_assignment_col_name].values
        cnf_matrix = statistics.confusion_matrix(y_true, y_pred)
        return cnf_matrix

    def diagnostics(self) -> str:
        """
        Function for displaying current state of pipeline. Useful for diagnosing problems.
        Changes often and should not be used in final build for anything important.
        """
        diag = f"""
self.is_built: {self.is_built}
unique sources in df_train GMM ASSIGNMENTS: {len(np.unique(self.df_features_train_scaled[self.clusterer_assignment_col_name].values)) if self.clf_assignment_col_name in set(self.df_features_train_scaled_train_split_only.columns) else "NA"}
unique sources in df_train SVM ASSIGNMENTS: {len(np.unique(self.df_features_train_scaled[self.clf_assignment_col_name].values)) if self.clf_assignment_col_name in set(self.df_features_train_scaled_train_split_only.columns) else "NA"}
self._is_training_data_set_different_from_model_input: {self._is_training_data_set_different_from_model_input}

Embedder params:
{self._embedder.params_as_string()}

Clusterer params:
{self._clusterer.params_as_string()}

Classifier params:
{self._clf.params_as_string()}

all_engineered_features = {self.all_engineered_features}

""".strip()
        return diag

    # built-ins
    def __bool__(self):
        return True

    def __repr__(self) -> str:
        # TODO: low: flesh out how these are usually built. Add a last updated info?
        return f'{self.name}'

 # DistanceForepawLeftToNosetip  DistanceForepawRightToNosetip  DistanceForepawLeftToHindpawLeft  DistanceForepawRightToHindpawRight  DistanceAvgHindpawToNosetip  DistanceAvgForepawToNosetip  VelocityAvgForepaw  index          NoseTip_x           NoseTip_y       ForepawLeft_x      ForepawLeft_y      ForepawRight_x      ForepawRight_y       HindpawLeft_x     HindpawLeft_y      HindpawRight_x     HindpawRight_y        TailBase_x         TailBase_y                                         scorer                                                                                                    file_source                                              data_source  frame        AvgForepaw_x       AvgForepaw_y        AvgHindpaw_x       AvgHindpaw_y is_test_data      dim_1      dim_2    g
# 0                      0.053019                       0.039239                          0.866732                            1.000000                     0.125995                     0.037744            0.000000    0.0                nan                 nan                 nan                nan                 nan                 nan                 nan               nan                 nan                nan               nan                nan  DLC_resnet50_Maternal_EPMDec28shuffle1_700000  C:\Users\killian\projects\DIBS\epm_data_csv_train\EPM-MCE-10DLC_resnet50_Maternal_EPMDec28shuffle1_700000.csv  EPM-MCE-10DLC_resnet50_Maternal_EPMDec28shuffle1_700000    0.0                 nan                nan                 nan                nan        False -27.276716  32.215048  [2]
# 1                      0.051575                       0.037303                          0.910765                            0.974973                     0.123089                     0.036062            0.091718    1.0  970.2467651367188  484.22308349609375    1009.06005859375  523.5932006835938  1016.3133544921875  495.00445556640625   1090.837158203125  542.942626953125    1107.18212890625   507.996337890625      1125.5078125  549.2086791992188  DLC_resnet50_Maternal_EPMDec28shuffle1_700000  C:\Users\killian\projects\DIBS\epm_data_csv_train\EPM-MCE-10DLC_resnet50_Maternal_EPMDec28shuffle1_700000.csv  EPM-MCE-10DLC_resnet50_Maternal_EPMDec28shuffle1_700000    3.0  1012.6867065429688      509.298828125  1099.0096435546875   525.469482421875        False -28.093036  31.893019  [2]
# 2                      0.047051                       0.032191                          0.899345                            0.780525                     0.102889                     0.031147            0.049857    2.0  976.1922607421875   475.8522033691406  1002.5944213867188    520.92822265625   1015.576904296875   495.1080017089844  1089.6236572265625   543.29150390625  1100.7110595703125  504.5792541503906  1123.61376953125  550.8467407226562  DLC_resnet50_Maternal_EPMDec28shuffle1_700000  C:\Users\killian\projects\DIBS\epm_data_csv_train\EPM-MCE-10DLC_resnet50_Maternal_EPMDec28shuffle1_700000.csv  EPM-MCE-10DLC_resnet50_Maternal_EPMDec28shuffle1_700000    6.0  1009.0856628417969  508.0181121826172  1095.1673583984375  523.9353790283203        False -27.697763  29.016086  [2]
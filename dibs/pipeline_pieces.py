import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple, Optional
from dibs import check_arg, config, feature_engineering
from dibs.logging_enhanced import get_current_function

# Models
from bhtsne import tsne as TSNE_bhtsne
# from cvae import cvae
from openTSNE import TSNE as OpenTsneObj
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE as TSNE_sklearn
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
import time

from dibs import logging_enhanced

logger = config.initialize_logger(__name__)


class WithRandomState(object):
    def __init__(self, random_state):
        # TODO: Figure out how to synchronize random state across models.  IMPORTANT FOR REPRODUCABILITY AND SOME FORMS OF SEARCH
        self._random_state = random_state

    @property
    def random_state(self): return self._random_state


class WithParams(object):

    # specify custom parameter checkers, especially useful if multiple types are valid, if only a certain range
    # for a numerical value is valid, or if only a specific set of strings is valid
    _param_checkers = dict()
    # Turn of parameter type checking completely if you want
    _check_parameter_types = True

    def set_params(self, params: Dict):

        # TODO: Derive checker from type of previous value.
        #       We can add custom checkers later if we need them.

        # The valid params are specified by all the attributes attached to the class that do not start with _
        old_params = self.get_params()
        assert not (invalid_keys := set(params.keys()) - set(old_params.keys())), \
            f'Recieved invalid parameter specifications for {self.__class__.__name__}: {invalid_keys}'

        if self._check_parameter_types:
            for param_name, old_value in old_params.items():
                if param_name in params:
                    value = params[param_name]
                    # check that new param has same type as old param?
                    if checker := self._param_checkers.get(param_name):
                        if callable(checker):
                            checker(value)
                        else:
                            raise RuntimeError(f'Expected callable custom parameter checker for {self.__class__.__name__}'
                                               f' when checking value associated with {param_name}. \n'
                                               f'Instead of a checker we got: {checker}.\n'
                                               f'Please put a function name or lambda definition instead.')
                    else: # use default checking method.  Will enforce all parameters are the same type as the
                          # defaults for this class, which might be hard coded, or might be parsed from config.ini
                          # by config.py
                        check_arg.ensure_type(value, type(old_value))

        for param_name, value in params.items():
            # set the parameters as names on this class
            self.__setattr__(param_name, value)

    def get_params(self) -> Dict:
        # Return all properties on this object that are not callable or hidden.
        # All such names are assumed parameters to the model.
        return {name: value for name, value in self.__dict__ if not name.startswith('_') and not callable(value)}


class WithStreamlitParamDialog(object):
    def st_param_dialogue(self):
        """ TODO: Implement default streamlit dialogue that says 'sorry, but we can only read the config',
                  and put a button here to read the config again...? Yes?? """
        # configuration = configparser.ConfigParser()
        # configuration.read(os.path.join(DIBS_BASE_PROJECT_PATH, config_file_name))

        raise NotImplementedError()


class FeatureEngineerer(object):
    """ Examples: Custom built feature engineering for each task"""

    _all_engineered_features = None # TODO: Force overriding _all_features

    def engineer_features(self, in_df: pd.DataFrame, average_over_n_frames: int) -> pd.DataFrame:
        raise NotImplementedError()


class HowlandFeatureEngineerer(FeatureEngineerer):
    """
    New features set created by the Howland Lab.
    """
    # Feature names
    intermediate_bodypart_avgForepaw = 'AvgForepaw'
    intermediate_bodypart_avgHindpaw = 'AvgHindpaw'
    feat_name_dist_forepawleft_nosetip = 'DistanceForepawLeftToNosetip'
    feat_name_dist_forepawright_nosetip = 'DistanceForepawRightToNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'DistanceForepawLeftToHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'DistanceForepawRightToHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'DistanceAvgHindpawToNosetip'
    feat_name_dist_AvgForepaw_NoseTip = 'DistanceAvgForepawToNosetip'
    feat_name_velocity_AvgForepaw = 'VelocityAvgForepaw'
    _all_engineered_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
    )

    def engineer_features(self, in_df: pd.DataFrame, average_over_n_frames) -> pd.DataFrame:
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
        df = in_df.astype({'frame': int}).sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawleft_nosetip)
        # 2
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawright_nosetip)
        # 3
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('HINDPAW_LEFT'), self.feat_name_dist_forepawLeft_hindpawLeft)
        # 4
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('HINDPAW_RIGHT'), self.feat_name_dist_forepawRight_hindpawRight)
        # 5, 6
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgForepaw)
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'), config.get_part('HINDPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgHindpaw)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgHindpaw_Nosetip)

        # 7
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('NOSETIP'), output_feature_name=self.feat_name_dist_AvgForepaw_NoseTip)

        # 8
        # df = feature_engineering.attach_velocity_of_feature(df, 'AvgForepaw', 1/config.VIDEO_FPS, self.feat_name_velocity_AvgForepaw)
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_bodypart_avgForepaw, action_duration=1 / config.VIDEO_FPS, output_feature_name=self.feat_name_velocity_AvgForepaw)

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
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method, average_over_n_frames)
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








class Embedderer(WithRandomState, WithParams):
    """ Examples: pca, tsne, umap """
    _model = None
    _n_reduced_dimensions = -1
    def embed(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError()

    def _train(self, data: pd.DataFrame):
        """ Some implementations can be trained and the model can be used later
        For example OpenTSNE allows for reusing of the embedding to add new data.
        Technially this could be used as a classifier, by reusing the OpenTSNE and
        a pre-trained clustering algorithm, although this may not scale well and would
        be much slower at classifying new data than a pre trained classifier.
        """
        raise NotImplementedError()


class TSNE(Embedderer):
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

    def set_params(self, params):
        # TODO: Genearliz set_params so we don't have to repeat ourselves so much.
        self.tsne_implementation = params.get('tsne_implementation')
        self.tsne_n_components = params.get('tsne_n_components')
        self.tsne_n_iter = params.get('tsne_n_iter')
        self.tsne_early_exaggeration = params.get('tsne_early_exaggeration')
        self.tsne_n_jobs = params.get('tsne_n_jobs')
        self.tsne_verbose = params.get('tsne_verbose')
        self.tsne_init = params.get('tsne_init')
        self._tsne_perplexity = params.get('_tsne_perplexity') # TODO: Is this what we call it everywhere?
        self.tsne_learning_rate = params.get('tsne_learning_rate')

    def get_params(self):
        # TODO: Generalize get_params so we don't have to repeat ourselves so much
        return {
            self.tsne_implementation: {
                'tsne_n_components': self.tsne_n_components,
                'tsne_n_iter': self.tsne_n_iter,
                'tsne_early_exaggeration': self.tsne_early_exaggeration,
                'tsne_n_jobs': self.tsne_n_jobs,
                'tsne_verbose': self.tsne_verbose,
                'tsne_init': self.tsne_init,
                'tsne_perplexity': self.tsne_perplexity,
                'tsne_learning_rate': self.tsne_learning_rate,
            }
        }

    @property
    def tsne_perplexity_relative_to_num_features(self) -> float:
        """
        TODO: Move or lose
        Calculate the perplexity relative to the number of features.
        """
        return self.tsne_perplexity / len(self.all_features)

    @property
    def tsne_perplexity_relative_to_num_data_points(self) -> float:
        """
        TODO: Move or lose
        Calculate the perplexity relative to the number of data points.
        """
        if self.num_training_data_points == 0:
            logger.warning(f'{logging_enhanced.get_caller_function()}() is calling to get perplexity, '
                           f'but there are zero data points. Returning 0 for TSNE perplexity!')
            return 0
        return self.tsne_perplexity / self.num_training_data_points

    def tsne_perplexity(self) -> float:
        """
        Fetch t-SNE perplexity. Since perplexity has been reworked to not just be used as a nominal
        number but also as a fraction of an existing property (notably # of training data points),
        this property will sort that math out and return the actual perplexity number used in training.
        TODO:
        """
        perplexity = self._tsne_perplexity
        if isinstance(perplexity, str):
            check_arg.ensure_valid_perplexity_lambda(perplexity)
            perplexity = eval(perplexity)(self)
        check_arg.ensure_type(perplexity, float)
        return perplexity

    # TSNE Transformations
    # TODO: Original name, remove #def _train_tsne_get_dimension_reduced_data(self, df: pd.DataFrame) -> np.ndarray:
    def embed(self, df):
        """
        TODO: elaborate

        :param df:
        :return:
        """
        # Check args
        check_arg.ensure_type(df, pd.DataFrame)
        logger.debug(f'Pre-TSNE info: Perplexity={self.tsne_perplexity} / Raw perplexity={self._tsne_perplexity} / num_training_data_points={self.num_training_data_points} / number of df_features_train={len(self.df_features_train)} / number of df_features_train_scaled={len(self.df_features_train_scaled)}')
        # Execute
        start_time = time.perf_counter()
        logger.debug(f'Now reducing data with {self.tsne_implementation} implementation...')
        if self.tsne_implementation == 'SKLEARN':
            arr_result = TSNE_sklearn(
                n_components=self.tsne_n_components,
                perplexity=self.tsne_perplexity,
                early_exaggeration=self.tsne_early_exaggeration,
                learning_rate=self.tsne_learning_rate,  # alpha*eta = n  # TODO: low: follow up with this
                n_iter=self.tsne_n_iter,
                # n_iter_without_progress=300,
                # min_grad_norm=1e-7,
                # metric="euclidean",
                init=self.tsne_init,
                verbose=self.tsne_verbose,
                random_state=self.random_state,
                # method='barnes_hut',
                # angle=0.5,
                n_jobs=self.tsne_n_jobs,
            ).fit_transform(df.values)
        elif self.tsne_implementation == 'BHTSNE':
            arr_result = TSNE_bhtsne(
                df,
                dimensions=self.tsne_n_components,
                perplexity=self.tsne_perplexity,
                theta=0.5,
                rand_seed=self.random_state,
            )
        elif self.tsne_implementation == 'OPENTSNE':
            tsne = OpenTsneObj(
                **{
                    k:v for k,v in dict(
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
                    ).items()
                    if v is not None # NOTE: Providing None values allows for OpenTSNE to do optimization itself
                }
            )
            arr_result = tsne.fit(df.values)
        else:
            err = f'Invalid TSNE source type fell through the cracks: {self.tsne_implementation}'
            logging_enhanced.log_then_raise(err, logger, RuntimeError)
        end_time = time.perf_counter()
        logger.info(f'Number of seconds it took to train TSNE ({self.tsne_implementation}): '
                    f'{round(end_time- start_time, 1)} seconds (# rows of data: {arr_result.shape[0]}).')
        return arr_result


class ISOMAP(Embedderer):
    """ TODO: What is Isomap? """
    # TODO: Isomap implementation
    isomap_n_neighbors: int = 7  # TODO: low:
    def train(self, data: pd.DataFrame):
        logger.debug(f'Reducing dimensions using ISOMAP now...')
        isomap = Isomap(
            n_neighbors=self.isomap_n_neighbors,
            n_components=self.tsne_n_components,
            eigen_solver='auto',
            tol=0,
            max_iter=5000,
            path_method='auto',
            neighbors_algorithm='auto',
            n_jobs=self.tsne_n_jobs,
            metric='minkowski',
            p=2,
            metric_params=None)
        arr_result = isomap.fit_transform(data[self.all_features_list].values)
        return arr_result



class UMAP(Embedderer):
    """ Uniform Manifold __ __ """
    umap_n_neighbors: int = 5
    umap_learning_rate: float = 1.0
    # TODO: UMAP


class CVAE(Embedderer):
    """ Variational Auto Encoder """
    # TODO: CVAE
    cvae_num_steps: int = 1000  # TODO: low: arbitrary default
    def train(self, data: pd.DataFrame):
        logger.debug(f'Reducing dims using CVAE now...')
        data_array = data[self.all_features_list].values
        embedder = cvae.CompressionVAE(
            data_array,
            train_valid_split=0.75,
            dim_latent=2,
            iaf_flow_length=5,
            cells_encoder=None,
            initializer='orthogonal',
            batch_size=64,
            batch_size_test=64,
            logdir='temp',
            feature_normalization=True,
            tb_logging=False
        )
        embedder.train(
            learning_rate=0.001,
            num_steps=self.cvae_num_steps,
            dropout_keep_prob=0.75,
            overwrite=True,
            test_every=100,
            lr_scheduling=True,
            lr_scheduling_steps=5,
            lr_scheduling_factor=5,
            lr_scheduling_min=1e-05,
            checkpoint_every=2000,
        )

        arr_result = embedder.embed(data_array)

        return arr_result



class LocallyLinearDimReduc(Embedderer):
    LLE_method: str = 'standard'
    LLE_n_neighbors: int = 5
    def _locally_linear_dim_reduc(self, df: pd.DataFrame) -> np.ndarray:
        logger.debug(f'Reducing dims using LocallyLinearEmbedding now...')
        data_arr = df[list(self.all_features)].values
        local_line = LocallyLinearEmbedding(
            n_neighbors=self.LLE_n_neighbors,
            n_components=self.tsne_n_components,
            n_jobs=self.tsne_n_jobs,
            random_state=self.random_state,
            reg=1E-3,
            eigen_solver='auto',
            tol=1E-6,
            max_iter=100,
            method='standard',
            hessian_tol=1E-4,
            modified_tol=1E-12,
            neighbors_algorithm='auto',
        )

        arr_result = local_line.fit_transform(data_arr)
        return arr_result




















class Clusterer(WithRandomState, WithParams):
    """ Examples: gmm, dbscan, spectral_clustering """
    _model = None
    def train(self, df: pd.DataFrame):
        raise NotImplementedError()

    def predict(self, df: pd.DataFrame) -> np.ndarray: # TODO: array or df??
        """ TODO: Is input df or np.ndarray?? """
        return self._model.predict(df)


class GMM(Clusterer):

    # TODO: Re-jigger parameter setting etc.
    # GMM
    gmm_n_components, gmm_covariance_type, gmm_tol, gmm_reg_covar = None, None, None, None
    gmm_max_iter, gmm_n_init, gmm_init_params = None, None, None
    gmm_verbose: int = config.gmm_verbose
    gmm_verbose_interval: int = config.gmm_verbose_interval

    def set_params(self, params):
        """ Set params, use fallback dict if provided (for example provide the config dict as fallback) """
        self.gmm_n_components = params.get('gmm_n_components') or self.gmm_n_components
        self.gmm_covariance_type = params.get('gmm_covariance_type') or self.gmm_covariance_type
        self.gmm_tol = params.get('gmm_tol') or self.gmm_tol
        self.gmm_reg_covar = params.get('gmm_reg_covar') or self.gmm_reg_covar
        self.gmm_max_iter = params.get('gmm_max_iter') or self.gmm_max_iter
        self.gmm_n_init = params.get('gmm_n_init') or self.gmm_n_init
        self.gmm_init_params = params.get('gmm_init_params') or self.gmm_init_params
        self.gmm_verbose = params.get('gmm_verbose') or self.gmm_verbose
        self.gmm_verbose_interval = params.get('gmm_verbose_interval') or self.gmm_verbose_interval

    def get_params(self):
        return {
            'gmm_n_components': self.gmm_n_components,
            'gmm_covariance_type': self.gmm_covariance_type,
            'gmm_tol': self.gmm_tol,
            'gmm_reg_covar': self.gmm_reg_covar,
            'gmm_max_iter': self.gmm_max_iter,
            'gmm_n_init': self.gmm_n_init,
            'gmm_init_params': self.gmm_init_params,
            'gmm_verbose': self.gmm_verbose,
            'gmm_verbose_interval': self.gmm_verbose_interval,
        }

    def train(self, df):
        self._model = GaussianMixture(
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
        ).fit(df.values)


class SPECTRAL(Clusterer):
    pass

class DBSCAN(Clusterer):
    pass












class CLF(WithRandomState, WithParams):
    """ Examples: SVM, random forest, neural network """
    _model = None

    def st_params_dialogue(self):
        """ AARONT: TODO: Decide on streamlit interface with all the model parameters. """
        raise NotImplementedError()

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        raise NotImplementedError()

    def predict(self, df: pd.DataFrame):
        raise NotImplementedError()


class SVM(CLF):

    # Classifier: SVM
    classifier_verbose: int = config.CLASSIFIER_VERBOSE
    svm_c, svm_gamma = config.svm_c, config.svm_gamma
    svm_probability, svm_verbose = config.svm_probability, config.svm_verbose

    def train(self, X, y):
        """
        Train classifier on non-test-assigned data from the training data set.
        For any kwarg that does not take it's value from the it's own Pipeline config., then that
        variable
        """
        clf = SVC(
            C=self.svm_c,
            gamma=self.svm_gamma,
            probability=self.svm_probability,
            verbose=bool(self.classifier_verbose),
            random_state=self.random_state,
            cache_size=500,  # TODO: LOW: add variable to CONFIG.INI later? Measured in MB.
            max_iter=-1,
        )
        self._model=clf


class RANDOMFOREST(CLF):
    # Classifier: Random Forest
    classifier_verbose: int = config.CLASSIFIER_VERBOSE
    rf_n_estimators: int = config.rf_n_estimators
    rf_n_jobs: int = config.rf_n_jobs
    rf_verbose = config.rf_verbose

    def train(self, X, y):
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

        # Fit classifier to non-test data
        logger.debug(f'Training {self.__class__} classifier now...')
        clf.fit(X=X, y=y)
        self._model=clf



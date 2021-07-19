import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple, Optional
from collections import defaultdict
from dibs import check_arg, config
from dibs.logging_enhanced import get_current_function
from dibs.feature_engineering import distance, velocity, average

# Models
# from bhtsne import tsne as TSNE_bhtsne # Aaron on Ferrari; June 6th/2021: Does not want to install, but we don't use this anymore anyways
# from cvae import cvae
from openTSNE import TSNE as OpenTsneObj
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE as TSNE_sklearn
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import time

from dibs import logging_enhanced
from dibs.feature_engineering import integrate_df_feature_into_bins

logger = config.initialize_logger(__name__)


class WithParams(object):

    # specify custom parameter checkers, especially useful if multiple types are valid, if only a certain range
    # for a numerical value is valid, or if only a specific set of strings is valid
    _param_checkers = dict()
    # Turn of parameter type checking completely if you want
    _check_parameter_types = True

    def __init__(self, params: Dict=None):
        if params is not None:
            self.set_params(params)

    def set_params(self, params: Dict):
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
        return {name: value for name in dir(self) if not name.startswith('_') and not callable(value := getattr(self, name))}

    def params_as_string(self) -> str:
        return '\n'.join([f'{name}: value' for name, value in self.get_params().items()])


class WithStreamlitParamDialog(object):
    def st_param_dialogue(self):
        """ TODO: Implement default streamlit dialogue that says 'sorry, but we can only read the config',
                  and put a button here to read the config again...? Yes?? """
        # 0. Does this work at all?  Do this without crashing.
        # 1. Get parameter values from this class.  Those are the defaults.  GET THE DEFAULTS FROM A NEW INSTANTIATION. Lol
        # 2. Display the name of each, and a dialog for entering new values
        # 3. BONUS: Can we pick up doc strings attached to parameters and display these as extra info?
        raise NotImplementedError()


class FeatureEngineerer(object):
    """ Examples: Custom built feature engineering for each task"""

    random_state = config.FEATURE_ENGINEERER.random_state # Controls train/test split.... TODO: Make this better

    @property
    def _intermediate_feature_specs(self): return [] # empty list if user doesn't specify

    @property
    def _real_feature_specs(self): raise NotImplementedError('You must specify a list of _real_feature_specs when you implement a feature engineering class')

    @property
    def _kwargs(self): return dict() # Use to pass any args to feature eng functions TODO: Might have to allow more flexible args
    # Empty dict if user does not define

    def __init__(self):
        self._map_feature_to_integrate_method = dict() # Map of aggregators, populated by engineer_features method

    @property
    def _all_engineered_features(self):
        return (self._extract_name_from_spec(spec) for spec in self._real_feature_specs)

    @staticmethod
    def _extract_name_from_spec(spec, intermediate=False):
        func = spec[0]
        arg_names = spec[1:]
        arg_names_formatted = '_'.join(arg_names)
        basename = f'{func.__name__}_{arg_names_formatted}'
        if intermediate:
            return 'intermediate_feat_'+basename
        else:
            return basename

    def engineer_features(self, in_df: pd.DataFrame, average_over_n_frames: int = -1) -> pd.DataFrame:

        def _compile_feature_def(output_col_name, func, arg_names, intermediate_feature=False):
            if not callable(func):
                if intermediate_feature:
                    msg = f'Failed to engineer intermediate feature {output_col_name}. ' \
                          f'Second part of an _intermediate_feature_specs entry should be a callable (a function object)'
                else:
                    msg = f'Failed to engineer feature {output_col_name}. ' \
                          f'First part of a _real_feature_specs entry should be a callable (a function object)'
                logger.error(msg)
                raise RuntimeError(msg)
            logger.debug(f'Engineering model input feature: {output_col_name}. Function {func.__name__} will be applied to columns: {arg_names}')

            #     logger.error(msg)
            #     raise RuntimeError(msg)
            # logger.debug(f'Engineering intermediate feature: {output_col_name}')

            # args = (in_df[name].values for name in arg_names)
            # HACK: Have to extract the real x/y arg names because of how things are done earlier
            args = []
            for arg_name in arg_names:
                if arg_name in in_df:
                    args.append(in_df[arg_name].values)
                elif (x_name := arg_name+'_x') in in_df and (y_name := arg_name+'_y') in in_df:
                    args.append(in_df[[x_name, y_name]].values)
                else:
                    msg = 'When engineering a feature, all supplied arguments should refer to columns in the source data frame (the input data).\n' \
                          f'\tProvided columns: {arg_names} \n' \
                          f'\tValid columns: {in_df.columns}'
                    logger.error(msg)
                    raise RuntimeError(msg)


            # arr, aggregate_strat = func(*args, **self._kwargs)
            arr, aggregate_strat = func(*args) #, kwargs=self._kwargs) # TODO: Pass kwargs
            # HACK: Add the '_x' '_y' ...
            if len(arr.shape) >= 2 and arr.shape[1] == 2:
                output_col_name = [output_col_name+'_x', output_col_name+'_y']
            in_df[output_col_name] = arr
            # HACK: Have to use tuple in case this is a list
            #       SUPER JANK!
            if isinstance(output_col_name, list):
                for output_col_name in output_col_name:
                    self._map_feature_to_integrate_method[output_col_name] = aggregate_strat
            else:
                self._map_feature_to_integrate_method[output_col_name] = aggregate_strat

        for spec in self._intermediate_feature_specs:
            # user should ensure _intermediate_feautre_specs are topo ordered
            output_col_name = self._extract_name_from_spec(spec, intermediate=True)
            func = spec[0]
            arg_names = spec[1:]
            _compile_feature_def(output_col_name, func, arg_names, intermediate_feature=True)

        for spec in self._real_feature_specs:
            output_col_name = self._extract_name_from_spec(spec)
            func = spec[0]
            arg_names = spec[1:] # Should be in df
            _compile_feature_def(output_col_name, func, arg_names)

        if average_over_n_frames > 0:
            check_arg.ensure_int(average_over_n_frames)
            logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(in_df)}')
            in_df = integrate_df_feature_into_bins(in_df, self._map_feature_to_integrate_method, average_over_n_frames)
            logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(in_df)}')

        return in_df


class NeoHowlandFeatureEngineering(FeatureEngineerer):
    """
    New features set created by the Howland Lab.
    """

    class c:
        FOREPAW_LEFT = config.get_part('FOREPAW_LEFT')
        FOREPAW_RIGHT = config.get_part('FOREPAW_RIGHT')
        HINDPAW_LEFT = config.get_part('HINDPAW_LEFT')
        HINDPAW_RIGHT = config.get_part('HINDPAW_RIGHT')
        NOSETIP = config.get_part('NOSETIP')

    _intermediate_feature_specs = [
        # 1
        (average, c.FOREPAW_LEFT, c.FOREPAW_RIGHT),
        # 2
        (average, c.HINDPAW_LEFT, c.HINDPAW_RIGHT),
        # 3...
    ]

    inter_names = [FeatureEngineerer._extract_name_from_spec(spec, intermediate=True)
                   for spec in _intermediate_feature_specs]

    _real_feature_specs = [
        (distance, c.FOREPAW_LEFT, c.FOREPAW_RIGHT),

        (distance, c.FOREPAW_LEFT, c.NOSETIP),
        (distance, c.FOREPAW_RIGHT, c.NOSETIP),

        (distance, c.FOREPAW_RIGHT, c.HINDPAW_RIGHT),
        (distance, c.FOREPAW_LEFT, c.HINDPAW_RIGHT),

        (distance, inter_names[0], c.NOSETIP), # avg of fore paws
        (distance, inter_names[1], c.NOSETIP), # avg of hind paws

        # TODO: velocity
        # df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_bodypart_avgForepaw, action_duration=1 / config.VIDEO_FPS, output_feature_name=self.feat_name_velocity_AvgForepaw)

    ]


class Embedder(WithParams):
    """ Examples: pca, tsne, umap """
    _model = None

    @property
    def n_components(self) -> int:
        """ Embedders must override n_components """
        raise NotImplementedError()

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

    def metrics(self):
        raise NotImplementedError()

class TSNE(Embedder):
    # TSNE
    implementation: str = config.TSNE.implementation
    n_components: int = config.TSNE.n_components
    n_iter: int = config.TSNE.n_iter
    early_exaggeration: float = config.TSNE.early_exaggeration
    n_jobs: int = config.TSNE.n_jobs  # n cores used during process
    verbose: int = config.TSNE.verbose
    init: str = config.TSNE.init
    make_this_better_perplexity: Union[float, str] = config.TSNE.perplexity
    learning_rate: float = config.TSNE.learning_rate
    random_state = config.EMBEDDER.random_state

    # Non settable.  Not considered by set_params/get_params
    _num_training_data_points: int = None # must be set at runtime
    _num_training_features: int = None

    # TSNE Transformations
    # TODO: Original name, remove #def _train_tsne_get_dimension_reduced_data(self, df: pd.DataFrame) -> np.ndarray:
    def embed(self, df):
        """
        TODO: elaborate

        :param df:
        :return:
        """
        self._num_training_data_points = len(df)
        self._num_training_features = len(df.columns)
        # Check args
        check_arg.ensure_type(df, pd.DataFrame)
        logger.debug(f'Pre-TSNE info: Perplexity={self.perplexity} / Raw perplexity={self.make_this_better_perplexity} / num_training_data_points={self._num_training_data_points} / number of df_features_train={self._num_training_features} / number of df_features_train_scaled=TODO: Is this value meaningful?')
        # Execute
        start_time = time.perf_counter()
        logger.debug(f'Now reducing data with {self.implementation} implementation...')
        if self.implementation == 'SKLEARN':
            arr_result = TSNE_sklearn(
                n_components=self.n_components,
                perplexity=self.perplexity,
                early_exaggeration=self.early_exaggeration,
                learning_rate=self.learning_rate,  # alpha*eta = n  # TODO: low: follow up with this
                n_iter=self.n_iter,
                # n_iter_without_progress=300,
                # min_grad_norm=1e-7,
                # metric="euclidean",
                init=self.init,
                verbose=self.verbose,
                random_state=self.random_state,
                # method='barnes_hut',
                # angle=0.5,
                n_jobs=self.n_jobs,
            ).fit_transform(df.values)
        # elif self.tsne_implementation == 'BHTSNE':
        #     arr_result = TSNE_bhtsne(
        #         df,
        #         dimensions=self.n_components,
        #         perplexity=self.perplexity,
        #         theta=0.5,
        #         rand_seed=self.random_state,
        #     )
        elif self.implementation == 'OPENTSNE':
            tsne = OpenTsneObj(
                **{
                    k:v for k,v in dict(
                        n_components=self.n_components,
                        perplexity=self.perplexity,
                        learning_rate=self.learning_rate,
                        early_exaggeration=self.early_exaggeration,
                        early_exaggeration_iter=250,  # TODO: med: review
                        n_iter=self.n_iter,
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
                        n_jobs=self.n_jobs,
                        # affinities=None,
                        # neighbors="auto",
                        negative_gradient_method='bh',  # Note: default 'fft' does not work with dims >2
                        # callbacks=None,
                    ).items()
                    if v is not None # NOTE: Providing MISSING values allows for OpenTSNE to do optimization itself
                }
            )
            arr_result = tsne.fit(df.values)
        else:
            err = f'Invalid TSNE source type fell through the cracks: {self.implementation}'
            logger.error(err)
            raise RuntimeError()
        end_time = time.perf_counter()
        logger.info(f'Number of seconds it took to train TSNE ({self.implementation}): '
                    f'{round(end_time- start_time, 1)} seconds (# rows of data: {arr_result.shape[0]}).')
        return arr_result

    @property
    def _tsne_perplexity_relative_to_num_features(self) -> float:
        """
        TODO: Move or lose
        Calculate the perplexity relative to the number of features.
        """
        return self.perplexity / self._num_training_features

    @property
    def _tsne_perplexity_relative_to_num_data_points(self) -> float:
        """
        TODO: Move or lose
        Calculate the perplexity relative to the number of data points.
        """
        if self._num_training_data_points == 0:
            logger.warning(f'{logging_enhanced.get_caller_function()}() is calling to get perplexity, '
                           f'but there are zero data points. Returning 0 for TSNE perplexity!')
            return 0
        return self.perplexity / self._num_training_data_points

    @property
    def perplexity(self) -> float:
        """
        Fetch t-SNE perplexity. Since perplexity has been reworked to not just be used as a nominal
        number but also as a fraction of an existing property (notably # of training data points),
        this property will sort that math out and return the actual perplexity number used in training.
        TODO:
        """
        perplexity = self.make_this_better_perplexity
        if isinstance(perplexity, str):
            check_arg.ensure_valid_perplexity_lambda(perplexity)
            perplexity = eval(perplexity)(self)
        check_arg.ensure_type(perplexity, float)
        return perplexity


class ISOMAP(Embedder):
    """ TODO: What is Isomap? """
    # TODO: Isomap implementation
    # TODO: ADD TO CONFIG?
    n_neighbors: int = 7  # TODO: low:
    n_jobs: int = 1
    n_components: int = 2 # TODO: I have no reason for picking this value!!!

    def embed(self, data):
        logger.debug(f'Reducing dimensions using ISOMAP now...')
        isomap = Isomap(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            eigen_solver='auto',
            tol=0,
            max_iter=5000,
            path_method='auto',
            neighbors_algorithm='auto',
            n_jobs=self.n_jobs,
            metric='minkowski',
            p=2,
            metric_params=None)
        arr_result = isomap.fit_transform(data.values)
        return arr_result


class UMAP(Embedder):
    """ Uniform Manifold __ __ """
    n_neighbors: int = 5
    learning_rate: float = 1.0
    n_components: int = 2 # TODO: No reason to pick this number
    n_jobs: int = 1
    # TODO: UMAP

    def embed(self, data):
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            low_memory=False,
        )

        arr_result = reducer.fit_transform(data.values)
        return arr_result


class CVAE(Embedder):
    """ Variational Auto Encoder """
    # TODO: CVAE
    num_steps: int = 1000  # TODO: low: arbitrary default
    n_components: int = 2  # TODO: No reason to pick this

    def embed(self, data: pd.DataFrame):
        logger.debug(f'Reducing dims using CVAE now...')
        data_array = data.values
        embedder = cvae.CompressionVAE(
            data_array,
            train_valid_split=0.75,
            dim_latent=self.n_components,
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
            num_steps=self.num_steps,
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


class LocallyLinearDimReducer(Embedder):
    n_neighbors: int = 5
    n_components: int = 2 # TODO: I have no reason to chose this!!!
    n_jobs: int = 1 # TODO: Could be more??
    random_state = config.EMBEDDER.random_state

    def embed(self, df: pd.DataFrame) -> np.ndarray:
        logger.debug(f'Reducing dims using LocallyLinearEmbedding now...')
        data_arr = df.values
        local_line = LocallyLinearEmbedding(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            n_jobs=self.n_jobs,
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




















class Clusterer(WithParams):
    """ Examples: gmm, dbscan, spectral_clustering """
    _model = None

    def train(self, df: pd.DataFrame):
        raise NotImplementedError()

    def predict(self, df: pd.DataFrame) -> np.ndarray: # TODO: array or df??
        """ TODO: Is input df or np.ndarray?? """
        return self._model.predict(df)


class GMM(Clusterer):
    n_components = config.GMM.n_components
    covariance_type = config.GMM.covariance_type
    tol = config.GMM.tol
    reg_covar = config.GMM.reg_covar
    max_iter = config.GMM.max_iter
    n_init = config.GMM.n_init
    init_params = config.GMM.init_params
    verbose: int = config.GMM.verbose
    verbose_interval: int = config.GMM.verbose_interval
    random_state = config.CLUSTERER.random_state

    def train(self, df):
        self._model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
            random_state=self.random_state,
        ).fit(df.values)


class SPECTRAL(Clusterer):
    pass


class DBSCAN(Clusterer):
    pass












class CLF(WithParams):
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
    classifier_verbose: int = config.CLASSIFIER.verbose
    c, gamma = config.SVM.c, config.SVM.gamma
    probability, verbose = config.SVM.probability, config.SVM.verbose
    random_state = config.CLASSIFIER.random_state

    def train(self, X, y):
        """
        Train classifier on non-test-assigned data from the training data set.
        For any kwarg that does not take it's value from the it's own Pipeline config., then that
        variable
        """
        clf = SVC(
            C=self.c,
            gamma=self.gamma,
            probability=self.probability,
            verbose=bool(self.classifier_verbose),
            random_state=self.random_state,
            cache_size=500,  # TODO: LOW: add variable to CONFIG.INI later? Measured in MB.
            max_iter=-1,
        )
        self._model = clf

    def predict(self, df):
        raise NotImplementedError()


class RANDOMFOREST(CLF):
    n_estimators: int = config.RANDOMFOREST.n_estimators
    n_jobs: int = config.RANDOMFOREST.n_jobs
    verbose = config.RANDOMFOREST.verbose
    random_state = config.CLASSIFIER.random_state

    _param_checkers = dict(
        n_estimators=lambda v: check_arg.ensure_int(v) and v > 0,
        n_jobs=lambda v: check_arg.ensure_int(v) and v > 0,
        verbose=lambda v: check_arg.ensure_int(v) and v >= 0,
    )

    def __init__(self, params=None):
        super().__init__(params)
        self._model = None

    def train(self, X, y):
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
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
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
        )

        # Fit classifier to non-test data
        clf.fit(X=X, y=y)
        self._model = clf

    def predict(self, df):
        return self._model.predict(df)


# class DimReducer(WithRandomState, WithParams):
#     # Eg SVD, PCA, kPCA
#     _model = None
#     def st_params_dialogue(self):
#         """ AARONT: TODO: Decide on streamlit interface with all the model parameters. """
#         raise NotImplementedError()

#     def train(self, X):
#         raise NotImplementedError()
    
#     def explained_variance(self):
#         raise NotImplementedError()

class PrincipalComponents(Embedder):

    n_components = config.PrincipalComponents.n_components
    svd_solver = config.PrincipalComponents.svd_solver
    random_state = config.EMBEDDER.random_state

    def embed(self, X):
        pca = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state
        )
        logger.debug(f'Training {self.__class__} Dimensionality Reducer now...')
        X_reduced = pca.fit_transform(X.values)
        self._model = pca
        return X_reduced

    def metrics(self):
        return {
            'explained_variance': self._model.explained_variance_ratio_
        }

import glob
import sys
import pandas as pd
import numpy
import os
import time
import pickle
from collections import namedtuple, defaultdict
import seaborn as sb
import matplotlib.pyplot as plt
from itertools import cycle

from fnmatch import fnmatch
from sklearn import tree
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.ensemble import RandomForestClassifier
# TODO: Import xgboost-gpu and use it
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

Results = namedtuple('Results', 'model precision recall f1 c_mat y_pred_test y_test test_dataset_i total_time')

# global_data_files = glob.glob('data/May1-2022-odourspan-6-object-open-field/SimBA_NOD/targets_inserted/*.csv')

args = sys.argv
print('Recieved args:', args)
training_input_path = args[1]
if not os.path.isdir(training_input_path):
    raise RuntimeError(f'Expected first argument to be an input path pull of csvs to glob!!! Recieved this instead: {training_input_path}')
# TODO: Check all the training input csvs have the targets inserted!

to_label_input_path = args[2]
if not os.path.isdir(to_label_input_path):
    raise RuntimeError(f'Expected the second argument to be a path to files that do not yet have targets inserted!')

output_path = args[3]
if os.path.isdir(output_path):
    raise RuntimeError(f'Expected the third argument to be an output path that does not exist yet! '
                       f'WARNING: The output path alread exists!  I do not want to clober your data!!'
                       f' Put it in an empty dir!!!.  output_path: {output_path}')
os.makedirs(output_path)

global_data_files = glob.glob(os.path.join(training_input_path, '*.csv'))
print(f'global_data_files: {global_data_files}')
df = pd.read_csv(global_data_files[0], index_col=0)

global_to_label_files = glob.glob(os.path.join(to_label_input_path, '*.csv'))

odour_features = [
    f for f in df.columns
    if fnmatch(f, '*Odour*')
]

interaction_features = [
    'Interaction',
    'Probability_Interaction'
]

prob_features = [
    f for f in df.columns
    if fnmatch(f, '*prob*')
]

raw_dlc_features_only = [
    "Ear_left_p",
    "Ear_left_x",
    "Ear_left_y",

    "Ear_right_p",
    "Ear_right_x",
    "Ear_right_y",

    "Lat_left_p",
    "Lat_left_x",
    "Lat_left_y",

    "Lat_right_p",
    "Lat_right_x",
    "Lat_right_y",

    "Center_p",
    "Center_x",
    "Center_y",

    "Nose_p",
    "Nose_x",
    "Nose_y",

    "Tail_base_p",
    "Tail_base_x",
    "Tail_base_y",

    "Tail_end_x",
    "Tail_end_y",
    "Tail_end_p",
]

filter_p = use_raw_dlc_features = True

if filter_p:
    raw_dlc_features_only = [s for s in raw_dlc_features_only if '_p' not in s]

if use_raw_dlc_features:
    non_odour_non_prob_features = raw_dlc_features_only

else:
    non_odour_non_prob_features = sorted(set(df.columns) - set(
        odour_features + prob_features + interaction_features + raw_dlc_features_only
    ))
print(len(df.columns), len(odour_features), len(prob_features))



## Lets try our feature engineering?
# from dibs import feature_engineering # Can't import with python 3.6...... BOOOO
# TODO: etc








## AARONT: TODO: Odour was removed! Only use the interaction column now!
# global_y_features = [f'Odour{i}' for i in range(1, 7)]
# distance_features = [f'{y_feature}_Animal_1_distance' for y_feature in global_y_features]
# facing_features = [f'{y_feature}_Animal_1_facing' for y_feature in global_y_features]


# Y funcs
def Y_combine_odours(_df, expand_to_original_labels=False):
    if expand_to_original_labels:
        # Needed for hack accuracy calc when we use the Y post processor
        odour1 = _df['Odour1'] * 1
        odour2 = _df['Odour2'] * 2
        odour3 = _df['Odour3'] * 3
        odour4 = _df['Odour4'] * 4
        odour5 = _df['Odour5'] * 5
        odour6 = _df['Odour6'] * 6
    else:
        odour1 = _df['Odour1'] * 1
        odour2 = _df['Odour2'] * 1  # 2
        odour3 = _df['Odour3'] * 1  # 3
        odour4 = _df['Odour4'] * 1  # 4
        odour5 = _df['Odour5'] * 1  # 5
        odour6 = _df['Odour6'] * 1  # 6

    assert not any((odour1 > 0) & (odour2 > 0) & (odour3 > 0) & (odour4 > 0) & (odour5 > 0) & (odour6 > 0))

    combined = odour1 + odour2 + odour3 + odour4 + odour5 + odour6
    combined_df = pd.DataFrame(combined, columns=['combined_odours'])
    return [combined_df['combined_odours']]


def Y_all_odours(_df):
    return [_df[y_feature] for y_feature in global_y_features]

def Y_get_interaction(_df):
    return [_df['Interaction']]

# X funcs
def X_only_builtins(_df):
    return [_df[non_odour_non_prob_features]]


def X_pre_animal_distance_only(_df):
    return [_df[x_feature] for x_feature in distance_features]


def X_builtins_and_distance(_df):
    return [_df[non_odour_non_prob_features + [x_feature]] for x_feature in distance_features]


def X_builtins_and_distance_combined(_df):
    min_dists = _df[distance_features].min(axis=1)
    new_df = _df.copy()
    new_df['min_distance_to_any_object'] = min_dists
    return [new_df[non_odour_non_prob_features + ['min_distance_to_any_object']]]


def X_builtins_and_distance_and_facing(_df):
    return [_df[non_odour_non_prob_features + [x_feature_dist] + [x_feature_facing]]
            for x_feature_dist, x_feature_facing in zip(distance_features, facing_features)]


# TODO: Cross validation would be one video file against the rest.  To expensive for now.
class Dataset(object):
    def __init__(self, input_files, x_func, y_func, y_post_processor=None, i=None, cv=False):
        assert i is None or cv is False, 'Only one of i (idx for test dataset) or cv should be provided'
        assert i is not None or cv is True, 'Must provide at least one of i and cv'
        # TODO: Include under and over sampling args, those must also be put into the keys etc.
        self.data_files = input_files.copy()
        self.cv = cv
        self.i = i
        self.x_func = x_func
        self.y_func = y_func
        self.y_post_processor = y_post_processor

    # TODO: under/over sampling!!
    def get_dfs(self, _df, x_or_y=None):
        assert x_or_y in {'x', 'y'}
        funcs = {'x': self.x_func, 'y': self.y_func}

        # call the func on the dataframe
        _dfs = funcs[x_or_y](_df)

        assert isinstance(_dfs, list)
        assert all(isinstance(__df, (pd.DataFrame, pd.Series)) for __df in
                   _dfs), f'Failed with types: {[type(__df) for __df in _dfs]}'
        return _dfs  # list of dataframes with features or targets

    def init_data(self, i):
        raise NotImplemented('You must override init_data')

    def __iter__(self):

        if self.cv:
            dataset_idxs = range(len(self.data_files))
        else:
            dataset_idxs = [self.i]

        for i in dataset_idxs:
            training_paths, testing_paths = self.init_data(i=i)
            orig_train_df = pd.concat([pd.read_csv(f, index_col=0) for f in training_paths])
            orig_test_df = pd.concat([pd.read_csv(f, index_col=0) for f in testing_paths])
            # x_func and y_func are used to extract the features and targets from the dataframes

            x_train_dfs = self.get_dfs(orig_train_df, 'x')
            y_train_dfs = self.get_dfs(orig_train_df, 'y')
            x_test_dfs = self.get_dfs(orig_test_df, 'x')
            y_test_dfs = self.get_dfs(orig_test_df, 'y')

            x_len = len(x_train_dfs)
            y_len = len(y_train_dfs)

            print('x_len:', x_len, 'y_len:', y_len)

            from itertools import cycle
            for x_train, y_train, x_test, y_test in zip(
                    cycle(x_train_dfs), y_train_dfs,
                    cycle(x_test_dfs), y_test_dfs,
            ):
                yield x_train, y_train, x_test, y_test, i

    def __str__(self):
        if not self.cv:
            train_path_str = '\n\t'.join([x for i, x in enumerate(self.data_files) if i != self.i])
            test_path_str = '\n\t'.join(self.data_files[self.i:self.i + 1])
            paths_str = f'Training paths: {train_path_str}\nTesting paths: {test_path_str}\n'
        else:
            files_str = '\n\t'.join(self.data_files)
            paths_str = f'Data files (cv is active): {files_str}'
        return f'{self.__class__.__name__}: x_func: {self.x_func.__name__}; y_func: {self.y_func.__name__}\n' \
               f'Paths: {paths_str}'

    def key(self):
        """ For aggregating experiments and taking averages """
        return frozenset({
            ('x_func', self.x_func.__name__),
            ('y_func', self.y_func.__name__),
            ('cv', self.cv),
            ('i', self.i),
            ('data_files', self.data_files),
        })


class ScriptDatasetForLabeling(object):
    def __init__(self, input_files, x_func, y_func, y_post_processor=None):
        # TODO: Include under and over sampling args, those must also be put into the keys etc.
        self.data_files = input_files.copy()
        self.x_func = x_func
        self.y_func = y_func
        self.y_post_processor = y_post_processor

    # TODO: under/over sampling!!
    def get_dfs(self, _df, x_or_y=None):
        assert x_or_y in {'x', 'y'}
        funcs = {'x': self.x_func, 'y': self.y_func}

        # call the func on the dataframe
        _dfs = funcs[x_or_y](_df)

        assert isinstance(_dfs, list)
        assert all(isinstance(__df, (pd.DataFrame, pd.Series)) for __df in
                   _dfs), f'Failed with types: {[type(__df) for __df in _dfs]}'
        return _dfs  # list of dataframes with features or targets

    def __iter__(self):
        testing_paths = self.data_files
        orig_test_dfs = [pd.read_csv(f, index_col=0) for f in testing_paths]
        # x_func and y_func are used to extract the features and targets from the dataframes

        x_test_dfs = [self.get_dfs(test_df, 'x') for test_df in orig_test_dfs]

        print('x_len:', len(x_test_dfs))

        from itertools import cycle
        for x_test, orig_file_path in zip(
                x_test_dfs, testing_paths
        ):
            yield x_test, orig_file_path

    def __str__(self):
        test_path_str = '\n\t'.join([x for x in self.data_files])
        paths_str = f'Testing paths: {test_path_str}\n'
        return f'{self.__class__.__name__}: x_func: {self.x_func.__name__}; y_func: {self.y_func.__name__}\n' \
               f'Paths: {paths_str}'

    def key(self):
        """ For aggregating experiments and taking averages """
        return frozenset({
            ('x_func', self.x_func.__name__),
            ('y_func', self.y_func.__name__),
            ('data_files', self.data_files),
        })

# Data to load classes
class ProtoDataset(Dataset):
    def init_data(self, i):
        print('WARNING: ignoring i')
        assert not self.cv, 'Prototype dataset does not support cross validation'
        data_files = self.data_files.copy()
        training_paths = [data_files[0]]
        testing_paths = [data_files[1]]
        return training_paths, testing_paths


class FullDataset(Dataset):
    def init_data(self, i):
        data_files = self.data_files.copy()
        testing_paths = [data_files[i]]
        del data_files[i]
        training_paths = data_files[:]
        return training_paths, testing_paths


_df_global = None
y_pred_global = None

## FIRST CELL

def Y_post_processor_1_to_many_classes(_df, y_pred):
    """ given y_pred with binary values, assign a class to each based on the closest object """
    # TODO: The thing

    # TODO: Temp, testing
    #     global _df_global
    #     _df_global = _df
    #     global y_pred_global
    #     y_pred_global = y_pred
    # Indexes will come back in [0, n_obj] (inclusive range math notation)
    # So we +1 to assign the correct y label and
    ### TODO: I think Tim said Simba has something internally for doing this already???
    nearest_obj = _df[distance_features].values.argmin(axis=1) + 1
    return nearest_obj * y_pred  # y_pred is binary


class Experiment(object):
    def __init__(self, dataset, model_type, model_kwargs):
        self.started_running = False
        self.finished_running = False
        self.dataset = dataset
        self.model_type = model_type
        self.model_kwargs = model_kwargs

    def run(self):
        print(self)
        self.started_running = True
        self.finished_running = False
        self.results = []
        first_run = True
        for x_train, y_train, x_test, y_test, test_dataset_i in self.dataset:
            start = time.time()
            if first_run:
                print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
                first_run = False

            model_kwargs = self.model_kwargs.copy()
            scale_pos_weight = model_kwargs.get('scale_pos_weight')
            if scale_pos_weight == 'AARONT_balanced':
                assert isinstance(y_train, (
                numpy.ndarray, pd.Series)), f'Expected numpy array or pandas Series, got {y_train} instead'
                # HACK: We will match the sklearn interface and calculate weight balancing ourselves...
                model_kwargs['scale_pos_weight'] = (
                            (y_train == 0).sum() / (y_train >= 1).sum())  # * 0.5 # deweighting a bit

            model = self.model_type(**model_kwargs).fit(x_train, numpy.ravel(y_train))
            y_pred_test = model.predict(x_test)

            # TODO: This is a hack, we should have a model class that encapsulates this!
            if self.dataset.y_post_processor:  # SUPER HACK: Reach into the dataset for the post processor
                orig_df = pd.read_csv(self.dataset.data_files[test_dataset_i], index_col=0)
                y_pred_test = self.dataset.y_post_processor(orig_df, y_pred_test)
                # TODO: HACK: Have to use the y_func with this arg to get the correct labels.
                y_test = self.dataset.y_func(orig_df, expand_to_original_labels=True)[0]

            end = time.time()
            total_time = end - start

            # average='binary' means to calculate the f1 score of the class with label defaulted to 1
            # average='macro' means to calculate the f1 score of all the classes combined
            # average='micro' isn't useful
            # agerage='weighted' weights by the occurrance
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
            except:
                # must be multi class
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
            # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')

            c_mat = confusion_matrix(y_test, y_pred_test)

            print_importance(model)

            self.results.append(
                Results(model, precision, recall, f1, c_mat, y_pred_test, y_test, test_dataset_i, total_time)
            )
            print('\n---- start results ----')
            print('run time:', total_time)
            print('recall', recall)
            print('f1:', f1)
            print(c_mat)
            print('---- end results ----\n')

        self.finished_running = True

    def generate_output_df(self):
        # TODO: Annotate all the output dfs, and place them on the path!
        # Just predict the interactions, and insert two columns:
        # Probability_Interaction, Interaction
        # 1. Predict
        # 2. Get the probabilities
        #
        # Make the dataset that will be predicted:
        # class Dataset(object):
        #     def __init__(self, x_func, y_func, y_post_processor=None, i=None, cv=False):
        #         assert i is None or cv is False, 'Only one of i (idx for test dataset) or cv should be provided'
        #         assert i is not None or cv is True, 'Must provide at least one of i and cv'
        #         # TODO: Include under and over sampling args, those must also be put into the keys etc.
        #         self.data_files = global_data_files.copy()
        #         self.cv = cv
        #         self.i = i
        #         self.x_func = x_func
        #         self.y_func = y_func
        #         self.y_post_processor = y_post_processor

        to_label_dataset = ScriptDatasetForLabeling(
            global_to_label_files,
            x_func=self.dataset.x_func,
            y_func=self.dataset.y_func,
            y_post_processor=self.dataset.y_post_processor)

        print(f'About to label the following data: {to_label_dataset}')

        # HACKS: Just reach in and get the model!
        model = self.results[0][0]

        for x_test, orig_file_path in to_label_dataset:
            # print(f'orig_file_path: {orig_file_path}')
            # print(f'x_test: {x_test}; len(x_test): {len(x_test)}')
            y_pred_test = model.predict(x_test[0])
            y_pred_test_proba = model.predict_proba(x_test[0])
            y_pred_test_proba = y_pred_test_proba[:, 1]
            base_file_name = os.path.basename(orig_file_path)
            output_file_path = os.path.join(output_path, base_file_name)
            print(f'Currently labelling: {output_file_path}')
            # Create a dataframe from x_tests original plus some extra stuff
            out_df = pd.read_csv(orig_file_path, index_col=0)

            # AARONT: TEMP: Put our interaction column next to the existing one
            # out_df['Interaction'] = y_pred_test
            out_df['NEW_Interaction'] = y_pred_test
            out_df['Probability_Interaction'] = y_pred_test_proba
            out_df.to_csv(output_file_path)

            # # TODO: This is a hack, we should have a model class that encapsulates this!
            # if self.dataset.y_post_processor:  # SUPER HACK: Reach into the dataset for the post processor
            #     orig_df = pd.read_csv(self.dataset.data_files[test_dataset_i], index_col=0)
            #     y_pred_test = self.dataset.y_post_processor(orig_df, y_pred_test)
            #     # TODO: HACK: Have to use the y_func with this arg to get the correct labels.
            #     y_test = self.dataset.y_func(orig_df, expand_to_original_labels=True)[0]

    def explain(self):
        for res in self.results:
            print('\n---- start results ----')
            print('run time:', res.total_time)
            print('recall:', res.recall)
            print('f1:', res.f1)
            print(res.c_mat)
            print('---- end results ----\n')

    def __str__(self):
        return f'{self.__class__.__name__}: ' \
               f'\n\tDataset: {self.dataset};' \
               f'\n\tModel Type: {self.model_type};' \
               f'\n\tModel args: {self.model_kwargs}'

    def __repr__(self):
        return self.__str__()

    def save_state(self):
        raise NotImplemented('TODO: pickle with dill')

    def load_state(self):
        raise NotImplemented('TODO: load with dill')

    # TODO: define and produce the analysis stats afterwards

def print_importance(model):
    if isinstance(model, xgb.XGBClassifier):
        print(f'xgb stuff: boosting rounds: {model.get_num_boosting_rounds()};')
        for importance_type in ('weight', 'total_gain', 'total_cover'):
            importances = model.get_booster().get_score(importance_type=importance_type)
            features_string = '\n'.join([f'{name:<30}: {val}' for name,val in sorted(
                importances.items(),
                key=lambda tup: tup[1], reverse=True
            )])
            print(f'''
    (xgb XGBClassifier)    importance_type: {importance_type}; 
    feature_importances: 
    {features_string}
    ''')

        # xgb.plot_importance(model, importance_type='total_gain')
        plt.show()
    if isinstance(model, DTree):
        importance_vals = model.feature_importances_
        # importance_names = model.feature_names_in_ # Need scikit learn version 1.0
        importance_names = non_odour_non_prob_features # HACK: The names we used at the beginning
        features_string = '\n'.join([f'{name:<30}: {val:e}' for name,val in zip(importance_names, importance_vals)])
        print(f'''
(sklearn DecisionTree) 
feature_importances:
{features_string}
''')

class ResultsAggregator(object):
    pass  # TODO: Aggregate experiments and produce a report


def dict_or_raise(d):
    assert isinstance(d, dict), f'Expected dict, got: {d}'
    return True


class ExperimentExpander(object):
    # OKAY: The best possible case would be this class hooked up to a database, so we run the exps if we need to,
    #       otherwise we retrieve the results from disk.  Then we can just keep iterating on experiments easily
    #       and keep getting the results back.
    # TODO: Take a model type, set of arg dicts, and set of datasets, and expand the product of the exps
    def __init__(self, model_type_to_arg_dicts_dict, datasets):
        self.exps = [
            Experiment(dataset, model_type, model_args)
            for dataset in datasets
            for model_type in model_type_to_arg_dicts_dict
            for model_args in model_type_to_arg_dicts_dict[model_type]
            if dict_or_raise(model_args)
        ]
        print(len(self.exps))

    def run(self):
        for exp in self.exps:
            exp.run()


dataset_type = FullDataset
# dataset_type = ProtoDataset
global_i = 0
# global_i = 1
# global_i = 2

## TODO: NEXT: Write a dataset that combines the Y's, uses the facing/distance to the closest thing,
#              and/or

# dataset_combined_ys = dataset_type(global_data_files, X_only_builtins, Y_combine_odours, i=global_i)
#
# dataset_X_buitin_Y_separate = dataset_type(global_data_files, X_only_builtins, Y_all_odours, i=global_i)
## dataset_X_buitin_Y_separate._orig_train_df.describe()
#
# dataset_X_builtin_plus_distance_Y_separate = dataset_type(global_data_files, X_builtins_and_distance, Y_all_odours, i=global_i)
# dataset_X_builtin_plus_distance_Y_separate._orig_train_df.describe()

# dataset_X_builtin_plus_distance_and_facing_Y_separate = dataset_type(
#     global_data_files, X_builtins_and_distance_and_facing, Y_all_odours, i=global_i)

dataset_X_buitin_Y_separate = dataset_type(global_data_files, X_only_builtins, Y_get_interaction, i=global_i)

# OKAY: Actually train the things now.
# xgb_really_high_recall_set = dict(
#     n_jobs=14,
#     tree_method='gpu_hist',
#     # tree_method='hist',
#     scale_pos_weight='AARONT_balanced',
#     #### What was the value of 'i' when I did this?????
#     # (this is a retarded number, look at the actual confusion matrix!!)
#     # 100. is actually better than 1000. it seems, based on the f1 scores
#     # Yeah 1000 is aggressive.  However if I was doing this in a semi-supervised context,
#     # I would heavily prefer the higher recall!
#     gamma=1000.,  # default 0.0; min_split_loss (minimum loss reduction to create a split)
#     max_depth=2,
#     use_label_encoder=False,
#     objective='binary:logistic',
#     max_delta_step=1,
#     subsample=0.5,  # sample of the rows to use, sampled once every boosting iteration
#     sampling_method='gradient_based',  # allows very small subsample, as low as 0.1
#     grow_policy='lossguide',
# )


if use_raw_dlc_features:
    # one experiment, with interaction constraints:
    # ---- start results ----
    # run time: 0.6402802467346191
    # recall 0.788244766505636
    # f1: 0.495570741584409
    # [[2428 1730]
    #  [ 263  979]]
    # ---- end results ----
    #
    # Without interaction constraints:
    # ---- start results ----
    # run time: 0.693350076675415
    # recall 0.6086956521739131  <- Almost 20% lower recall.  With constraints was overall more aggressive
    # f1: 0.529597197898424
    # [[3301  857]
    #  [ 486  756]]
    # ---- end results ----
    interaction_constraints = defaultdict(list)
    for s in raw_dlc_features_only:
        parts = s.split('_')
        # prefix = f'{parts[0]}_{parts[1]}'
        prefix = parts[0]
        interaction_constraints[prefix].append(raw_dlc_features_only.index(s))
        # prefix = parts[1]
        # interaction_constraints[prefix].append(raw_dlc_features_only.index(s))
    print(interaction_constraints)
    interaction_constraints = list(interaction_constraints.values())
    print(interaction_constraints)
else:
    interaction_constraints = None

# xgb_params = xgb_high_recall_set
xgb_params = dict(
    n_jobs=14,
    tree_method='gpu_hist',
    # tree_method='exact', # More time but enumarates all possible splits
    #                 base_score=base_score,
    #                 scale_pos_weight='AARONT_balanced',
    scale_pos_weight='AARONT_balanced',
    # eta=0.3, # default 0.3; learning rate
    gamma=100,  # default 0.0; min_split_loss (minimum loss reduction to create a split)
    max_depth=4,  # default is 6; Experiments showed 2 or 3 worked best
    n_estimators=300, # The number of boosting rounds; default 100?
    # early_stopping_rounds=20, # TODO: Tune early stopping??
    # use_label_encoder=False,
    objective='binary:logistic',
    #                 eval_metric=eval_metric,
    #                 objective=objective,
    max_delta_step=1,  # Was set to 1; VERY IMPORTANT PARAMETER! Read description in comments elsewhere
    #                 subsample=subsample,
    subsample=0.5,  # sample of the rows to use, sampled once every boosting iteration
    sampling_method='gradient_based',  # allows very small subsample, as low as 0.1
    grow_policy='lossguide',
    # num_parallel_tree=10,
    # colsample_bytree=0.75, colsample_bylevel=0.75, colsample_bynode=0.75,
    # min_child_weight=1,
    interaction_constraints=interaction_constraints,
    asdf='not a param', # xgb does not raise an error for an unrecognized parameter...
)

# Turn this on if you want to compare against the method used by Simba
exp = Experiment(
    dataset_X_buitin_Y_separate,
    # RandomForestClassifier,
    DTree,
    dict(
        # n_estimators=2000,
        criterion='entropy',  # Gini is standard, shouldn't be a huge factor
        min_samples_leaf=2,
        max_features='sqrt',
        max_depth=15,  # LIMIT MAX DEPTH!!  Runtime AND generalization error should improve drastically
        #     ccp_alpha=0.005, # NEW PARAMETER, I NEED TO DEFINE MY EXPERIMENT SETUPS BETTER, AND STORE SOME RESULTS!!
        # Probably need to whip up a database again, that's the only way I have been able to navigate this in the past
        # Alternatively I could very carefully define my experiments, and then run them all in a batch and create a
        # meaningful report.  This is probably the best way to proceed.  It will lead to the most robust iteration
        # and progress.
        class_weight='balanced',  # balance weights at nodes based on class frequencies
    )
)
exp.run()

exp = Experiment(
    dataset_X_buitin_Y_separate,
    xgb.XGBClassifier,
    xgb_params
)
exp.run()

# output_df = exp.generate_output_df()










# TODO: A very large experiment and then get the marginals for each variable...
# TODO: Hook this up to DIBS so we can grab and play the videos of a model.
#       Start picking models, filtering/smoothing, and seeing what the actual results look like.
# TODO: MOST FIRST: Build the combined model as a transformation that happens in the... dataset class?
#                   So that a model is given the combined whys, but it's predictions are transformed into
#                   the per class predictions.
# TODO: AND after hooking up to DIBS, use our DIBS feature engineering code.

# expander = ExperimentExpander(
#     {
#         xgb.XGBClassifier: [xgb_really_high_recall_set] + [
#             dict(
#                 n_jobs=14,
#                 tree_method='gpu_hist',
#                 #                 base_score=base_score,
#                 #                 scale_pos_weight='AARONT_balanced',
#                 scale_pos_weight=scale_pos_weight,
#                 #                 eta=eta, # default 0.3; learning rate
#                 gamma=gamma,  # default 0.0; min_split_loss (minimum loss reduction to create a split)
#                 #                 scale_pos_weight=scale_pos_weight,
#                 max_depth=max_depth,  # default is 6
#                 use_label_encoder=False,
#                 objective='binary:logistic',
#                 #                 eval_metric=eval_metric,
#                 #                 objective=objective,
#                 max_delta_step=max_delta_step,  # Was set to 1
#                 #                 subsample=subsample,
#                 subsample=0.5,  # sample of the rows to use, sampled once every boosting iteration
#                 sampling_method='gradient_based',  # allows very small subsample, as low as 0.1
#                 grow_policy='lossguide',
#                 #                 num_parallel_tree=num_parallel_tree,
#                 min_child_weight=min_child_weight,
#
#             )
#             for scale_pos_weight in ('AARONT_balanced',)  # , 1)
#             for max_depth in (2, 4, 6)
#             #             for sampling_method in ('gradient_based', 'uniform') # with subsample=0.5, positive results in favour grad
#             #             for grow_policy in ('lossguide', 'depthwise') # null exp, might be meaningful to deeper trees
#             # TODO: for max_bin in (256, 512, 1024) # better continuous feature binning
#             #             for objective in ('binary:logistic', 'binary:hinge')
#             #             for eval_metric in ('logloss', 'error', 'error@0.2', 'error@0.8') # totally null exp
#             # max_delta_step is too important, make sure you test it against 0!
#             # max_delta_step - The maximum step size that a leaf node can take.
#             #     In practice, this means that leaf values can be no larger than max_delta_step * eta
#             for max_delta_step in (0, 1)  # max_delta_step has a large impact.  Why? What is it?
#             #             for eta in (0.1, 0.2, 0.3)
#             # gamme is used during prunning.  min_split_loss
#             for gamma in (0.0, 100.,)  # , 1000.,)
#             #             for subsample in (1, 0.5, 0.1)
#             #             for num_parallel_tree in (1, 10) # it is 39 seconds per model with 100 trees...
#             # # RF was a noop, I didn't use column sub sampling though.  Not necessary I think
#             #        for col_split_by_node in (0.75, )
#             # TODO: min_child_weight
#             #    For classification this gives the required sum of p*(1-p), where p is the probability
#             #        THE SUM of p*(1-p), which is 0.25 max.  You will need at least 4 * min_child_weight rows
#             #        to keep the split.
#             for min_child_weight in (0, 2)  # inconclusive experiments, just sanity checking
#             #             for base_score in (0.5, 0.1) # default 0.5, leave this alone because we are already weighting.
#             # TODO: EARLY STOPPING BASED ON A HOLDOUT SET!!!
#             # https://xgboost.readthedocs.io/en/latest/python/python_intro.html#early-stopping
#             # This works with both metrics to minimize (RMSE, log loss, etc.) and to maximize (MAP, NDCG, AUC).
#             # Note that if you specify more than one evaluation metric the last one in param['eval_metric']
#             # is used for early stopping. (saw someone do this with 'ams@0.15' in a competition)
#
#             # COULD have more permutations here
#         ],
#         # COULD have more model types here
#         DTree: [
#             dict(
#                 criterion='entropy',  # Gini is standard, shouldn't be a huge factor
#                 min_samples_leaf=2,
#                 max_features='sqrt',
#                 max_depth=15,  # LIMIT MAX DEPTH!!  Runtime AND generalization error should improve drastically
#                 #     ccp_alpha=0.005, # NEW PARAMETER, I NEED TO DEFINE MY EXPERIMENT SETUPS BETTER, AND STORE SOME RESULTS!!
#                 # Probably need to whip up a database again, that's the only way I have been able to navigate this in the past
#                 # Alternatively I could very carefully define my experiments, and then run them all in a batch and create a
#                 # meaningful report.  This is probably the best way to proceed.  It will lead to the most robust iteration
#                 # and progress.
#                 class_weight='balanced',  # balance weights at nodes based on class fequencies
#             )
#         ]
#     },
#     [
#         # COULD just have the dataset constructors here...
#         #         dataset_X_buitin_Y_separate, # Adding times distance and facing features is almost always superior
#         FullDataset(
#             x_func=X_builtins_and_distance_combined,
#             y_func=Y_combine_odours,
#             y_post_processor=Y_post_processor_1_to_many_classes,
#             i=global_i)
#         for i in range(len(global_data_files))
#     ] + [
#         dataset_X_builtin_plus_distance_Y_separate
#     ],
#     #         dataset_X_builtin_plus_distance_and_facing_Y_separate,
# )
#
# expander.run()


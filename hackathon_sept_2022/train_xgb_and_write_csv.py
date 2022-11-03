import abc
import functools
import glob
import sys
import pandas as pd
import numpy
import numpy as np
import os
import time
from typing import Optional
import pickle
from collections import namedtuple, defaultdict
import seaborn as sb
import matplotlib.pyplot as plt
from itertools import cycle

from fnmatch import fnmatch
from sklearn import tree
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# TODO: Import xgboost-gpu and use it
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PartialResults = namedtuple(
    'PartialResults',
    'precision recall f1 c_mat X y_true y_pred')

FullResults = namedtuple(
    'FullResults',
    'model test_results train_results total_time')

## Lets try our feature engineering?
# from dibs import feature_engineering # Can't import with python 3.6...... BOOOO
# TODO: etc; Just fix the not supported features in dibs??????


class CrossValidator(object):
    """ TODO: Implement cross validation """
    pass


class Dataset(abc.ABC):
    def __init__(self, input_files, with_cv_indexes):
        # assert isinstance(input_files, list)
        self.data_files = None
        self.init_data(input_files)
        assert self.data_files is not None
        dfs = [pd.read_csv(f, index_col=0) for f in self.data_files]
        self._dfs = dfs
        self.df = pd.concat(dfs)
        if with_cv_indexes:
            self.init_cv_indexes(dfs)

    @abc.abstractmethod
    def init_data(self, input_files):
        raise NotImplemented('You must override init_data')

    def get_dataframe(self):
        return self.df

    def get_dfs(self):
        return self._dfs

    def __str__(self):
        return f'{self.__class__.__name__}:'
        # files_str = '\n\t'.join(self.data_files)
        # paths_str = f'Data files: {files_str}'
        # return f'{self.__class__.__name__}:\n' \
        #        f'Paths: {paths_str}'

    def key(self):
        """ For aggregating experiments and taking averages """
        return frozenset({
            ('data_files', self.data_files),
        })

    def init_cv_indexes(self, dfs):
        # Record CV indexes
        per_video_cv_indexes = []
        start = 0
        for df in dfs:
            end = start + len(df)
            per_video_cv_indexes.append(numpy.array(range(start, end)))
            start = end

        self.cv_indexes = []
        from itertools import chain
        for i in range(len(per_video_cv_indexes)):
            # xgb can not handle iterators, maybe GridSearchCV can??
            train_idxs = list(chain.from_iterable([idxs for m, idxs in enumerate(per_video_cv_indexes) if m != i]))
            test_idxs = per_video_cv_indexes[i]
            self.cv_indexes.append( (train_idxs, test_idxs) )
        # nums = [len(train) + len(test) for (train, test) in self.cv_indexes]
        # import functools
        # num = functools.reduce(lambda x,y: x+y, nums, 0)
        # print(f'AARONT: Number of integers we created...: {num}; Number of cv splits: {len(self.cv_indexes)}')

    def get_cv_indexes(self):
        return self.cv_indexes


class ProtoDataset(Dataset):
    def init_data(self, input_files):
        self.data_files = [input_files[0], input_files[1]]


class FullDataset(Dataset):
    def init_data(self, input_files):
        self.data_files = input_files
        dfs = [pd.read_csv(f, index_col=0) for f in self.data_files]
        self.df = pd.concat(dfs)

class FullDatasetFromSingleDf(Dataset):
    def init_data(self, df):
        self.data_files = 'NA' # Sorry
        self.df = df

class Transformer(object):
    """ Dataset classes will use to prepare X and y inputs,
     then a model will be fit by someone else,
     then the same someone else will use this transformer to transform the outputs"""
    def __init__(self, *, x_extractor, y_extractor, x_pre_processors, y_pre_processors, y_post_processors,
                 y_final_post_processor=None):
        if not isinstance(x_pre_processors, list):
            x_pre_processors = [x_pre_processors]
        if not isinstance(y_pre_processors, list):
            y_pre_processors = [y_pre_processors]
        if not isinstance(y_post_processors, list):
            y_post_processors = [y_post_processors]
        # extractors take the dataframes and extract the data for the rest of the transforms
        self.x_extractor = x_extractor
        self.y_extractor = y_extractor
        # transforms. X is transformed into the model, y requires in and out transformations
        # Applied in order of specification
        self.x_pre_processors = x_pre_processors
        self.y_pre_processors = y_pre_processors
        self.y_post_processors = y_post_processors
        # If training with binary Interaction label, we need to transform the final output to be 6 labels.
        # This will require the RoI data.
        self.y_final_post_processor = y_final_post_processor

    def __str__(self):
        result_str = ''
        if self.x_extractor:
            result_str += '\n\tx_extractor: ' + self.x_extractor.__name__
        if self.y_extractor:
            result_str += '\n\ty_extractor: ' + self.y_extractor.__name__
        if self.x_pre_processors:
            result_str += '\n\tx_pre_processors: ' + ' -> '.join([f.__name__ for f in self.x_pre_processors])
        if self.y_pre_processors:
            result_str += '\n\ty_pre_processors: ' + ' -> '.join([f.__name__ for f in self.y_pre_processors])
        if self.y_post_processors:
            result_str += '\n\ty_post_processors: ' + ' -> '.join([f.__name__ for f in self.y_post_processors])
        return result_str

    def forward(self, df, x_only=False, y_only=False):
        X = self.x_extractor(df)
        assert isinstance(X, (np.ndarray, pd.DataFrame)), f'x_extractor {self.x_extractor.__name__} failed; returned: {X}'
        y = self.y_extractor(df)
        assert isinstance(y, (np.ndarray, pd.DataFrame)), f'y_extractor {self.y_extractor.__name__} failed; returned: {y}'
        for x_pre in self.x_pre_processors:
            X = x_pre(X, df)
            assert isinstance(X, (np.ndarray, pd.DataFrame)), f'x_pre {x_pre.__name__} failed; returned: {X}'
        for y_pre in self.y_pre_processors:
            y = y_pre(y, df)
            assert isinstance(y, (np.ndarray, pd.DataFrame)), f'y_pre {y_pre.__name__} failed; returned: {y}'
        # NOTE FOR LATER: If you want to use RoI data to do separately later
        #                 you will have to change some stuff here
        assert not (x_only and y_only), 'Can only specify one of the only args lol'
        if x_only:
            return X
        elif y_only:
            return y
        else:
            return X, y

    def backward(self, y_pred, y_prob, df, with_final=False):
        """ y_pred and y_prob should be returned as 1-dim numpy arrays that can be used for analysis
        or put directly into the output csv.  df is passed as an argument so that post processing steps
        have any additional information they need"""
        assert isinstance(y_pred, numpy.ndarray), 'y input for transforming needs to be a numpy array'
        assert isinstance(y_prob, numpy.ndarray), 'y input for transforming needs to be a numpy array'
        for y_post in self.y_post_processors:
            y_pred, y_prob = y_post(y_pred, y_prob, df)
        if self.y_final_post_processor and with_final:
            y_pred, y_prob = self.y_final_post_processor(y_pred, y_prob, df)
        return y_pred, y_prob


# TODO: Why is the ModelWrapper separate from the Transformer??
#       Because the ModelWrapper will use the transformer.
#       This makes the ModelWrapper class simpler



class ModelWrapper(object):
    """ Uses a model_type and kwargs to create a template for models.
    Then uses a Transformer to do pre/post processing.
    And all methods accept preloaded dataframes, just plain from disk dataframes."""
    def __init__(self, *, model_type, model_kwargs, transformer: Transformer):
        # NOTE: If you want to do any extra special model specific things like specify constraints
        #       you should do that as an additional wrapper around your base model
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.transformer = transformer
        self._trained_model = None

    def __str__(self):
        return f"""
        Model type: {self.model_type}
        Model kwargs: {self.model_kwargs}
        Transformer: {self.transformer}
        """

    def fit(self, dataset):
        df = dataset.get_dataframe() # For non-CV just get the dataframe
        assert isinstance(df, pd.DataFrame)
        assert self._trained_model is None

        X, y = self.transformer.forward(df)
        model_kwargs = self.model_kwargs.copy()
        self._trained_model = self.model_type(**model_kwargs).fit(X, y) # NOTE: Used numpy.ravel(y) before. Might be a good idea still
        return self # just for interface compatibility

    def predict(self, df):
        """ You will always get back predictions and probabilities. Deal with it. """
        assert self._trained_model, 'Need to call fit on this ModelWrapper first'
        X = self.transformer.forward(df, x_only=True)

        y_pred = self._trained_model.predict(X)
        y_prob = self._trained_model.predict_proba(X)
        # y_pred = self._trained_model.predict(X, ntree_limit=300) # HACK: pass ntree_limit just for xgb with DART...
        # y_prob = self._trained_model.predict_proba(X, ntree_limit=300)

        assert isinstance(y_prob, numpy.ndarray) and y_prob.shape[1] == 2, 'Assuming nd array with 2 output cols, further the second col should be associated with label 1!'
        y_prob = y_prob[:,1] # We just want the second col, with the correct labels!
        y_pred, y_prob = self.transformer.backward(y_pred, y_prob, df)
        return y_pred, y_prob

    def extract_X(self, df):
        """ We need the X data """
        X = self.transformer.forward(df, x_only=True)
        return X

    def extract_y_test(self, df):
        """ We need the y_test data to """
        y = self.transformer.forward(df, y_only=True)
        return y


import torch
import torch.nn as nn
torch.manual_seed(42)


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn_hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
                          batch_first=True) # batch is first dim
        # OR: nn.GRU, nn.LSTM are both options.  No extra params for either
        self.fc = nn.Linear(hidden_size, 1) # TODO: Why (hidden_size, 1)? What exactly do we pass to this layer?
        # self.sigmoid = nn.Sigmoid()

    def init_hidden(self, device): # batch_size=1, so we ignore batching for now
        # 1 frame, and number hidden
        hx = torch.zeros(self.num_layers, self.rnn_hidden_size).to(device)
        # print('hx before init:', hx)
        # torch.nn.init.xavier_normal_(hx)
        # print('hx after init:', hx)
        return hx

    def forward(self, x, hx, c):
        # hx is the final hidden state, we propagate forward manually.  This is slow in it's current implementation.
        # Would be faster to batch up videos, or possibly pass whole video at a time and apply fc to whole thing,
        # but I am not sure how the backprop works in that case (does it work?)
        out, hx = self.rnn(x, hx)
        # print(f'shape of thing: {_all_hidden_state.shape}; {_all_hidden_state}; shape of hidden: {hidden.shape}; hidden: {hidden}')
        # out = hidden[-1,:,:] # As per textbook I am following, we use the final hidden state from the last hidden
        #                      layer as the input to the fully connected layer.  TODO: Y thou????
        print_iter = 2000
        # if c % print_iter == 0:
        #     print('hx:', hx)
        #     print('out1:', out)
        out = self.fc(out)
        # if c % print_iter == 0:
        #     print('out2:', out)
        # TODO: Add layers if working: 1 extra fully connected with paired ReLU
        ## Disabling the sigmoid, using BCE with logits instead
        # out = self.sigmoid(out) # Binary classification task
        # if c % print_iter == 0:
        #     print('out3:', out)
        return out, hx


class RNN_ModelWraper(ModelWrapper):
    def __init__(self, *, holdout_dataset, transformer: Transformer):
        self.holdout_dataset = holdout_dataset
        self.transformer = transformer
        self._trained_model = None

    def __str__(self):
        return f"""
        Transformer: {self.transformer}
        """

    def fit(self, dataset):
        ## AARONT: TEMP: Hard coding the RNN layers here while prototyping
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        if device == 'cpu': raise RuntimeError('Why using CPU???')

        train_dfs = dataset.get_dfs()
        holdout_dfs = self.holdout_dataset.get_dfs()

        # (batches, sequence, input_variables)
        def f(df):
            X,y = self.transformer.forward(df)
            # X = X.values[np.newaxis, ...] # This if you want batches to run in parallel.  We don't
            # y = y[np.newaxis, ..., ]
            X = X.values
            X = torch.from_numpy(X).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            return X,y
        # train_batches =   [ (f(X),f(y)) for X,y in map(self.transformer.forward, train_dfs)]
        # holdout_batches = [ (f(X),f(y)) for X,y in map(self.transformer.forward, holdout_dfs)]
        train_batches =   list(map(f, train_dfs))
        holdout_batches = list(map(f, holdout_dfs))

        def get_batches(_list_of_batches, batch_size=5):
            prev_i = 0
            new_batches = []
            for i in range(batch_size, len(_list_of_batches), batch_size):
                # TODO: have to pad?? Embed??
                new_batch = torch.Tensor(zip(_list_of_batches[prev_i:i])).to(device)
                new_batches.append(new_batch)
                prev_i = i
            return new_batches

        tempX,tempY = train_batches[0]
        print(f'Shape of Xs: {tempX.shape}; shape of yx: {tempY.shape}')
        n_vars = tempX.shape[1]
        rnn_model = MyRNN(input_size=n_vars, hidden_size=8, num_layers=20)
        print('rnn_model:', rnn_model)
        rnn_model.to(device)
        self._trained_model = rnn_model

        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
        # WithLogits combines loss with a sigmoid.  Is more numerically stable!
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4]).to(device))

        def train():
            rnn_model.train()
            total_acc, total_loss = 0,0
            new_train_batches = get_batches(train_batches)
            for i, (X, y) in enumerate(new_train_batches):
                optimizer.zero_grad()
                # TODO: Does X need to be a torch tensor or something??
                # TODO: How run on GPU again???
                # print(f'Type of data in X??: {X}')
                # print(f'Type of data in y??: {y}')
                hx = rnn_model.init_hidden(device)
                batch_acc, batch_loss = 0, 0
                loss = 0
                pred_ys = []
                # print('X.shape:',X.shape)
                for c in range(X.shape[0]):
                    x = X[:,c,:].reshape(-1,1,-1)
                    pred, hx = rnn_model(x, hx, c)
                    pred_ys.append(pred)
                    curr_y = y[c].reshape(1,-1)
                    # print(f'What is pred??? shape: {pred.shape}; pred: {pred}; curr_y.shape: {curr_y.shape}')
                    # if c % 2000 == 0:
                    #     print('curr_y', curr_y)
                    loss += loss_fn(pred, curr_y)
                loss.backward()  # back-prop?
                optimizer.step()
                batch_loss = loss.item() / y.size(0) # This is from text, probably doesn't work on numpy array.
                pred_ys = torch.Tensor(pred_ys).to(device)
                batch_acc += (
                    (pred_ys > 0.0).float() == y
                ).float().sum().item() / y.size(0)
                print(f'Batch {i} training accuracy: {batch_acc:.4f}; training loss: {batch_loss:.4f}')
                total_acc += batch_acc
                total_loss += batch_loss
            return total_acc/len(train_batches), total_loss/len(train_batches)
            # TODO: At end of train step, save progress??  Maybe every X steps if it goes well

        def evaluate():
            rnn_model.eval()
            total_acc, total_loss = 0,0
            with torch.no_grad():
                hx = rnn_model.init_hidden(device)
                batch_acc, batch_loss, loss = 0, 0, 0
                for i, (X, y) in enumerate(holdout_batches):
                    pred_ys = []
                    # print('X.shape:',X.shape)
                    for c in range(X.shape[0]):
                        x = X[c, :].reshape(1, -1)
                        pred, hx = rnn_model(x, hx, c)
                        pred_ys.append(pred)
                        # print(f'What is pred??? shape: {pred.shape}; pred: {pred};')
                        curr_y = y[c].reshape(1, -1)
                        # if c % 2000 == 0:
                        #     print('curr_y', curr_y)
                        loss += loss_fn(pred, curr_y)
                batch_loss = loss.item() / y.size(0)
                print('pred_ys[:50]:', pred_ys[:50])
                pred_ys = torch.Tensor(pred_ys).to(device)
                batch_acc += (
                                     (pred_ys > 0.0).float() == y
                             ).float().sum().item() / y.size(0)
                print(f'Batch {i} training accuracy: {batch_acc:.4f}; training loss: {batch_loss:.4f}')
                total_acc += batch_acc
                total_loss += batch_loss
            return total_acc/len(holdout_batches), total_loss/len(holdout_batches)

        num_epochs = 1000
        models_dir_name = 'TEMP_RNN_CHECKPOINTS'
        os.makedirs(models_dir_name, exist_ok=True)
        old_models = glob.glob(os.path.join(models_dir_name, '*.model'))
        # if old_models:
        #     raise NotImplementedError('AARONT: HERE; Have to load models')
        #     rnn_model.load_state_dict(torch.load(asdf))
        #     rnn_model.eval()
        for epoch in range(num_epochs):
            acc_train, loss_train = train()
            acc_valid, loss_valid = evaluate()
            print(f'Epoch {epoch} train accuracy: {acc_train:.4f}; validation accuracy: {acc_valid:.4f}; train loss: {loss_train:.4f}; validation loss: {loss_valid:.4f}')
            # Save to disk after every epoch
            if epoch % 100 == 0:
                torch.save(rnn_model.state_dict(), os.path.join(models_dir_name, f'rnn_epoch_{epoch}.model'))
            # TODO: Add signal handler to save model as well.

    def predict(self, df):
        raise NotImplementedError('AARONT: Here.')


class GridCvModelWrapper(ModelWrapper):
    def __init__(self, *, model_type, model_static_args, model_kwargs_grid, transformer: Transformer):
        self.model_type = model_type
        self.model_static_args = model_static_args
        self.model_kwargs_grid = model_kwargs_grid
        self.transformer = transformer
        self._trained_model = None

    def __str__(self):
        return f"""
        Model type: {self.model_type}
        Model static args: {self.model_static_args}
        Model kwargs grid: {self.model_kwargs_grid}
        Number of perms in grid: {functools.reduce(lambda acc,l: acc*len(l), self.model_kwargs_grid.values(), 1)}
        Best model: {self._trained_model}
        Transformer: {self.transformer}
        """

    def fit(self, dataset):
        cv_indexes = dataset.get_cv_indexes()
        df = dataset.get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert self._trained_model is None

        print('AARONT: Training model... (remove this line if everything works)')
        X, y = self.transformer.forward(df)

        model_kwargs_grid = self.model_kwargs_grid
        ### HACK: We do manual sample weighting for now, XGB doesn't support
        scale_pos_weights = model_kwargs_grid.get('scale_pos_weight')
        if scale_pos_weights:
            # We could potentially have multiple scale_pos weights and want to compare them, but usually this will just handle a single instance
            new_scale_pos_weights = []
            for scale_pos_weight in scale_pos_weights:
                if isinstance(scale_pos_weight, str) and scale_pos_weight.startswith('AARONT_balanced'):
                    assert isinstance(y, (numpy.ndarray, pd.Series)), f'Expected numpy array or pandas Series, got {y} instead'
                    # HACK: We will match the sklearn interface and calculate weight balancing ourselves...
                    _, modifier = scale_pos_weight.split(':')
                    if modifier == 'no_weight':
                        new_scale_pos_weights.append(1)
                    else:
                        modifier = float(modifier)
                        new_scale_pos_weights.append(((y == 0).sum() / (y >= 1).sum()) * modifier)  # * 0.5 # deweighting a bit
            model_kwargs_grid['scale_pos_weight'] = new_scale_pos_weights
        ### END HACKS...
        clf = GridSearchCV(
            self.model_type(**self.model_static_args),
            self.model_kwargs_grid,
            # scoring='precision',
            # scoring='f1', # binary target f1
            scoring='precision',
            # scoring='neg_log_loss', # uses predict_proba... hmm; Only problem: imbalance!!
            verbose=3,
            cv=cv_indexes,
            # TODO: My machine is cooking itself with anything more than 1 job, need to figure out how to get the fans running correctly in windows 11
            n_jobs=1, # Hard to know what is best when training on the GPU... From task manager looks like maybe 4 jobs at a time can be kept busy
            # 8 jobs locked the whole machine up eventually, somehow the scheduler thought it was a good idea to let them try and share the GPU
            error_score='raise',
            return_train_score=True, # Calculating train score is not required to select the parameters,
                                       # however can be used to get insight into over/under fitting trade off
        )
        self._trained_model = clf.fit(X, y)

        return self # just for interface compatibility

    ## AARONT: Commenting out for now because we should be able to use the generic version for GridCV predict
    # def predict(self, df):
    #     """ You will always get back predictions and probabilities. Deal with it. """
    #     assert self._trained_model, 'Need to call fit on this ModelWrapper first'
    #     X = self.transformer.forward(df, x_only=True)
    #
    #     y_pred = self._trained_model.predict(X)
    #     y_prob = self._trained_model.predict_proba(X)
    #
    #     assert isinstance(y_prob, numpy.ndarray) and y_prob.shape[1] == 2, 'Assuming nd array with 2 output cols, further the second col should be associated with label 1!'
    #     y_prob = y_prob[:,1] # We just want the second col, with the correct labels!
    #     y_pred, y_prob = self.transformer.backward(y_pred, y_prob, df)
    #     return y_pred, y_prob


class XgbCvModelWrapper(ModelWrapper):
    """ Uses a model_type and kwargs to create a template for models.
    Then uses a Transformer to do pre/post processing.
    And all methods accept preloaded dataframes, just plain from disk dataframes."""
    def __init__(self, *, model_type, model_kwargs, transformer: Transformer):
        raise NotImplementedError('No longer works, crashes GPU with memory due to expansion of CV indexes, which is not avoidable')
        # NOTE: If you want to do any extra special model specific things like specify constraints
        #       you should do that as an additional wrapper around your base model
        assert model_type is None, 'HACK: When using the xgb interface it does not make sense to store a model_type'
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.transformer = transformer
        self._trained_model = None

    def __str__(self):
        return f"""
        Model type: {self.model_type}
        Model kwargs: {self.model_kwargs}
        Transformer: {self.transformer}
        """

    def fit(self, dataset):
        cv_indexes = dataset.get_cv_indexes()
        df = dataset.get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert self._trained_model is None

        print('AARONT: Training model... (remove this line if everything works)')
        X, y = self.transformer.forward(df)
        mat_X = xgb.DMatrix(X, label=y)
        model_kwargs = self.model_kwargs.copy()

        def fpreproc(dtrain, dtest, params):
            label = dtrain.get_label()
            ratio = float(np.sum(label == 0)) / np.sum(label == 1)
            params['scale_pos_weight'] = ratio
            return (dtrain, dtest, params)

        (_, _, model_kwargs) = fpreproc(mat_X, None, model_kwargs)

        self._trained_model = xgb.XGBClassifier(**model_kwargs).fit(X, y)  # Fit model for later use?

        xgb.cv(model_kwargs, mat_X, shuffle=False, show_stdv=True,
               fpreproc=fpreproc, folds=cv_indexes, nfold=len(cv_indexes), seed=42
               )
               # callbacks=[xgb.callback.EvaluationMonitor(show_std=True)])
               # stratified = False, # Hmm, undersampling????
               # verbose_eval = False,
               # metrics=, # Monitor more metrics
               # early_stopping_rounds=) # Go faster
        return self # just for interface compatibility

    def predict(self, df):
        """ You will always get back predictions and probabilities. Deal with it. """
        assert self._trained_model, 'Need to call fit on this ModelWrapper first'
        X = self.transformer.forward(df, x_only=True)
        # mat_X = xgb.DMatrix(X)

        y_pred = self._trained_model.predict(X)
        y_prob = self._trained_model.predict_proba(X)

        assert isinstance(y_prob, numpy.ndarray) and y_prob.shape[1] == 2, 'Assuming nd array with 2 output cols, further the second col should be associated with label 1!'
        y_prob = y_prob[:,1] # We just want the second col, with the correct labels!
        y_pred, y_prob = self.transformer.backward(y_pred, y_prob, df)
        return y_pred, y_prob


# y extractors
def build_Y_all_odours(y_features):
    """ Return all the individual labels... This probably isn't useful anymore?? Hard to say """
    def Y_all_odours(df):
        return [df[y_feature] for y_feature in y_features] # was global_y_features
    return Y_all_odours

def build_Y_get_interaction(col='Interaction'):
    """ For the objects datasets """
    def Y_get_interaction(df: pd.DataFrame):
        ys = df[col]
        ys = ys.fillna(value=0.0)
        return ys.values
    return Y_get_interaction


# x extractors
def build_X_only_builtins(non_odour_non_prob_features):
    def X_only_builtins(df):
        return df[non_odour_non_prob_features]
    return X_only_builtins

def build_X_pre_animal_distanc_only(distance_features):
    def X_pre_animal_distance_only(df):
        raise NotImplementedError('AARONT: This is no longer a working implementation')
        return [df[x_feature] for x_feature in distance_features]
    return X_pre_animal_distance_only

def build_X_buildins_and_distance(distance_features):
    def X_builtins_and_distance(df):
        raise NotImplementedError('AARONT: This is also not a working implementation any longer')
        # Probably can just put all the distance features in the thing... should be fine?  IDK
        return [df[non_odour_non_prob_features + [x_feature]] for x_feature in distance_features]
    return X_builtins_and_distance

def build_X_builting_and_distance_combined(distance_features):
    def X_builtins_and_distance_combined(df):
        min_dists = df[distance_features].min(axis=1)
        new_df = df.copy()
        new_df['min_distance_to_any_object'] = min_dists
        return new_df[non_odour_non_prob_features + ['min_distance_to_any_object']]
    return X_builtins_and_distance_combined

def build_X_builtins_and_distance_and_facing(distance_features, facing_features):
    def X_builtins_and_distance_and_facing(df):
        return [df[non_odour_non_prob_features + [x_feature_dist] + [x_feature_facing]]
                for x_feature_dist, x_feature_facing in zip(distance_features, facing_features)]
    return X_builtins_and_distance_and_facing


# Y pre processors
def build_Y_combine_odours(expand_to_original_labels=False):
    def Y_combine_odours(df):
        if expand_to_original_labels:
            # Needed for hack accuracy calc when we use the Y post processor
            odour1 = df['Odour1'] * 1
            odour2 = df['Odour2'] * 2
            odour3 = df['Odour3'] * 3
            odour4 = df['Odour4'] * 4
            odour5 = df['Odour5'] * 5
            odour6 = df['Odour6'] * 6
        else:
            odour1 = df['Odour1'] * 1
            odour2 = df['Odour2'] * 1  # 2
            odour3 = df['Odour3'] * 1  # 3
            odour4 = df['Odour4'] * 1  # 4
            odour5 = df['Odour5'] * 1  # 5
            odour6 = df['Odour6'] * 1  # 6

        assert not any((odour1 > 0) & (odour2 > 0) & (odour3 > 0) & (odour4 > 0) & (odour5 > 0) & (odour6 > 0))

        combined = odour1 + odour2 + odour3 + odour4 + odour5 + odour6
        combined_df = pd.DataFrame(combined, columns=['combined_odours'])
        return [combined_df['combined_odours']]
    return Y_combine_odours

# X pre processors
### We don't have any X pre processors yet, we might want to in the future.
### We would probably want to put feature engineering here kinda, but really
### we only want to do the feature engineering once! So a bit of redesign would
### be in order, but not too much.

def build_feature_selection():
    cols_to_drop = None # Memo
    def feature_selection(X, _df):
        """ NOTE: Time intensive and no benefit for XGB! """
        nonlocal cols_to_drop
        print(f'AARONT: Starting feature selection')
        assert isinstance(X, list) and len(X) == 1
        features = pd.DataFrame(X[0])
        if cols_to_drop is None:
            # First time this was called, better be while fitting!
            cols_to_drop = []
            featureCorrelationMatrix = features.corr().abs()
            col_corr = set()
            print(f'len(features.columns): {len(features.columns)}')
            num_cols_dropped = 0
            for i in range(len(featureCorrelationMatrix.columns)):
                for j in range(i):
                    if (featureCorrelationMatrix.iloc[i, j] >= 0.95) and (
                            featureCorrelationMatrix.columns[j] not in col_corr):
                        colname = featureCorrelationMatrix.columns[i]  # getting the name of column
                        col_corr.add(colname)
                        if colname in features.columns:
                            num_cols_dropped += 1
                            cols_to_drop.append(colname)
                            # del features[colname]  # deleting the column from the dataset
            print(f'AARONT: Dropped {num_cols_dropped} columns')
        return [features.drop(columns=cols_to_drop).values]
    return feature_selection


# y post processors
def build_Y_post_processor_1_to_many_classes(distance_features):
    """ Odour or object distances are passed in """
    def Y_post_processor_1_to_many_classes(y_pred, _y_prob, df):
        """ given y_pred with binary values, assign a class to each based on the closest object """
        nearest_obj = df[distance_features].values.argmin(axis=1) + 1
        return nearest_obj * y_pred  # y_pred is binary
    return Y_post_processor_1_to_many_classes

def build_Y_post_processor_klienberg_filtering():
    def Y_post_processor_klienberg_filtering(y_pred, _y_prob, df):
        # AARONT: TODO: Had 'math domain error downstream here, would have to fix that!  Turning off'
        from simba.Kleinberg_burst_analysis import kleinberg
        # kleinberg filtering setup args etc
        classifierName = 'Interaction'
        logs_path = 'logs_path'
        hierarchy = 1
        assert len(df) == len(y_pred)
        currDf = df[y_pred == 1]
        offsets = list(currDf.index.values)
        # split into offsets by video

        # kleinberg apply algorithm
        print(f'offsets: {offsets}')
        print(f'df cols: {df.columns}')
        # AARONT: TODO: I think the math domain error is due to the offsets calculation, they need to have some spacing
        #               or something like that and are not getting the spacing they need!
        # From the paper: Adjusting 'b' controls inertia that keeps automaton in it's current state (which arg is b?)
        #
        kleinbergBouts = kleinberg(offsets, s=2.0, gamma=0.3) # TODO: Params?
        print(f'AARONT: k-filtering bouts: {kleinbergBouts}')
        kleinbergDf = pd.DataFrame(kleinbergBouts, columns=['Hierarchy', 'Start', 'Stop'])
        kleinbergDf['Stop'] += 1
        file_name = 'Kleinberg_log_' + classifierName + '.csv'
        logs_file_name = os.path.join(logs_path, file_name)
        kleinbergDf.to_csv(logs_file_name)
        kleinbergDf_2 = kleinbergDf[kleinbergDf['Hierarchy'] == hierarchy].reset_index(drop=True)
        df[classifierName] = 0
        for index, row in kleinbergDf_2.iterrows():
            rangeList = list(range(row['Start'], row['Stop']))
            for frame in rangeList:
                df.at[frame, classifierName] = 1
        y_pred = df[classifierName].values
        return y_pred, _y_prob
    return Y_post_processor_klienberg_filtering


def FROM_SIMBA_plug_holes_shortest_bout(y_pred, min_bout_duration): #, fps=None, shortest_bout=None):
    """
    First, find all patterns like `1 0 0 0 ... 0 0 0 1` where the number of frames that are zeros is
    less than or equal to min_bout_duration and fill them with 1's.
    Then find all patterns like `0 1 1 1 ... 1 1 1 0` with the same length specification, and fill those
    with 0's.
    """
    col_name = 'y_pred_col'
    data_df = pd.DataFrame(y_pred, columns=[col_name])
    # frames_to_plug = int(int(fps) * int(shortest_bout) / 1000)
    frames_to_plug_lst = list(range(1, min_bout_duration + 1))
    frames_to_plug_lst.reverse()
    patternListofLists, negPatternListofList = [], []
    for k in frames_to_plug_lst:
        zerosInList, oneInlist = [0] * k, [1] * k
        currList = [1]
        currList.extend(zerosInList)
        currList.extend([1])
        currListNeg = [0]
        currListNeg.extend(oneInlist)
        currListNeg.extend([0])
        patternListofLists.append(currList)
        negPatternListofList.append(currListNeg)
    fill_patterns = np.asarray(patternListofLists)
    remove_patterns = np.asarray(negPatternListofList)

    for currPattern in fill_patterns:
        n_obs = len(currPattern)
        data_df['rolling_match'] = (data_df[col_name].rolling(window=n_obs, min_periods=n_obs)
                                    .apply(lambda x: (x == currPattern).all(), raw=True)
                                    .mask(lambda x: x == 0)
                                    .bfill(limit=n_obs - 1)
                                    .fillna(0)
                                    .astype(bool)
                                    )
        data_df.loc[data_df['rolling_match'] == True, col_name] = 1
        data_df = data_df.drop(['rolling_match'], axis=1)

    for currPattern in remove_patterns:
        n_obs = len(currPattern)
        data_df['rolling_match'] = (data_df[col_name].rolling(window=n_obs, min_periods=n_obs)
                                    .apply(lambda x: (x == currPattern).all(), raw=True)
                                    .mask(lambda x: x == 0)
                                    .bfill(limit=n_obs - 1)
                                    .fillna(0)
                                    .astype(bool)
                                    )
        data_df.loc[data_df['rolling_match'] == True, col_name] = 0
        data_df = data_df.drop(['rolling_match'], axis=1)

    return data_df[col_name]


def build_Y_post_processor_min_bought_duration(min_bout_duration):
    def Y_post_processor_min_bought_duration(y_pred: np.ndarray, _y_prob, _df):
        """ given y_pred a vector of binary predictions, enforce a minimum number of
        concurrent predictions """
        assert numpy.all((y_pred == 1) | (y_pred == 0)), f'ERROR: y_pred must be a binary vector.  Got this instead: {y_pred}'
        assert isinstance(min_bout_duration, int)

        # print(f'y_pred BEFORE min_bought: (sum is {numpy.sum(y_pred)}; {y_pred}')
        y_pred = FROM_SIMBA_plug_holes_shortest_bout(y_pred, min_bout_duration)
        # print(f'y_pred AFTER min_bought: (sum is {numpy.sum(y_pred)}; {y_pred}')
        return y_pred, _y_prob
    return Y_post_processor_min_bought_duration

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
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
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
    """ Aggregate the results of all the experiments and produce a report!"""
    pass  # TODO: Aggregate experiments and produce a report


def dict_or_raise(d):
    assert isinstance(d, dict), f'Expected dict, got: {d}'
    return True


class ExperimentExpander(object):
    # OKAY: The best possible case would be this class hooked up to a database, so we run the exps if we need to,
    #       otherwise we retrieve the results from disk.  Then we can just keep iterating on experiments easily
    #       and keep getting the results back.
    # TODO: Take a model type, set of arg dicts, and set of datasets, and expand the product of the exps
    def __init__(self, *, model_type_to_arg_dicts_dict, datasets_and_transformers):
        self.exps = [
            Experiment(
                train_dataset=train_dataset,
                holdout_dataset=holdout_dataset,
                model_wrapper=ModelWrapper(
                    model_type=model_type, model_kwargs=model_kwargs,
                    transformer=transformer)
            )
            for (train_dataset, holdout_dataset, transformer) in datasets_and_transformers
            for model_type in model_type_to_arg_dicts_dict
            for model_kwargs in model_type_to_arg_dicts_dict[model_type]
            if dict_or_raise(model_kwargs)
        ]
        print(f'Built expander with {len(self.exps)} experiments')

    def run(self):
        for exp in self.exps:
            exp.run()

class Experiment(object):
    def __init__(self, *, train_dataset, holdout_dataset, model_wrapper: ModelWrapper):
        self.train_dataset = train_dataset
        self.holdout_dataset = holdout_dataset
        self.model_wrapper = model_wrapper
        self.started_running = False
        self.finished_running = False
        self.results: Optional[FullResults] = None

    def run(self):
        print(self)
        self.started_running = True
        self.finished_running = False
        # IMPORTANT: There will be multiple models if we are training on each label individually,
        #            usually, right now (Sept 2022) we are training on a single Interaction column.
        #            NOTE: We don't support multi label training right now! But lots of stuff was
        #                  done to enable this in the future
        start = time.time()

        model = self.model_wrapper.fit(self.train_dataset) # Pass the dataset so CV has access to indexes
        tm = model._trained_model
        if isinstance(tm, GridSearchCV):
            print('saving cv_results...  From this average metrics per parameter can be calculated')
            with open('sklearn_grid_search_cv_results.pkl', 'wb') as out_file:
                import pickle
                pickle.dump(tm.cv_results_, out_file, protocol=pickle.HIGHEST_PROTOCOL)

            print(f'Finished training model; best_score: {tm.best_score_}; best_estimator: {tm.best_estimator_}')
            print(f'Best grid search params: {tm.best_params_}')
            print(f'''cv_results:
    mean_test_score: {tm.cv_results_['mean_test_score']}
    std_test_score: {tm.cv_results_['std_test_score']}
    mean_train_score: {tm.cv_results_['mean_train_score']}
    std_train_score: {tm.cv_results_['std_train_score']}
    ''')

        df_test = self.holdout_dataset.get_dataframe()
        df_train = self.train_dataset.get_dataframe()
        y_true_TRAIN = self.model_wrapper.extract_y_test(df_train)
        y_pred_TRAIN, _y_pred_prob_TRAIN = model.predict(df_train)
        X_TRAIN = self.model_wrapper.extract_X(df_train)
        y_true_TEST = self.model_wrapper.extract_y_test(df_test)
        y_pred_TEST, _y_pred_prob_TEST = model.predict(df_test)
        X_TEST = self.model_wrapper.extract_X(df_test)

        end = time.time()
        total_time = end - start

        # average='binary' means to calculate the f1 score of the class with label defaulted to 1
        # average='macro' means to calculate the f1 score of all the classes combined
        # average='micro' isn't useful
        # agerage='weighted' weights by the occurrance
        def create_partial_results(X, y_true, y_pred):
            # try:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            # except:
            #     # must be multi class
            #     precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            c_mat = confusion_matrix(y_true, y_pred)
            # print_importance(model)
            return PartialResults(precision, recall, f1, c_mat, X, y_true, y_pred)

        test_results = create_partial_results(X_TEST, y_true_TEST, y_pred_TEST)
        train_results = create_partial_results(X_TRAIN, y_true_TRAIN, y_pred_TRAIN)
        self.results = FullResults(
            model=model,
            test_results=test_results,
            train_results=train_results,
            total_time=total_time)

        show_train_results = True
        print('\n---- start results ----')
        print('run time:', total_time)
        print('-- test results --')
        print('test precision', test_results.precision)
        print('test recall', test_results.recall)
        print('test f1:', test_results.f1)
        print(test_results.c_mat)
        if show_train_results:
            print('-- train results --')
            print('train precision', train_results.precision)
            print('train recall', train_results.recall)
            print('train f1:', train_results.f1)
            print(train_results.c_mat)
        print('---- end results ----\n')

        self.finished_running = True

    def generate_output_df(self, to_label_input_files, output_path):
        to_label_datasets = [
            (pd.read_csv(f, index_col=0), f)
            for f in to_label_input_files]

        msg = '\n'.join(to_label_input_files)
        print(f'About to label the following data: {msg}')

        for df, orig_file_path in to_label_datasets:
            # print(f'orig_file_path: {orig_file_path}')
            y_pred_test, y_pred_test_proba = self.model_wrapper.predict(df)

            # print(f'WARNING::: y_pred_test_proba BEFORE this mysterious line: {y_pred_test_proba}')
            # # LOL, Had errors so lets skip this
            # y_pred_test_proba = y_pred_test_proba[:, 1]
            # print(f'WARNING::: y_pred_test_proba AFTER this mysterious line: {y_pred_test_proba}')

            base_file_name = os.path.basename(orig_file_path)
            output_file_path = os.path.join(output_path, base_file_name)
            print(f'Currently labelling: {output_file_path}')
            # Create a dataframe from x_tests original plus some extra stuff
            out_df = pd.read_csv(orig_file_path, index_col=0)

            # AARONT: TEMP: Put our interaction column next to the existing one
            if 'Interaction' in out_df.columns:
                print('WARNING:::::::::::::::::: The data to label already has interactions so we are going to add our own column next to it to make analysis easier!!')
                out_df['NEW_Interaction'] = y_pred_test
                out_df['NEW_Probability_Interaction'] = y_pred_test_proba
                #raise RuntimeError('SUGGESTION: You may want to save the existing interactions to make the data easier to analyze')
            else:
                out_df['Interaction'] = y_pred_test
                out_df['Probability_Interaction'] = y_pred_test_proba
            out_df.to_csv(output_file_path)
            # AARONT: TODO: Run kleinburg filtering here

    def generate_classification_report(self, out_path):
        from yellowbrick.classifier import ClassificationReport
        test_results: PartialResults = self.results.test_results
        X = test_results.X
        y_true = test_results.y_true
        clf = self.results.model._trained_model
        print(f'CLF!!!!!!!!!: {clf}')
        ## TODO: Why did he need 2 class names??? class_names = class_names = ['Not_' + classifierName, classifierName]
        clf_name = clf.__class__.__name__
        classes = ['Not_' + clf_name, clf_name]
        visualizer = ClassificationReport(clf, classes=classes, support=True)
        visualizer.score(X=X, y=y_true)
        g = visualizer.poof(outpath=out_path, clear_figure=True)

    def __str__(self):
        if self.results is not None:
            raise RuntimeError('You probably want to have a way to show the results too right??? Right????  YOu can delete this if not!! Just this line will do it, (and the if statement above).  Nothing bad will happen I swear')
        return f'{self.__class__.__name__}: ' \
               f'\n\tTrain Dataset: {self.train_dataset};' \
               f'\n\tTest Dataset: {self.holdout_dataset};' \
               f'\n\tModel Wrapper: {self.model_wrapper};'

    def __repr__(self):
        return self.__str__()

    def save_state(self):
        raise NotImplemented('TODO: pickle with dill; OR: Dump the results into a grid search DB!!!!???? AND pickle with dill!')

    def load_state(self):
        raise NotImplemented('TODO: load with dill')

    # TODO: define and produce the analysis stats afterwards


if __name__ == '__main__':
    args = sys.argv
    # print('Recieved args:', args)
    assert len(args) > 1, 'Need at least 2 arguments.  First is directory with training files, second is directory with holdout files'

    training_input_path = args[1]
    holdout_input_path = args[2]
    if not os.path.isdir(training_input_path):
        raise RuntimeError(
            f'Expected first argument to be an input path pull of csvs to glob!!!'
            fr'Example: C:\Users\toddy\Documents\workspace\HowlandProjects\Final Object\targets_inserted'
            f'Recieved this instead: {training_input_path}')
    # TODO: Check all the training input csvs have the targets inserted!

    if len(args) > 4:
        assert len(args) > 3
        to_label_input_path = args[3]
        if not os.path.isdir(to_label_input_path):
            raise RuntimeError(f'Expected the second argument to be a path to files that do not yet have targets inserted! (Can often be the same as first argument if there is no holdout set!)')

        output_path = args[4]
        if os.path.isdir(output_path):
            raise RuntimeError(f'Expected the third argument to be an output path that does not exist yet! '
                               f'WARNING: The output path alread exists!  I do not want to clober your data!!'
                               f' Put it in an empty dir!!!.  output_path: {output_path}')
        os.makedirs(output_path)
    else:
        to_label_input_path = None
        output_path = None

    global_train_files = glob.glob(os.path.join(training_input_path, '*.csv'))
    global_holdout_files = glob.glob(os.path.join(holdout_input_path, '*.csv'))
    # print(f'global_data_files: {global_data_files}')
    temp_df = pd.read_csv(global_train_files[0], index_col=0)

    ## AARONT: TODO: Odour was removed! Only use the interaction column now!
    # global_y_features = [f'Odour{i}' for i in range(1, 7)]
    # distance_features = [f'{y_feature}_Animal_1_distance' for y_feature in global_y_features]
    # facing_features = [f'{y_feature}_Animal_1_facing' for y_feature in global_y_features]


    odour_features = [
        f for f in temp_df.columns
        if fnmatch(f, '*Odour*')
    ]

    interaction_features = [
        'Interaction',
        'Probability_Interaction'
    ]

    prob_features = [
        f for f in temp_df.columns
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

    filter_p = use_raw_dlc_features = False

    if filter_p:
        raw_dlc_features_only = [s for s in raw_dlc_features_only if '_p' not in s]

    if use_raw_dlc_features:
        non_odour_non_prob_features = raw_dlc_features_only
    else:
        non_odour_non_prob_features = sorted(set(temp_df.columns) - set(
            # odour_features + # Obsolete
            # prob_features + # Don't filter out probability features, use exactly what Simba is using
            interaction_features +
            raw_dlc_features_only
        ))
        msg = '\n'.join(non_odour_non_prob_features)
        # print(f'non_odour_non_prob_features: {msg}')

    if use_raw_dlc_features:
        ######## TODO: Create an extra xgb wrapper that does the constraints!
        #
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
        interaction_constraints['Nose'].extend(interaction_constraints['Tail'])
        print(interaction_constraints)
        interaction_constraints = list(interaction_constraints.values())
        print(interaction_constraints)
    else:
        interaction_constraints = None

    xgb_params_grid_full = dict(
        # TODO: Average results over 'dart' and 'gbtree'.  Dart is 6x slower, so would need a significant improvement to justify
        #          Actually dart is max_depth * 6x slower!!
        scale_pos_weight=[
            'AARONT_balanced:no_weight',
            'AARONT_balanced:1.0',
            # 'AARONT_balanced:1.5', # NOTE: Has to be in grid search part
        ],
        #             for sampling_method in ('gradient_based', 'uniform') # with subsample=0.5, positive results in favour grad
        #             for grow_policy in ('lossguide', 'depthwise') # null exp, might be meaningful to deeper trees
        # TODO: for max_bin in (256, 512, 1024) # better continuous feature binning
        #             for objective in ('binary:logistic', 'binary:hinge')
        #             for eval_metric in ('logloss', 'error', 'error@0.2', 'error@0.8') # totally null exp
        # max_delta_step is too important, make sure you test it against 0!
        # max_delta_step - The maximum step size that a leaf node can take.
        #     In practice, this means that leaf values can be no larger than max_delta_step * eta
        ##for max_delta_step in (0, 1)  # max_delta_step has a large impact.  Why? What is it?
        #             for eta in (0.1, 0.2, 0.3)
        # gamme is used during prunning.  min_split_loss
        ### for gamma in (0.0, 100.,)  # , 1000.,)
        #             for subsample in (1, 0.5, 0.1)
        #             for num_parallel_tree in (1, 10) # it is 39 seconds per model with 100 trees...
        max_depth=[6,10],  # default is 6; Experiments showed 2 or 3 worked best
        max_delta_step=[0, 1],  # Was set to 1; VERY IMPORTANT PARAMETER! Read description in comments elsewhere
        gamma=[0.0, 10.],
        base_score=[0.5, 0.9, 0.1], # Has some impact.  Not sure if this applies to the positive class or 0 class. (as in, is 0.9 the probability of 0 class or 1 class?)
        # # RF was a noop, I didn't use column sub sampling though.  Not necessary I think
        #        for col_split_by_node in (0.75, )
        # TODO: min_child_weight
        #    For classification this gives the required sum of p*(1-p), where p is the probability
        #        THE SUM of p*(1-p), which is 0.25 max.  You will need at least 4 * min_child_weight rows
        #        to keep the split.
        # TODO: Best model had min child weight 2, could be the good regularizer
        # min_child_weight=[0, 2, 4, 6, 8],
        min_child_weight=[0, 4],
        ## for min_child_weight in (0, 2)  # inconclusive experiments, just sanity checking
        #             for base_score in (0.5, 0.1) # default 0.5, leave this alone because we are already weighting.
        # TODO: EARLY STOPPING BASED ON A HOLDOUT SET!!!
        # https://xgboost.readthedocs.io/en/latest/python/python_intro.html#early-stopping
        # This works with both metrics to minimize (RMSE, log loss, etc.) and to maximize (MAP, NDCG, AUC).
        # Note that if you specify more than one evaluation metric the last one in param['eval_metric']
        # is used for early stopping. (saw someone do this with 'ams@0.15' in a competition)
        # base_score=['match_scale_pos_weight'], # AARONT: TODO: This is a global bias for prediction!!! Could be inverse of scale_pos_weight???
    )
    xgb_model_static_args = dict(
        n_estimators=[300],  # The number of boosting rounds; default 100?
        learning_rate=[0.3],
        booster=['gbtree', 'dart'],  # TODO: Try dart, note: Using predict() with DART booster
        # If the booster object is DART type, predict() will perform dropouts, i.e. only some of the trees will be evaluated.
        # This will produce incorrect results if data is not the training data. To obtain correct results on test sets,
        # set iteration_range to a nonzero value, e.g.
        ################# ACTUALLY: from the repl:  set ntree_limit=num_round.....
        #             preds = bst.predict(dtest, iteration_range=(0, num_round))

        ######## Below is true static
        # early_stopping_rounds=[3, 20, 30],  # TODO: Tune early stopping?? DOESN'T HAVE AN IMPACT
        n_jobs=[14], # Wait... should this still be 14 jobs with GPU?? Doesn't make a difference.  Can't see any jobs in the task manager
        # tree_method='exact', # More time but enumarates all possible splits
        #                 base_score=base_score,
        #                 scale_pos_weight='AARONT_balanced',
        # eta=0.3, # default 0.3; learning rate NOTE: sklearn interface uses: learning_rate instead!
        # gamma=100,  # default 0.0; min_split_loss (minimum loss reduction to create a split)
        # use_label_encoder=False,
        objective=['binary:logistic'],
        #                 eval_metric=eval_metric,
        #                 objective=objective,
        #                 subsample=subsample,
        # subsample=[0.1],  # sample of the rows to use, sampled once every boosting iteration
        subsample=[0.5],  # sample of the rows to use, sampled once every boosting iteration
        tree_method=['gpu_hist'],
        # sampling_method=['gradient_based'],  # allows very small subsample, as low as 0.1
        grow_policy=['lossguide'],  # TODO: What are the grow policies? Why this option?
        # num_parallel_tree=[10],
        # colsample_bytree=[0.75],
        # colsample_bylevel=[0.75],
        # colsample_bynode=[0.75],
        # asdf='not a param', # xgb does not raise an error for an unrecognized parameter...
        interaction_constraints=[interaction_constraints], # AARONT: Commenting out for moment
    )
    xgb_model_static_args = {k:v[0] for k,v in xgb_model_static_args.items()}
    print(xgb_model_static_args)

    dataset_type = FullDataset
    # dataset_type = ProtoDataset

    split_by_video = True
    with_stratification = False
    if split_by_video:
        train_dataset = dataset_type(global_train_files, with_cv_indexes=True)
        holdout_dataset = dataset_type(global_holdout_files, with_cv_indexes=False)
        ## For NN don't need cv indexes
        # train_dataset = dataset_type(global_train_files, with_cv_indexes=False)
        # holdout_dataset = dataset_type(global_holdout_files, with_cv_indexes=False)
    else:
        temp_dataset = dataset_type(global_train_files, with_cv_indexes=False)
        temp_df = temp_dataset.get_dataframe()
        from sklearn.model_selection import train_test_split
        temp_df_train, temp_df_test = train_test_split(
            temp_df, test_size=0.20, random_state=42, # shuffle=False,
            stratify=temp_df['Interaction'].values if with_stratification else None)
        train_dataset = FullDatasetFromSingleDf(temp_df_train, with_cv_indexes=False)
        holdout_dataset = FullDatasetFromSingleDf(temp_df_test, with_cv_indexes=False)
        # by frame

    # def __init__(self, *, x_extractor, y_extractor, x_pre_processors, y_pre_processors, y_post_processors,
    #              y_final_post_processor=None):
    transformer = Transformer(
        x_extractor=build_X_only_builtins(non_odour_non_prob_features),
        y_extractor=build_Y_get_interaction(),
        x_pre_processors=[
        # build_under_sampling() # AARONT: TODO: Add under sampling here?  Would be useful if we build per label models.
            # build_feature_selection()
        ],
        y_pre_processors=[], # Example: for odours we have to combine 6 labels into 1
        y_post_processors=[
            # TODO: Turn on min bought duration
            # build_Y_post_processor_min_bought_duration(min_bout_duration=10),
            # AARONT: TODO: Had 'math domain error downstream here, would have to fix that!  Turning off'
            # build_Y_post_processor_klienberg_filtering()
        ],
        y_final_post_processor=None
    )


    exp = Experiment(
        # AARONT: TODO: Implement so that score is calculated for each step of the y post processing transforms.
        #               Also: Do we still need the y_final_processor thing??? That we might handle differently.
        train_dataset=train_dataset,
        holdout_dataset=holdout_dataset,
        model_wrapper=GridCvModelWrapper(
            transformer=transformer,
            model_type=xgb.XGBClassifier,
            model_static_args=xgb_model_static_args,
            model_kwargs_grid=xgb_params_grid_full,
        )
    )
    exp.run()

    # RNN_exp = Experiment(
    #     train_dataset=train_dataset,
    #     holdout_dataset=holdout_dataset,
    #     model_wrapper=RNN_ModelWraper(
    #         holdout_dataset=holdout_dataset, # AARONT: TEMP: Hack to get holdout dataset to NN for validation while training
    #         transformer=transformer,
    #     )
    # )
    # RNN_exp.run()


    # TODO: If we are going to handle the different labels separately, then build one full pipeline and full optimization
    #       for each.  Build a final class to aggregate the models.  This allows each model to optimize it's own parameters.
    ## TODO: Make my own visualization, don't want to do CV again
    # exp.generate_classification_report('XGB_classification_report.pdf')

    if to_label_input_path and output_path:
        to_label_files = glob.glob(os.path.join(to_label_input_path, '*.csv'))
        output_df = exp.generate_output_df(to_label_files, output_path)
        raise RuntimeError('CONGRATS!! You made it to the end of the script and MIGHT want to start labeling the output!!')

    # Turn this on if you want to compare against the method used by Simba
    exp = Experiment(
        train_dataset=train_dataset,
        holdout_dataset=holdout_dataset,
        # RandomForestClassifier,
        model_wrapper=ModelWrapper(
            transformer=transformer,
            # model_type=DecisionTreeClassifier,
            model_type=RandomForestClassifier,
            model_kwargs=dict(
                # n_estimators=1000,
                n_estimators=200,
                bootstrap=True,
                verbose=1,
                n_jobs=-1,
                criterion='gini',  # Gini is standard, shouldn't be a huge factor
                min_samples_leaf=50,
                max_features='sqrt',
                # max_depth=15,  # LIMIT MAX DEPTH!!  Runtime AND generalization error should improve drastically
                random_state=42,
                #     ccp_alpha=0.005, # NEW PARAMETER, I NEED TO DEFINE MY EXPERIMENT SETUPS BETTER, AND STORE SOME RESULTS!!
                # Probably need to whip up a database again, that's the only way I have been able to navigate this in the past
                # Alternatively I could very carefully define my experiments, and then run them all in a batch and create a
                # meaningful report.  This is probably the best way to proceed.  It will lead to the most robust iteration
                # and progress.
                # class_weight='balanced',  # balance weights at nodes based on class frequencies
            )
        )
    )
    exp.run()
    exp.generate_classification_report('DTREE_classification_report.pdf')





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
    #         DecisionTreeClassifier: [
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


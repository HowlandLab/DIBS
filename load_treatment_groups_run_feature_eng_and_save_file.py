import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dibs import config
from dibs import io

from dibs.pipeline_pieces import NeoHowlandFeatureEngineering

def create_flat_df():
    # 3. Save flat csv of everything
    global flat_df
    flat_df = io.load_data_based_on_directory_spec_and_engineer_features(fe=NeoHowlandFeatureEngineering())
    print('Finished creating flat_df')
    flat_df.to_csv(os.path.join(config.DEFAULT_TRAIN_DATA_DIR, 'flat_df.csv'))

def load_flat_df():
    return pd.read_csv(os.path.join(config.DEFAULT_TRAIN_DATA_DIR, 'flat_df.csv'))

def logistic_reg_on_videos():
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    # TODO: Training on rows directly doesn't make sense.
    #       We need to do something smarter. Either a time dependent model,
    #       or train on some aggregated/expanded representation of rows (example with summary
    #       statistics etc)
    model = LogisticRegressionCV(
        random_state=0,
        solver='liblinear', # default is lbfgs
        max_iter=200, # default 100
    )

    X = flat_df[eng_features]
    y = flat_df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rep = classification_report(y_true=y_test, y_pred=y_pred)
    print(model, rep)

    print('Features and coefs:')
    for feat, coef in zip(eng_features, model.coef_[0]):
        print(feat, coef)

    import shelve
    with shelve.open('model_etc.shelve') as db:
        db['model'] = model
        db['X_train'] = X_train
        db['X_test'] = X_test
        db['y_train'] = y_train
        db['y_test'] = y_test
    return model, rep


fe = NeoHowlandFeatureEngineering()
eng_features = fe.all_engineered_features
y_col = 'treatment_group'
# flat_df = None

if __name__ == '__main__':
    create_flat_df()
    # print('loading flat_df'); flat_df = load_flat_df()
    # print('doing logistic regression'); model, rep = logistic_reg_on_videos()
    # print(model, rep)








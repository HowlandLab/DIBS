import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dibs import config
from dibs import io

from dibs.pipeline_pieces import NeoHowlandFeatureEngineering

class DataForOneVideo(object):
    csvs_path = config.DEFAULT_TRAIN_DATA_DIR
    def __init__(self, *, animal, dlc_csv, roi_csv):
        self.animal = animal
        self.dlc_csv = dlc_csv
        self.roi_csv = roi_csv

class TreatmentGroupsData(object):
    def __init__(self, *, group):
        self.group = group
        self.datas = []

    def add_data(self, data: DataForOneVideo):
        self.datas.append(data)


# 1. Load data
def eng_data():
    training_data_dir = DataForOneVideo.csvs_path

    for inode in os.listdir(training_data_dir):
        if not os.path.isdir(os.path.join(training_data_dir, inode)):
            continue
        # is a dir
        group = TreatmentGroupsData(inode) # is a treatment group

        #

        for file in os.listdir(os.path.join(training_data_dir, group)):
            if not file.endswith('.csv'):
                continue
            # is a csv
            animal = file[0:file.find('_')]
            if 'DLC' in file:
                dlc_csv

    names_to_paths = {
        file: os.path.join(training_data_dir, file) for file in os.listdir(training_data_dir)
    }
    dirs = {
        name: path for name, path in names_to_paths.items() if os.path.isdir(path)
    }
    print(dirs)
    assert dirs

    # Only look at dirs with csv files inside of them
    treatment_groups = {
        name for name,path in names_to_paths.items()
        if any(file.endswith('.csv') for file in os.listdir(path))
    }
    print(treatment_groups)
    assert treatment_groups

    def get_csvs(path):
        return [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

    group_to_csvs = {
        group: get_csvs(dirs[group])
        for group in treatment_groups
    }

    print(group_to_csvs)
    assert group_to_csvs

    # 2. Run feature eng
    dlc_only_dfs = []
    for group, csvs in group_to_csvs.items():
        # split on _ and take first thing to get animal number
        dlc_csvs = {file.split('_')[0]: file for file in csvs if 'DLC' in file}
        print(dlc_csvs)
        simba_roi_csvs = {file for file in csvs if 'DLC' not in file}
        for csv in csvs:
            print(csv)
            df = io.read_csv(csv)
            eng_df = fe.engineer_features(df)
            eng_df['treatment_group'] = group
            dlc_only_dfs.append(eng_df)

    # TODO: Load which are we are in classification and attach to dfs

    # 3. Save flat csv of everything
    global flat_df
    flat_df = pd.concat(dlc_only_dfs)
    flat_df.to_csv('flat_df.csv')

def load_flat_df():
    return pd.read_csv('flat_df.csv')

def logistic_reg_on_videos():
    from sklearn.linear_model import LogisticRegressionCV
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

    model.fit(X, y)
    # rep = classification_report() # TODO: Need test split I guess..??
    print(model)


fe = NeoHowlandFeatureEngineering()
eng_features = fe.all_engineered_features
y_col = 'treatment_group'
# flat_df = None

if __name__ == '__main__':
    eng_data()
    print('loading flat_df'); flat_df = load_flat_df()
    # print('doing logistic regression'); logistic_reg_on_videos()








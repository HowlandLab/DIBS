import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dibs import config
from dibs import io

from dibs.pipeline_pieces import NeoHowlandFeatureEngineering

class DataForOneVideo(object):
    root_path = config.DEFAULT_TRAIN_DATA_DIR
    dlc_csvs_dir = 'csv_filtered'
    simba_roi_dir = 'csv_descriptive'
    videos_dir = 'videos'
    def __init__(self, *, animal, dlc_df, roi_df, video_path):
        self.animal = animal
        self.dlc_df = dlc_df
        self.roi_df = roi_df
        self.video_path = video_path

class TreatmentGroupsData(object):
    def __init__(self, *, group):
        self.name = group
        self.datas: list[DataForOneVideo] = []

    def add_data(self, data: DataForOneVideo):
        self.datas.append(data)


from contextlib import contextmanager
from pathlib import Path

@contextmanager
def set_directory(path, cd_back=True):
    """Sets the cwd within the context"""

    try:
        assert (not cd_back or '/' not in path), 'Only expect to go up/down 1 dir at a time'
        os.chdir(path)
        yield
    finally:
        if cd_back:
            os.chdir('..')

# 1. Load data
def eng_data():
    with set_directory(DataForOneVideo.root_path, cd_back=False):
        groups = [] # list of treatment groups data things

        for inode in os.listdir():
            if not os.path.isdir(inode):
                continue
            # is a dir
            group = TreatmentGroupsData(group=inode) # is a treatment group
            with set_directory(inode):
                dlc_csvs = os.listdir(DataForOneVideo.dlc_csvs_dir)
                roi_csvs = os.listdir(DataForOneVideo.simba_roi_dir)
                video_paths = os.listdir(DataForOneVideo.videos_dir)

                # get list of animals based on dlc data, which must exist.  Everything else can be missing
                for file in dlc_csvs:
                    animal = file.split('_')[0]
                    print(f'Reading data for animal {animal}')
                    dlc_df = io.read_dlc_csv(os.path.join(DataForOneVideo.dlc_csvs_dir, file))
                    try:
                        roi_csv_path = next(file for file in roi_csvs if file.startswith(animal))
                        roi_df = pd.read_csv(os.path.join(DataForOneVideo.simba_roi_dir, roi_csv_path))
                    except StopIteration:
                        roi_df = None
                    try:
                        video_path = os.path.join(DataForOneVideo.videos_dir, next(file for file in video_paths if file.startswith(animal)))
                    except StopIteration:
                        video_path = None
                    group.add_data(
                        DataForOneVideo(
                            animal=animal,
                            dlc_df=dlc_df,
                            roi_df=roi_df,
                            video_path=video_path
                        )
                    )

                # Have list of groups now
            groups.append(group)

        def extract_simba_roi(roi_df, n_frames):
            print(f'roi_df: {roi_df}; \nn_frames: {n_frames}')
            cols = list(roi_df.columns)
            roi_index = cols.index('Shape_name')
            entry_frame_index = cols.index('Entry_frame')
            exit_frame_index = cols.index('Exit_frame')

            vals = np.array([None for _ in range(n_frames)], dtype=object)
            for row in roi_df.values:
                row = list(row)
                simba_roi = row[roi_index]
                enter, exit = row[entry_frame_index], row[exit_frame_index]
                # print(f'Filling {enter} to {exit} with {simba_roi}')
                vals[enter: exit] = simba_roi
            # concat the ROI stuffs, just join them all and forward fill
            df = pd.DataFrame(dict(simba_roi=vals))
            df.ffill(inplace=True)
            return df['simba_roi'].values

        # 2. Run feature eng
        full_dfs = []
        for group in groups:
            for data in group.datas:
                print(f'Building engineered and aggregated dataframe for animal {data.animal}')
                # 1. Put the simba ROI data back onto the DLC data frame
                df = data.dlc_df
                animal_string = data.animal
                assert isinstance(animal_string, str)
                df['animal'] = animal_string
                if data.roi_df is None:
                    df['simba_roi'] = None
                else:
                    df['simba_roi'] = extract_simba_roi(data.roi_df, len(data.dlc_df))

                eng_df = fe.engineer_features(df)
                assert 'simba_roi' in eng_df.columns
                group_string = group.name
                assert isinstance(group_string, str)
                eng_df['treatment_group'] = group_string
                full_dfs.append(eng_df)

    # TODO: Load which are we are in classification and attach to dfs

    # 3. Save flat csv of everything
    global flat_df
    flat_df = pd.concat(full_dfs)
    flat_df.to_csv('flat_df.csv')

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
    eng_data()
    print('loading flat_df'); flat_df = load_flat_df()
    print('doing logistic regression'); model, rep = logistic_reg_on_videos()
    print(model, rep)








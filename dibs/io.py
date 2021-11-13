"""
Functions related to opening/saving files should go here

"""

from contextlib import contextmanager
from typing import List
import joblib
import inspect
import numpy as np
import os
import dill
import pandas as pd
import re
import sys

from dibs import check_arg, config, logging_enhanced
from dibs.logging_enhanced import get_current_function
from dibs.feature_engineering import filter_dlc_output
from dibs.pipeline_pieces import FeatureEngineerer

logger = config.initialize_logger(__name__)


########################################################################################################################


# TODO: med/high: change function to accept either CSV or h5 files from DLC. Functionally, should be the same except for
#  deciding to use read_h5() or read_csv()
def read_dlc_csv(csv_file_path: str, do_our_filtering=True, **kwargs) -> pd.DataFrame:
    """
    Reads in a CSV that is assumed to be an output of DLC. The raw CSV is re-formatted to be more
    friendly towards data manipulation later in the B-SOiD feature engineering pipeline.

        * NO MATH IS DONE HERE & NO DATA IS REMOVED *

    Parameters:
    :param csv_file_path: (str, absolute path) The input file path requires the CSV file in question to be
        an output of the DLC process. If the file is not, use pd.read_csv() instead.

    EXAMPLE data: DataFrame directly after invoking pd.read_csv(csv, header=None):
                   0                                              1                                               2                                               3                                                4  ...
        0     scorer  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  ...
        1  bodyparts                                      Snout/Head                                      Snout/Head                                      Snout/Head                               Forepaw/Shoulder1  ...
        2     coords                                               x                                               y                                      likelihood                                               x  ...
        3          0                                 1013.7373046875                                661.953857421875                                             1.0                              1020.1138305664062  ...
        4          1                              1012.7627563476562                               660.2426147460938                                             1.0                              1020.0912475585938  ...

    :return: (DataFrame)
        EXAMPLE OUTPUT:
                 Snout/Head_x       Snout/Head_y Snout/Head_likelihood Forepaw/Shoulder1_x Forepaw/Shoulder1_y Forepaw/Shoulder1_likelihood  ...                                          scorer
        0     1013.7373046875   661.953857421875                   1.0  1020.1138305664062   621.7146606445312           0.9999985694885254  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
        1  1012.7627563476562  660.2426147460938                   1.0  1020.0912475585938   622.9310913085938           0.9999995231628418  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
        2  1012.5982666015625   660.308349609375                   1.0  1020.1837768554688   623.5087280273438           0.9999994039535522  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
        3  1013.2752685546875  661.3504028320312                   1.0     1020.6982421875   624.2875366210938           0.9999998807907104  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
        4  1013.4093017578125  661.3643188476562                   1.0  1020.6074829101562     624.48486328125           0.9999998807907104  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
    """
    # Arg checking
    if not os.path.isfile(csv_file_path):
        err = f'Input filepath to csv was not a valid file path: {csv_file_path} (type = {type(csv_file_path)}.'
        logger.error(err)
        raise FileNotFoundError(err)
    # Read in kwargs
    nrows = kwargs.get('nrows', sys.maxsize)  # TODO: address case where nrows is <= 3 (no data parsed then)
    # file_path = csv_file_path  # os.path.split(csv_file_path)[-1]
    file_folder, file_name = os.path.split(csv_file_path)
    # file_name_without_extension, extension = file_name.split('.')  # Old way of doing things. remove this line later.
    ext_common_idx = file_name.rfind('.')
    file_name_without_extension, extension = file_name[:ext_common_idx], file_name[ext_common_idx + 1:]
    assert file_name_without_extension == config.get_data_source_from_file_path(
        csv_file_path)  # TODO: low: delete this line only after a test has been implemented

    # # # # # # #
    # Read in CSV
    df = pd.read_csv(csv_file_path, header=None, nrows=nrows)
    # # Manipulate Frame based on top row
    # Check that the top row is like [scorer, DLCModel, DLCModel.1, DLCModel.2, ...] OR [scorer, DLCModel, DLCModel,...]
    # Use regex to truncate the decimal/number suffix if present.
    top_row_values_set: set = set([re.sub(r'(.*)(\.\w+)?', r'\1', x) for x in df.iloc[0]])
    top_row_without_scorer: tuple = tuple(top_row_values_set - {'scorer', })
    if len(top_row_without_scorer) != 1:
        non_standard_dlc_top_row_err = f'The top row of this DLC file ({csv_file_path}) is not standard. ' \
                                       f'Top row values set = {top_row_values_set}. / ' \
                                       f'DataFrame = {df.head().to_string()}'
        logger.error(non_standard_dlc_top_row_err)
        raise ValueError(non_standard_dlc_top_row_err)
    # Save scorer/model name for later column creation
    dlc_scorer = top_row_without_scorer[0]
    # Remove top row (e.g.: [scorer, DLCModel, DLCModel, ...]) now that we have saved the model name
    df = df.iloc[1:, :]

    # # Manipulate Frame based on next two rows to create column names.
    # Create columns based on next two rows. Combine the tow rows of each column and separate with "_"
    array_of_next_two_rows = np.array(df.iloc[:2, :])
    new_column_names: List[str] = []
    for col_idx in range(array_of_next_two_rows.shape[1]):
        new_col = f'{array_of_next_two_rows[0, col_idx]}_{array_of_next_two_rows[1, col_idx]}'
        new_column_names += [new_col, ]
    df.columns = new_column_names

    # TODO: Revert to supporting the same data model as dlc, with the multi index.  It was very nice.
    # Remove next two rows (just column names, no data here) now that columns names are instantiated
    df = df.iloc[2:, :]

    # # Final touches
    # Delete "coords" column since it is just a numerical counting of rows. Not useful data.
    df = df.drop('bodyparts_coords', axis=1)
    # Convert all values to float in case they are parsed as string
    df = df.astype(np.float)
    # Reset index (good practice) after chopping off top 3 columns so index starts at 0 again
    df = df.reset_index(drop=True)
    # Instantiate 'scorer' column so we can track the model if needed later
    df['scorer'] = dlc_scorer
    # File source __________
    # df['file_source'] = csv_file_path
    # Save data file name (different from pathing source)
    df['data_source'] = file_name_without_extension
    # Number the frames
    df['frame'] = list(range(len(df)))
    if not do_our_filtering:
        logger.info('reading in DLC csv WITHOUT applying our filtering to it')
        return df
    df_filtered = filter_dlc_output(df)
    return df_filtered


class DataForOneVideo(object):
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


def extract_simba_roi(roi_df, n_frames):
    """
    From a pre-loaded simba_roi dataframe, and a number of frames
    (REALLY, number of rows in the associated dataframe........ MIGHT have to validate this!!!!!)
    expand the data that is in (roi_name, entry_frame, exit_frame) format, take that and expand it to n_frames
    values all in roi_name, one for each frame, in order (sorted, like the input df, by temporalness)
    """
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
    df.ffill(inplace=True) # forward fill the None/NaN values or whatever
    return df['simba_roi'].values


def load_data_based_on_directory_spec_and_engineer_features(fe: FeatureEngineerer):
    initial_dir = os.getcwd()
    with set_directory(config.DEFAULT_TRAIN_DATA_DIR, cd_back=False):
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
                    dlc_df = read_dlc_csv(os.path.join(DataForOneVideo.dlc_csvs_dir, file))
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
    os.chdir(initial_dir)
    return pd.concat(full_dfs)


def read_h5(data_file_path, **kwargs) -> pd.DataFrame:
    # TODO: HIGH IMPLEMENT! :)
    raise NotImplementedError(f'')
    return


# Saving and stuff
def save_to_folder(p, output_path_dir=config.OUTPUT_PATH, read_and_return=False, df_export=None, stage=''):
    """
    Defaults to config.ini OUTPUT_PATH variable if a save path not specified beforehand.
    :param output_path_dir: (str) an absolute path to a DIRECTORY where the pipeline will be saved.
    """

    # Check if valid directory
    check_arg.ensure_is_dir(output_path_dir)

    # Execute
    final_out_path = os.path.join(output_path_dir, generate_pipeline_filename(p._name))
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
            # joblib.dump(self, model_file)
            # pickle.dump(self, model_file)
            # TODO: Write data frames to disc
            # dill.detect.trace(True) # Debugging trace.
            # IMPORTANT: protocol=dill.HIGHEST_PROTOCOL is necessary.  If this argument is not provided,
            #            the oldest compatible protocol will be used, and that will be inefficient as well
            #            as not supporting some required features.
            #            recurse=True may not be required, but improves performance.
            dill.dump(p, model_file, protocol=dill.HIGHEST_PROTOCOL, recurse=True)
    except Exception as e:
        err = f'{get_current_function()}(): An unexpected error occurred: {repr(e)}'
        logger.error(err)
        raise e

    logger.debug(f'{inspect.stack()[0][3]}(): Pipeline ({p.name}) saved to: {final_out_path}')
    if df_export is not None:
        save_csv(os.path.join(output_path_dir, stage+'.csv'), df_export)
    if read_and_return:
        return read_pipeline(final_out_path)
    else:
        return p


def save_csv(path, df: pd.DataFrame):
    # Check if valid final path to be saved
    check_arg.ensure_is_valid_path(path)
    if not check_arg.is_pathname_valid(path):
        invalid_path_err = f'Invalid output path save: {path}'
        logger.error(invalid_path_err)
        raise ValueError(invalid_path_err)

    logger.debug(f'{inspect.stack()[0][3]}(): Attempting to save file as csv: {path}.')

    df.to_csv(path, index=False)


def read_pipeline(path_to_file: str):
    """
    With a valid path, read in an existing pipeline
    :param path_to_file:
    :return:
    """
    check_arg.ensure_is_file(path_to_file)
    logger.debug(f'{logging_enhanced.get_current_function()}(): Trying to open: {path_to_file}')
    with open(path_to_file, 'rb') as file:
        # p = joblib.load(file)
        # p = pickle.load(file)
        # TODO: read data frames from disk
        import dill  # Dill may not be strictly required, but allows for much more robust serialization.
        # If dill fails in the future, a fallback to pickle should be okay, but any lambdas
        # (amoung other exotic python constructs) will not be serializable without dill.
        p = dill.load(file)

    # # HACKS: TODO: After we implement loading and saving dataframes this won't be necessary anymore
    ## DOESN"T WORK!! Have to do this in streamlit direct.....
    # p.force_reengineer_train_features=True
    # p.reengineer_predict_features=True
    # p._embedder_is_built = False
    # p._clusterer_is_built = False
    # p._clf_is_built = False
    logger.debug(f'{logging_enhanced.get_current_function()}(): Pipeline at {path_to_file} opened successfully!')
    return p


def generate_pipeline_filename(name: str):
    """
    Generates a pipeline file name given its name.

    This is an effort to standardize naming for saving pipelines.
    """
    file_name = f'{name}.pipeline'
    return file_name


def clean_string(s):
    if isinstance(s, str):
        return s.strip().strip('"').strip("'")
    else:
        return s

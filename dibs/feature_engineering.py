"""
All data engineering goes here.

For the functions that have a prefix of "attach_", these functions attach new columns to the input DataFrame.

Potential abbreviations:
    sn: snout
    pt: proximal tail ?

DELETE THIS STRING
7 Features listed in paper (terms in brackets are cursive and were written in math format. See paper page 12/13):
    1. body length (or "[d_ST]"): distance from snout to base of tail
    2. [d_SF]: distance of front paws to base of tail relative to body length (formally: [d_SF] = [d_ST] - [d_FT],
        where [d_FT] is the distance between front paws and base of tail
    3. [d_SB]: distance of back paws to base of tail relative to body length (formally: [d_SB] = [d_ST] - [d_BT]
    4. Inter-forepaw distance (or "[d_FP]"): the distance between the two front paws

    5. snout speed (or "[v_s]"): the displacement of the snout location over a period of 16ms
    6. base-of-tail speed (or ["v_T"]): the displacement of the base of the tail over a period of 16ms
    7. snout to base-of-tail change in angle:

Author also specifies that: the features are also smoothed over, or averaged across,
    a sliding window of size equivalent to 60ms (30ms prior to and after the frame of interest).
"""
from tqdm import tqdm
from typing import List, Tuple
import inspect
import itertools
import math
import numpy as np
import pandas as pd

from dibs import check_arg, config, statistics, logging_enhanced


logger = config.initialize_logger(__name__)


### Attach features as columns to a DataFrame of DLC data

def attach_average_bodypart_xy(df: pd.DataFrame, bodypart_1: str, bodypart_2: str, output_bodypart: str, resolve_bodyparts_with_config_ini=False, copy=False) -> pd.DataFrame:
    """
    Returns 2-d array where the average location of the hindpaws are

    """
    bodypart_1 = config.get_part(bodypart_1) if resolve_bodyparts_with_config_ini else bodypart_1
    bodypart_2 = config.get_part(bodypart_2) if resolve_bodyparts_with_config_ini else bodypart_2

    # Arg checking
    check_arg.ensure_type(df, pd.DataFrame)
    for body_part, xy in itertools.product((bodypart_1, bodypart_2), ('x', 'y')):
        feat_xy = f'{body_part}_{xy}'
        if feat_xy not in set(df.columns):
            err_missing_feature = f'{logging_enhanced.get_current_function()}(): missing feature column "{feat_xy}", ' \
                                  f'so cannot calculate avg position. Columns = {list(df.columns)}'
            logging_enhanced.log_then_raise(err_missing_feature, logger, KeyError)

    # hindpaw_left = config.get_part('HINDPAW_LEFT') if hindpaw_left is None else hindpaw_left
    # hindpaw_right = config.get_part('HINDPAW_RIGHT') if hindpaw_right is None else hindpaw_right
    # for feat, xy in itertools.product((hindpaw_left, hindpaw_right), ['x', 'y']):
    #     if f'{feat}_{xy}' not in df.columns:
    #         err_missing_feature = f'{logging_dibs.get_current_function()}(): missing feature column "{feat}_{xy}", so cannot calculate avg position. Columns = {list(df.columns)}'.replace('\\', '')
    #         logging_dibs.log_then_raise(err_missing_feature, logger, KeyError)

    #
    df = df.copy() if copy else df

    # Execute
    feature_1_xy: np.ndarray = df[[f'{bodypart_1}_x', f'{bodypart_1}_y']].values
    feature_2_xy: np.ndarray = df[[f'{bodypart_2}_x', f'{bodypart_2}_y']].values

    output_new_bodypart_xy: np.ndarray = np.array(list(map(average_vector_between_n_vectors, feature_1_xy, feature_2_xy)))

    # Create DataFrame from result; attach to existing data
    # df_avg = pd.DataFrame(output_feature_xy, columns=[f'{output_bodypart}_x', f'{output_bodypart}_y'])
    # df = pd.concat([df, df_avg], axis=1)
    df[[f'{output_bodypart}_x', f'{output_bodypart}_y']] = output_new_bodypart_xy

    return df


def attach_feature_distance_between_2_bodyparts(df: pd.DataFrame, bodypart_1, bodypart_2, output_feature_name, resolve_bodyparts_with_config_ini=False, copy=False) -> pd.DataFrame:
    """

    :param df: (DataFrame)
    :param bodypart_1: (str)
    :param bodypart_2: (str)
    :param output_feature_name: (str)
    :param resolve_bodyparts_with_config_ini: (bool)
    :param copy: (bool)
    :return:
    """
    # Arg checking
    check_arg.ensure_type(df, pd.DataFrame)
    bodypart_1 = config.get_part(bodypart_1) if resolve_bodyparts_with_config_ini else bodypart_1
    bodypart_2 = config.get_part(bodypart_2) if resolve_bodyparts_with_config_ini else bodypart_2
    for feat, xy in itertools.product((bodypart_1, bodypart_2), ['x', 'y']):
        bodypart_xy = f'{feat}_{xy}'
        if bodypart_xy not in set(df.columns):
            err_missing_feature = f'{logging_enhanced.get_current_function()}(): missing feature column "{bodypart_xy}", so cannot calculate avg position. Columns = {list(df.columns)}'
            logging_enhanced.log_then_raise(err_missing_feature, logger, KeyError)
    # Resolve kwargs
    df = df.copy() if copy else df
    # Execute
    feature_1_xy_arr = df[[f'{bodypart_1}_x', f'{bodypart_1}_y']].values
    feature_2_xy_arr = df[[f'{bodypart_2}_x', f'{bodypart_2}_y']].values

    distance_between_features_array: np.ndarray = np.array(list(map(distance_between_two_arrays, feature_1_xy_arr, feature_2_xy_arr)))
    # Create DataFrame from result, attach to existing data
    df[output_feature_name] = distance_between_features_array
    # df_avg = pd.DataFrame(distance_between_features_array, columns=[output_feature_name, ])
    # df = pd.concat([df, df_avg], axis=1)

    return df


def average_xy_between_2_features(df: pd.DataFrame, bodypart_1, bodypart_2, output_bodypart, copy=False) -> pd.DataFrame:
    """
    Returns 2-d array where the average location between feature1 and feature2
    """
    # Arg checking
    check_arg.ensure_type(df, pd.DataFrame)
    for bodypart, xy in itertools.product((bodypart_1, bodypart_2), ['x', 'y']):
        featxy = f'{bodypart}_{xy}'
        if featxy not in df.columns:
            err_missing_feature = f'{logging_enhanced.get_current_function()}(): missing feature column "{featxy}", ' \
                                  f'so cannot calculate avg position. Columns = {list(df.columns)}'
            logging_enhanced.log_then_raise(err_missing_feature, logger, KeyError)
    # Resolve kwargs
    df = df.copy() if copy else df
    # Execute
    feature1_xy_arr = df[[f'{bodypart_1}_x', f'{bodypart_1}_y']].values
    feature2_xy_arr = df[[f'{bodypart_2}_x', f'{bodypart_2}_y']].values
    avg_feature1_feature2_xy_arr: np.ndarray = np.array(list(map(average_vector_between_n_vectors, feature1_xy_arr, feature2_xy_arr)))
    # Create DataFrame from result, attach to existing data
    df_avg = pd.DataFrame(avg_feature1_feature2_xy_arr, columns=[f'{output_bodypart}_x', f'{output_bodypart}_y'])
    df = pd.concat([df, df_avg], axis=1)

    return df


def attach_feature_velocity_of_bodypart(df: pd.DataFrame, bodypart: str, action_duration: float, output_feature_name: str, infer_bodypart_name_from_config=False, copy=False) -> pd.DataFrame:
    """
    Attaches a new column to DataFrame that is the velocity of a SINGLE bodypart.
    :param df: (DataFrame)
    :param bodypart: (str) A feature in the DataFrame which has columns for "_x" and a "_y" suffixes.
    :param action_duration: (float) The number of seconds in which the action occurs.
    :param output_feature_name: (str) The name of the column that gets added to DataFrame
    :param infer_bodypart_name_from_config: (bool) If true, input bodypart will be the CONFIG.INI name,
        not the literal column name for that part
    :param copy: (bool) If True, create a copy of the input DataFrame for result. Otherwise,
        transform existing input DataFrame
    :return: (DataFrame)
    """
    # Check args
    check_arg.ensure_type(df, pd.DataFrame)
    check_arg.ensure_type(bodypart, str)
    check_arg.ensure_type(output_feature_name, str)
    check_arg.ensure_type(copy, bool)
    check_arg.ensure_type(infer_bodypart_name_from_config, bool)
    # Resolve kwargs
    bodypart = config.get_part(bodypart) if infer_bodypart_name_from_config else bodypart
    df = df.copy() if copy else df
    # Calculate velocities
    velocity_array: np.ndarray = velocity_of_xy_feature(df[[f'{bodypart}_x', f'{bodypart}_y']].values, action_duration)
    # With output array of values, attach to DataFrame
    df[f'{output_feature_name}'] = velocity_array

    return df


# def attach_snout_tail_angle(df, output_feature_name, copy=False) -> pd.DataFrame:
#     df = df.copy() if copy else df
#     # TODO: HIGH: implement
#     df[output_feature_name] = 1  # <- this is a stand-in
#     return df


def attach_angle_between_bodyparts(df, bodypart_1, bodypart_2, output_feature_name, copy=False) -> pd.DataFrame:
    """

    :param df: (DataFrame)
    :param bodypart_1: (str)
    :param bodypart_2: (str)
    :param output_feature_name: (str)
    :param copy: (bool)
    :return: (DataFrame)
    """
    df = df.copy() if copy else df
    # TODO: HIGH: implement

    df[output_feature_name] = 1.  # <- this is a stand-in

    return df


### Numpy array feature creation functions

def distance_between_two_arrays(arr1, arr2) -> float:
    """
    Calculates the distance between two arrays of 2-dimensions (1 row, n columns), assuming
    the first column in both arrays is the x-data and the second column is the y-data.
    Returns float distance between the two arrays.

    :param arr1: (Array)
    :param arr2: (Array)
        arr1/arr2 Example:
            [[5.   2.   3.75]]
            [[20.  15.5  7. ]]

    :returns: (float)
        Example output: 20.573040611440984

    """
    check_arg.ensure_numpy_arrays_are_same_shape(arr1, arr2)

    # Execute
    try:
        distance = (np.sum((arr1 - arr2)**2))**0.5
    except ValueError as ve:
        # Raises ValueError if array shape is not the same
        err = f'Error occurred when calculating distance between two arrays. ' \
              f'Array 1 = "{arr1}" (shape = "{arr1.shape}"). ' \
              f'Array 2 = "{arr2}" (shape = "{arr2.shape}"). Error raised is: {repr(ve)}.'
        logger.error(err)
        raise ve
    return distance


def velocity_of_xy_feature(arr: np.ndarray, secs_between_rows: float) -> np.ndarray:
    """

    :param arr:
        Example:

    :param secs_between_rows: (float) Should be the value obtained from (t_n - t_n-1).
    :return: outputs a 1-d array of velocities of each row
        since v(xy@t=1) = (xy1 - xy0) / (t1 - t0), we will also need the time between each row
    """
    # Arg checking
    check_arg.ensure_type(arr, np.ndarray)
    # TODO: add array shape check (should be shape of (n_rows, 2 columns)
    check_arg.ensure_type(secs_between_rows, float, int)
    # Execute
    # TODO: low: implement a vectorized function later
    veloc_values = [np.NaN for _ in range(len(arr))]  # velocity cannot be determined for t0, so remains as NAN
    for i in range(1, arr.shape[0]):
        veloc_i = distance_between_two_arrays(arr[i], arr[i-1]) / secs_between_rows
        veloc_values[i] = veloc_i

    # Last minute result checking
    if len(veloc_values) != arr.shape[0]:
        err_mismatch_input_output = f'The length of the input array and the length of the ' \
                                    f'output array do not match. This is incorrect. Input ' \
                                    f'array length = {arr.shape[0]}, and output array length = {len(veloc_values)}'
        logging_enhanced.log_then_raise(err_mismatch_input_output, logger, ValueError)

    veloc_array = np.array(veloc_values)

    # Sanity check
    if veloc_array.shape != (len(arr), ):
        err_incorrect_columns = f'The return array should just have one column of velocities but an incorrect number of columns was discovered. Number of columns = {veloc_array.shape[1]} (return array shape = {veloc_array.shape}).'
        logging_enhanced.log_then_raise(err_incorrect_columns, logger, ValueError)

    return veloc_array


def delta_two_body_parts_angle(body_part_arr_1, body_part_arr_2) -> np.ndarray:
    """
    TODO: explain the math
    :param body_part_arr_1: (array) Expected dimensions of (n_rows, 2 columns)
    :param body_part_arr_2: (array) Expected dimensions of (n_rows, 2 columns)
    :return:
    """
    # TODO: Aaron this is the function stub
    x = np.array([])
    return x


### Binning
def average_values_over_moving_window(data, method, n_frames: int) -> np.ndarray:
    """
    Use a moving window which covers
    :param data:
    :param method:
    :param n_frames:
    :return:
    """
    # Arg checking
    valid_methods: set = {'avg', 'sum', 'mean', 'average'}
    check_arg.ensure_type(method, str)
    if method not in valid_methods:
        err = f'Input method ({method}) was not a valid method- to apply to a feature. Valid methods: {valid_methods}'
        logger.error(err)
        raise ValueError(err)
    if not isinstance(n_frames, int):
        type_err = f'Invalid type found for n_Frames TODO elaborate. FOund type: {type(n_frames)}'
        logger.error(type_err)
        raise TypeError(type_err)
    # Arg resolution
    if method in {'avg', 'mean', }:
        averaging_function = statistics.mean
    elif method == 'sum':
        averaging_function = statistics.sum_args
    else:
        err = f'{logging_enhanced.get_current_function()}(): This should never be read since ' \
              f'method was validated earlier in function'
        logger.error(err)
        raise TypeError(err)

    if isinstance(data, pd.Series):
        data = data.values

    # Create iterators
    iterators = itertools.tee(data, n_frames)
    for i in range(len(iterators)):
        for _ in range(i):
            next(iterators[i], None)
    # Execute
    # TODO: rename `asdf`
    asdf = [averaging_function(*iters_tuple) for iters_tuple in itertools.zip_longest(*iterators, fillvalue=float('nan'))]

    return_array = np.array(asdf)

    return return_array


def average_array_into_bins(arr, n_rows_per_bin, average_method: str):
    """"""
    # ARg checking
    valid_avg_methods = {'sum', 'avg', 'average', 'mean', 'first'}
    if average_method not in valid_avg_methods:
        err_invalid_method = f'Invalid method specified: {average_method}'  # TODO: low: improve err msg later
        logging_enhanced.log_then_raise(err_invalid_method, logger, ValueError)
    #
    if average_method in {'sum', }:
        method = statistics.sum_args
    elif average_method in {'avg', 'average', 'mean', }:
        method = statistics.mean
    elif average_method in {'first', }:
        method = statistics.first_arg
    else:
        err_impossible = f'An impossible avg method found. SHouldve been arg checked at start. Value = "{average_method}"'  # TODO: low: improve err msg later
        logger.error(err_impossible)
        raise ValueError(err_impossible)

    # Execute
    integrated_data = []
    for i in range(0, len(arr), n_rows_per_bin):
        integrated_val = method(*arr[i: i + n_rows_per_bin])
        integrated_data.append(integrated_val)

    integrated_arr = np.array(integrated_data)
    return integrated_arr


def integrate_df_feature_into_bins(df: pd.DataFrame, map_features_to_bin_methods: dict, n_rows: int, copy: bool = False) -> pd.DataFrame:
    """
    TODO
    :param df: (DataFrame)
    :param map_features_to_bin_methods: Dictionary of features and associated averaging methods. If a column name in the
        argument DataFrame (`df`) is NOT specified in this mapping, then the first value in that batch of rows
        will be chosen
        e.g.:   {'velocity': 'sum',
                'distance_1': 'avg',
                'distance_2': 'first', }

    :param n_rows: (int)
    :param copy: (bool)
    :return:
    """
    # Arg checking
    check_arg.ensure_type(df, pd.DataFrame)
    check_arg.ensure_type(n_rows, int)
    for feature in map_features_to_bin_methods.keys():
        if feature not in df.columns:
            err = f'{logging_enhanced.get_current_function()}(): TODO: elaborate: feature not found: "{feature}". ' \
                  f'Cannot integrate into ?ms bins.'
            logger.error(err)
            raise ValueError(err)
    # Kwarg resolution
    df = df.copy() if copy else df

    # Execute
    data_list_of_arrays, corresponding_cols_list = [], []
    for col in df.columns:
        if col in map_features_to_bin_methods:
            integrated_array = average_array_into_bins(df[col].values, n_rows, map_features_to_bin_methods[col])
        else:
            integrated_array = average_array_into_bins(df[col].values, n_rows, 'first')

        data_list_of_arrays.append(integrated_array)
        corresponding_cols_list.append(col)

    out_df = pd.DataFrame(np.array(data_list_of_arrays), index=corresponding_cols_list).transpose()

    return out_df


#### Newer, reworked feature engineer from previous ############################

def adaptively_filter_dlc_output(in_df: pd.DataFrame, copy=False) -> Tuple[pd.DataFrame, List[float]]:  # TODO: implement new adaptive-filter_data for new data pipelineing
    """ *NEW* --> Successor function to old method in likelikhood processing. Uses new DataFrame type for input/output.
    Takes in a ____ TODO: low: ...

    Usually this function is completed directly after reading in DLC data.

    (Described as adaptive high-pass filter by author)
    Note: this function follows same form as legacy only for
        continuity reasons. Can be refactored for performance later.

    Note: the top row ends up as ZERO according to original algorithm implementation; however, we do not remove
        the top row like the original implementation.
    :param in_df: (DataFrame) expected: raw DataFrame of DLC results right after reading in using dibs.read_csv().

        EXAMPLE `df_input_data` input:  # TODO: remove bodyparts_coords col? Check dibs.io.read_csv() return format.
              bodyparts_coords        Snout/Head_x       Snout/Head_y Snout/Head_likelihood Forepaw/Shoulder1_x Forepaw/Shoulder1_y Forepaw/Shoulder1_likelihood  ...                                          scorer          source
            0                0     1013.7373046875   661.953857421875                   1.0  1020.1138305664062   621.7146606445312           0.9999985694885254  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
            1                1  1012.7627563476562  660.2426147460938                   1.0  1020.0912475585938   622.9310913085938           0.9999995231628418  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
            2                2  1012.5982666015625   660.308349609375                   1.0  1020.1837768554688   623.5087280273438           0.9999994039535522  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
            3                3  1013.2752685546875  661.3504028320312                   1.0     1020.6982421875   624.2875366210938           0.9999998807907104  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
            4                4  1013.4093017578125  661.3643188476562                   1.0  1020.6074829101562     624.48486328125           0.9999998807907104  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
    :param copy: (bool) Indicates whether to create an entirely new DataFrame object as a result so that
        the original input DataFrame is not changed afterwards.

    :return
        : DataFrame of filtered data
            Example:
                    EXAMPLE `df_input_data` input:  # TODO: remove bodyparts_coords col? Check dibs.io.read_csv() return format.
              bodyparts_coords        Snout/Head_x       Snout/Head_y Forepaw/Shoulder1_x Forepaw/Shoulder1_y  ...                                          scorer          source
            0                0     1013.7373046875   661.953857421875  1020.1138305664062   621.7146606445312  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
            1                1  1012.7627563476562  660.2426147460938  1020.0912475585938   622.9310913085938  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
            2                2  1012.5982666015625   660.308349609375  1020.1837768554688   623.5087280273438  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
            3                3  1013.2752685546875  661.3504028320312     1020.6982421875   624.2875366210938  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
            4                4  1013.4093017578125  661.3643188476562  1020.6074829101562     624.48486328125  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  Video1_DLC.csv
        : 1D array, percent filtered per BODYPART

    """
    # TODO: HIGH: for this function that does not have expected cols (like 'scorer', etc.) it should not fail!
    # Checking args
    check_arg.ensure_type(in_df, pd.DataFrame)
    # Continue
    # # Scorer
    set_in_df_columns = set(in_df.columns)
    if 'scorer' not in set_in_df_columns:
        col_not_found_err = f'TODO: "scorer" col not found but should exist (as a result from dibs.read_csv()) // ' \
                            f'All columns: {in_df.columns}'
        logger.error(col_not_found_err)
        raise ValueError(col_not_found_err)  # TODO: should this raise an error?
    scorer_values = np.unique(in_df['scorer'])
    if len(scorer_values) != 1:
        err = f'TODO: there should be 1 unique scorer value. If there are more than 1, too many values. TODO '
        logger.error(err)
        raise ValueError(err)  # TODO: should this raise an error?
    scorer_value: str = scorer_values[0]

    # # Source
    if 'source' in set_in_df_columns:
        source_filenames_values = np.unique(in_df['source'])
        if len(scorer_values) != 1:
            err = f'TODO: there should be 1 unique source value. If there are more than 1, too many values, ' \
                  f'makes no sense to adaptively filter over different datasets.'
            logger.error(err)
            raise ValueError(err)  # # TODO: should this raise an error?
        source = in_df['source'].values[0]
    else:
        source = None

    # if 'file_source' in set_in_df_columns:
    file_source = in_df['file_source'][0] if 'file_source' in set_in_df_columns else None
    data_source = in_df['data_source'][0] if 'data_source' in set_in_df_columns else None

    # Resolve kwargs
    df = in_df.copy() if copy else in_df


    # Loop over columns, aggregate which indices in the data fall under which category.
    #   x, y, and likelihood are the three main types of columns output from DLC.
    x_index, y_index, l_index, percent_filterd_per_bodypart__perc_rect = [], [], [], []
    map_back_index_to_col_name = {}
    coords_cols_names = []
    for idx_col, col in enumerate(df.columns):
        # Assign ___ TODO
        map_back_index_to_col_name[idx_col] = col
        # Columns take the regular form `FEATURE_(x|y|likelihood|coords|)`, so split by final _ OK
        column_suffix = col.split('_')[-1]
        if column_suffix == "likelihood":
            l_index.append(idx_col)
        elif column_suffix == "x":
            x_index.append(idx_col)
        elif column_suffix == "y":
            y_index.append(idx_col)
        elif column_suffix == 'coords':  # todo: delte this elif. Coords should be dropped with the io.read_csv implementation?
            # Record and check later...likely shouldn't exist anymore since its just a numbered col with no data.
            coords_cols_names.append(col)
        elif col == 'scorer': pass  # Ignore 'scorer' column. It tracks the DLC data source.
        elif col == 'source': pass  # Keeps track of CSV/h5 source
        elif col == 'frame': pass  # Keeps track of frame numbers
        elif col == 'file_source': pass
        elif col == 'data_source': pass
        else:
            err = f'{inspect.stack()[0][3]}(): An inappropriate column header was found: ' \
                  f'{column_suffix}. Column = "{col}". ' \
                  f'Check on CSV to see if has an unexpected output format from DLC.'
            logger.error(err)
            # raise ValueError(err)
    if len(coords_cols_names) > 1:
        err = f'An unexpected number of columns were detected that contained the substring "coords". ' \
              f'Normally, there is only 1 "coords" column in a DLC output CSV, but this is an abnormal case. ' \
              f'Coords columns: {coords_cols_names} / df.head(): {df.head().to_string()}'
        logger.error(err)
        raise ValueError(err)

    # Sanity check
    if len(coords_cols_names) > 0: raise ValueError(f'coords should not exist anymore')

    # Slice data into separate arrays based on column names (derived earlier from the respective index)
    data_x: np.ndarray = np.array(df.iloc[:, np.array(x_index)])
    data_y: np.ndarray = np.array(df.iloc[:, np.array(y_index)])
    data_likelihood: np.ndarray = np.array(df.iloc[:, np.array(l_index)])
    # Note: at this point, the above 3 data arrays will all have the exact same shape

    # The below variable is instantiated with same rows as total minus 1 (for reasons TBD) and
    #   with column room for x and y values (it appears as though the likelihood values disappear)
    array_data_filtered = np.zeros((data_x.shape[0], (data_x.shape[1]) * 2))  # Initialized as zeroes to be populated later  # currdf_filt: np.ndarray = np.zeros((data_x.shape[0]-1, (data_x.shape[1]) * 2))

    logger.debug(f'{inspect.stack()[0][3]}(): Computing data threshold to forward fill any sub-threshold (x,y)...')
    percent_filterd_per_bodypart__perc_rect: List = [0. for _ in range(data_likelihood.shape[1])]  # for _ in range(data_lh.shape[1]): perc_rect.append(0)

    # Loop over data and do adaptive filtering.
    # logger.debug(f'{inspect.stack()[0][3]}: Loop over data and do adaptive filtering.')
    idx_col = 0
    for idx_col_i in tqdm(range(data_likelihood.shape[1]),
                          desc=f'{logging_enhanced.get_current_function()}(): Adaptively filtering DLC data...',
                          disable=False if config.stdout_log_level.strip().upper() == 'DEBUG' else True):
        # Get histogram of likelihood data in col_i (ignoring first row since its just labels (e.g.: [x  x  x  x ...]))
        histogram, bin_edges = np.histogram(data_likelihood[:, idx_col_i].astype(np.float))
        # Determine "rise".
        rise_arr = np.where(np.diff(histogram) >= 0)
        if isinstance(rise_arr, tuple):  # Sometimes np.where returns a tuple depending on input dims
            rise_arr = rise_arr[0]
        rise_0, rise_1 = rise_arr[0], rise_arr[1]

        # Threshold for bin_edges?
        if rise_arr[0] > 1:
            likelihood_threshold: np.ndarray = (bin_edges[rise_0] + bin_edges[rise_0 - 1]) / 2
        else:
            likelihood_threshold: np.ndarray = (bin_edges[rise_1] + bin_edges[rise_1 - 1]) / 2

        # Change data type to float because its currently string
        data_likelihood_col_i = data_likelihood[:, idx_col_i].astype(np.float)

        # Record percent filtered (for "reasons")
        percent_filterd_per_bodypart__perc_rect[idx_col_i] = np.sum(data_likelihood_col_i < likelihood_threshold) / data_likelihood.shape[0]

        # Note: the slicing below is just slicing the x and y columns.
        for i in range(data_likelihood.shape[0] - 1):  # TODO: low: rename `i`
            if data_likelihood_col_i[i + 1] < likelihood_threshold:
                array_data_filtered[i + 1, (2 * idx_col_i):(2 * idx_col_i + 2)] = \
                    array_data_filtered[i, (2 * idx_col_i):(2 * idx_col_i + 2)]
            else:
                array_data_filtered[i + 1, (2 * idx_col_i):(2 * idx_col_i + 2)] = \
                    np.hstack([data_x[i, idx_col_i], data_y[i, idx_col_i]])

    # ### Adaptive filtering is all done. Clean up and return.
    # # Remove first row in data array (values are all zeroes)
    # array_filtered_data_without_first_row = np.array(array_data_filtered[1:]).astype(np.float)
    array_filtered_data_without_first_row = np.array(array_data_filtered[:]).astype(np.float)

    # Create DataFrame with columns by looping over x/y indices.
    columns_ordered: List[str] = []
    for x_idx, y_idx in zip(x_index, y_index):
        columns_ordered += [map_back_index_to_col_name[x_idx], map_back_index_to_col_name[y_idx]]

    # Create frame, replace 'scorer' column. Return.
    df_adaptively_filtered_data = pd.DataFrame(array_filtered_data_without_first_row, columns=columns_ordered)
    df_adaptively_filtered_data['scorer'] = scorer_value
    # Re-add source, etc
    if source is not None:
        df_adaptively_filtered_data['source'] = source
    if file_source is not None:
        df_adaptively_filtered_data['file_source'] = file_source
    if data_source is not None:
        df_adaptively_filtered_data['data_source'] = data_source

    df_adaptively_filtered_data['frame'] = range(len(df_adaptively_filtered_data))
    if len(in_df) != len(df_adaptively_filtered_data):
        missing_rows_err = f'Input df has {len(df)} rows but output df ' \
                           f'has {len(df_adaptively_filtered_data)}. Should be same number.'
        logger.error(missing_rows_err)
        raise ValueError(missing_rows_err)
    return df_adaptively_filtered_data, percent_filterd_per_bodypart__perc_rect


def average_vector_between_n_vectors(*arrays) -> np.ndarray:
    """
    TODO
    """
    # Arg Checks
    if len(arrays) == 0:
        cannot_average_0_arrays_err = f'Cannot average between 0 arrays'  # TODO: improve err message
        logger.error(cannot_average_0_arrays_err)
        raise ValueError(cannot_average_0_arrays_err)
    for arr in arrays:
        check_arg.ensure_type(arr, np.ndarray)
    check_arg.ensure_numpy_arrays_are_same_shape(*arrays)
    #
    set_of_shapes = set([arr.shape for arr in arrays])
    if len(set_of_shapes) > 1:
        err_disparate_shapes_of_arrays = f'Array shapes are not the same. Shapes: [{set_of_shapes}]'  # TODO
        logger.error(err_disparate_shapes_of_arrays)
        raise ValueError(err_disparate_shapes_of_arrays)
    # Execute
    averaged_array = arrays[0]
    for i in range(1, len(arrays)):
        averaged_array += arrays[i]
    averaged_array = averaged_array / len(arrays)
    # TODO: med/high: implement !!!
    return averaged_array


def engineer_7_features_dataframe(df: pd.DataFrame, features_names_7: List[str] = ['DistFrontPawsTailbaseRelativeBodyLength', 'DistBackPawsBaseTailRelativeBodyLength', 'InterforepawDistance', 'BodyLength', 'SnoutToTailbaseChangeInAngle', 'SnoutSpeed', 'TailbaseSpeed', ], map_names: dict = None, copy: bool = False, win_len: int = None) -> pd.DataFrame:
    # TODO: med: ensure ALL columns in input DataFrame also come out of the output
    # TODO: review https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
    #   Computationally intensive! Work on performance later.
    """ *NEW*
    A copy of the similarly-named feature engineering function; however, the top array element is NOT chopped off,
    ensuring that the number of sample that enter is the same number that exits.

    There are 6 required body parts:
        -

    Note: you'll end up with 1 less row than you started with on input
    :param features_names_7:  TODO?
    :param win_len: TODO
    :param df: (DataFrame)
    :param map_names (dict)
    :param copy:
    :return: (DataFrame)
    """
    if win_len is None:
        win_len = win_len_formula(config.VIDEO_FPS)
    logger.debug(f'{logging_enhanced.get_current_function()}(): `win_len` was calculated as: {win_len}')

    required_features_from_config = {
        'Head': 'SNOUT/HEAD',
        'ForepawLeft': 'LEFT_SHOULDER/FOREPAW',
        'ForepawRight': 'RIGHT_SHOULDER/FOREPAW',
        'HindpawLeft': 'LEFT_HIP/HINDPAW',
        'HindpawRight': 'RIGHT_HIP/HINDPAW',
        'Tailbase': 'TAILBASE',
    }
    ###

    # # Arg checks
    # Initial args checks
    check_arg.ensure_type(df, pd.DataFrame)
    check_arg.ensure_type(win_len, int)
    assert len(features_names_7) == 7, f'features_names_7 should be 7 items long. TODO: formally replace this error later.'

    # Replace any keys to check for in config.ini file
    if map_names is not None:
        check_arg.ensure_type(map_names, dict)
        for mouse_part, config_file_key_name in map_names.items():
            required_features_from_config[mouse_part] = config_file_key_name

    # Check for required columns
    set_df_columns = set(df.columns)
    # Check if the required parts are present in data set before proceeding
    for feature, data_label in required_features_from_config.items():
        feature_x, feature_y = f'{config.get_part(data_label)}_x', f'{config.get_part(data_label)}_y'
        if feature_x not in set_df_columns:
            err_feature_x_missing = f'`{feature_x}` is required for this feature ' \
                                    f'engineering but was not found. All submitted columns are: {df.columns}'
            logger.error(err_feature_x_missing)
            raise ValueError(err_feature_x_missing)
        if feature_y not in set_df_columns:
            err_feature_y_missing = f'`{feature_y}` is required for this feature ' \
                                    f'engineering but was not found. All submitted columns are: {df.columns}'
            logger.error(err_feature_y_missing)
            raise ValueError(err_feature_y_missing)
        set_df_columns -= {feature_x, feature_y}

    if 'scorer' in df.columns:
        unique_scorers = np.unique(df['scorer'].values)
        if len(unique_scorers) != 1:
            err = f'More than one scorer value found. Expected only 1. Scorer values: {unique_scorers}'
            logger.error(err)
            raise ValueError(err)
        scorer = unique_scorers[0]
    else:
        scorer = None
    if 'source' in df.columns:
        unique_sources = np.unique(df['source'].values)
        if len(unique_sources) != 1:
            err = f'More than one source value found. Expected only 1. source values: {unique_sources}'
            logger.error(err)
            raise ValueError(err)
        source = unique_sources[0]
    else:
        source = None
    # Solve kwargs

    # Do
    # Enumerate necessary variables for specifying data
    num_data_rows: int = len(df)
    """
        req_config_feats = {
        'Head': 'SNOUT/HEAD',
        'ForepawLeft': 'LEFT_SHOULDER/FOREPAW',
        'ForepawRight': 'RIGHT_SHOULDER/FOREPAW',
        'HindpawLeft': 'LEFT_HIP/HINDPAW',
        'HindpawRight': 'RIGHT_HIP/HINDPAW',
        'Tailbase': 'TAILBASE',
    }
    """
    head_x = f'{config.get_part(required_features_from_config["Head"])}_x'
    head_y = f'{config.get_part(required_features_from_config["Head"])}_y'
    left_shoulder_x = f'{config.get_part(required_features_from_config["ForepawLeft"])}_x'
    left_shoulder_y = f'{config.get_part(required_features_from_config["ForepawLeft"])}_y'
    right_shoulder_x = f'{config.get_part(required_features_from_config["ForepawRight"])}_x'
    right_shoulder_y = f'{config.get_part(required_features_from_config["ForepawRight"])}_y'
    left_hip_x = f'{config.get_part(required_features_from_config["HindpawLeft"])}_x'
    left_hip_y = f'{config.get_part(required_features_from_config["HindpawLeft"])}_y'
    right_hip_x, right_hip_y = [f'{config.get_part(required_features_from_config["HindpawRight"])}_{suffix}'
                                for suffix in ('x', 'y')]
    tailbase_x, tailbase_y = [f'{config.get_part(required_features_from_config["Tailbase"])}_{suffix}'
                              for suffix in ('x', 'y')]

    ####################################################################################################################
    # Create intermediate variables to solve for final features.

    # fpd
    inter_forepaw_distance = df[[left_shoulder_x, left_shoulder_y]].values - df[[right_shoulder_x, right_shoulder_y]].values  # Previously: 'fpd'

    # cfp
    cfp__center_between_forepaws = np.vstack((
        (df[left_shoulder_x].values + df[right_shoulder_x].values) / 2,
        (df[left_shoulder_y].values + df[right_shoulder_y].values) / 2,
    )).T  # Previously: cfp

    # chp
    chp__center_between_hindpaws = np.vstack((
        (df[left_hip_x].values + df[right_hip_x].values) / 2,
        (df[left_hip_y].values + df[right_hip_y].values) / 2,
    )).T
    # cfp_pt
    dFT__cfp_pt__center_between_forepaws__minus__proximal_tail = np.vstack(([
        cfp__center_between_forepaws[:, 0] - df[tailbase_x].values,
        cfp__center_between_forepaws[:, 1] - df[tailbase_y].values,
    ])).T

    # chp_pt
    chp__center_between_hindpaws__minus__proximal_tail = np.vstack(([
        chp__center_between_hindpaws[:, 0] - df[tailbase_x].values,
        chp__center_between_hindpaws[:, 1] - df[tailbase_y].values,
    ])).T  # chp_pt

    # sn_pt
    snout__proximal_tail__distance__aka_BODYLENGTH = np.vstack(([
        df[head_x].values - df[tailbase_x].values,
        df[head_y].values - df[tailbase_y].values,
    ])).T  # previously: sn_pt

    ### Create the 4 static measurement features
    inter_forepaw_distance__normalized = np.zeros(num_data_rows)  # originally: fpd_norm
    cfp_pt__center_between_forepaws__minus__proximal_tail__normalized = np.zeros(num_data_rows)  # originally: cfp_pt_norm
    chp__proximal_tail__normalized = np.zeros(num_data_rows)  # originally: chp_pt_norm
    snout__proximal_tail__distance__aka_BODYLENGTH__normalized = np.zeros(num_data_rows)  # originally: sn_pt_norm

    for j in range(1, num_data_rows):
        # Each of these steps below produces a single-valued-array (shape: (1,1)) and inserted it into the noramlized
        inter_forepaw_distance__normalized[j] = np.array(np.linalg.norm(inter_forepaw_distance[j, :]))
        cfp_pt__center_between_forepaws__minus__proximal_tail__normalized[j] = np.linalg.norm(dFT__cfp_pt__center_between_forepaws__minus__proximal_tail[j, :])
        chp__proximal_tail__normalized[j] = np.linalg.norm(chp__center_between_hindpaws__minus__proximal_tail[j, :])
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized[j] = np.linalg.norm(snout__proximal_tail__distance__aka_BODYLENGTH[j, :])
    ## "Smooth" features for final use
    # Body length (1)
    snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed = statistics.boxcar_center(
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized, win_len)  # sn_pt_norm_smth
    # Inter-forepaw distance (4)
    inter_forepaw_distance__normalized__smoothed = statistics.boxcar_center(
        inter_forepaw_distance__normalized, win_len)  # fpd_norm_smth
    # (2)
    snout__center_forepaws__normalized__smoothed = statistics.boxcar_center(
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized -
        cfp_pt__center_between_forepaws__minus__proximal_tail__normalized,
        win_len)  # sn_cfp_norm_smth
    # (3)
    snout__center_hindpaws__normalized__smoothed = statistics.boxcar_center(
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized -
        chp__proximal_tail__normalized,
        win_len)  # sn_chp_norm_smth

    ### Create the 3 time-varying features (out of a total of 7 final features)
    snout__proximal_tail__angle = np.zeros(num_data_rows - 1)  # originally: sn_pt_ang
    snout_speed__aka_snout__displacement = np.zeros(num_data_rows - 1)  # originally: sn_disp
    tail_speed__aka_proximal_tail__displacement = np.zeros(num_data_rows - 1)  # originally: pt_disp
    for k in range(num_data_rows - 1):
        a_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k, :], 0])
        b_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :], 0])
        c = np.cross(b_3d, a_3d)
        snout__proximal_tail__angle[k] = np.dot(
            np.dot(np.sign(c[2]), 180) / np.pi, math.atan2(np.linalg.norm(c), np.dot(
                snout__proximal_tail__distance__aka_BODYLENGTH[k, :],
                snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :])))

        snout_speed__aka_snout__displacement[k] = np.linalg.norm(
            # df[[head_x, head_y]].iloc[k + 1].values -  # TODO: IMPORTANT ******************* While this snout speed implementation matches the legacy implementation, is it really generating snout speed at all? .... Why only the x?
            # df[[head_x, head_y]].iloc[k].values)
            df[[head_x, ]].iloc[k + 1].values -  # TODO: IMPORTANT ******************* While this snout speed implementation matches the legacy implementation, is it really generating snout speed at all? .... ^
            df[[head_x, ]].iloc[k].values)

        tail_speed__aka_proximal_tail__displacement[k] = np.linalg.norm(
            # df[[tailbase_x, tailbase_y]].iloc[k+1, :].values -
            # df[[tailbase_x, tailbase_y]].iloc[k, :].values)
            df[[tailbase_x, ]].iloc[k + 1, :].values -
            df[[tailbase_x, ]].iloc[k, :].values)  # TODO: why only the x?

    snout__proximal_tail__angle__smoothed = statistics.boxcar_center(snout__proximal_tail__angle, win_len)  # sn_pt_ang_smth =>
    snout_speed__aka_snout_displacement_smoothed = statistics.boxcar_center(snout_speed__aka_snout__displacement, win_len)  # sn_disp_smth =>
    tail_speed__aka_proximal_tail__displacement__smoothed = statistics.boxcar_center(tail_speed__aka_proximal_tail__displacement, win_len)  # originally: pt_disp_smth

    # Aggregate/organize features according to original implementation
    # Note that the below features array is organized in shape: (number of features, number of records) which
    #   is typically backwards from how DataFrames are composed.
    value_to_prepend_to_time_variant_features = 0.
    features = np.vstack((
        snout__center_forepaws__normalized__smoothed[:],  # 2
        snout__center_hindpaws__normalized__smoothed[:],  # 3
        inter_forepaw_distance__normalized__smoothed[:],  # 4
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed[:],  # 1
        # time-varying features
        np.insert(snout__proximal_tail__angle__smoothed[:], 0, snout__proximal_tail__angle__smoothed[0]),  # 7
        np.insert(snout_speed__aka_snout_displacement_smoothed[:], 0, snout_speed__aka_snout_displacement_smoothed[0]),  # 5
        np.insert(tail_speed__aka_proximal_tail__displacement__smoothed[:], 0, tail_speed__aka_proximal_tail__displacement__smoothed[0]),  # 6
    ))
    # Create DataFrame for features. Flip the features so that the records run along the rows and the
    #   features are in the columns.
    features_for_dataframe: np.ndarray = features.T
    results_cols: List[str] = features_names_7

    df_engineered_features = pd.DataFrame(features_for_dataframe, columns=results_cols)
    for col in set_df_columns:
        if col not in df_engineered_features.columns:
            df_engineered_features[col] = df[col]

    if 'scorer' in set_df_columns:
        df_engineered_features['scorer'] = scorer
    if 'source' in set_df_columns:
        df_engineered_features['source'] = source
    if 'frame' in set_df_columns:
        df_engineered_features['frame'] = df['frame'].values

    return df_engineered_features


def win_len_formula(fps: int) -> int:
    """
    A mimic of the original win_len formula except without relying on numpy
    :param fps: (int)
    :return: (int)
    """
    win_len = int(round(0.05 / (1 / fps)) * 2 - 1)
    return win_len


def run_openTSNE(train_features):
  x_train = np.array(train_features)
  n_components = 3
  tsne = TSNE(
	n_components=n_components, # https://github.com/pavlin-policar/openTSNE/issues/121
	negative_gradient_method='bh',
	perplexity=30,
	metric='euclidean',
	verbose=True,
	n_jobs=10,
	random_state=42
      )
  embedding = tsne.fit(x_train)
  # np.savetxt(f"tsne{n_components}dims.csv", embedding, delimiter=',', header=",".join([f'X{i}' for i in range(embedding.shape[1])]))
  return embedding
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
from scipy.spatial import ConvexHull
from tqdm import tqdm
from typing import List, Optional, Tuple
import inspect
import itertools
import math
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
from dibs import check_arg, config, logging_enhanced, statistics

from sklearn.utils import shuffle as sklearn_shuffle_dataframe

logger = config.initialize_logger(__name__)


### Attach features as columns to a DataFrame of DLC data

def time_shifted(v, tau: int):
    """ Create time shifted copy of an input feature to allow simple time series modelling """
    v = pd.Series(v)  # TODO: Test, should take a numpy array and turn it into a Series
    return v.shift(periods=tau), 'avg'


def attach_time_shifted_data(df: pd.DataFrame, bodypart: str, tau: int, output_feature_name: str,
                             copy=False) -> pd.DataFrame:
    # Check args
    check_arg.ensure_type(df, pd.DataFrame)
    check_arg.ensure_type(bodypart, str)
    check_arg.ensure_type(output_feature_name, str)
    check_arg.ensure_type(copy, bool)
    df = df.copy() if copy else df
    # Calculate velocities
    bodyparts = [col for col in df.columns if bodypart in col]
    if not bodyparts:
        err = f'There are no columns in the data frame which contain {bodypart} as a sub string, as such we can not produce the time shifted features'
        logger.error(err)
        raise RuntimeError(err)

    for b in bodyparts:
        arr: pd.Series = df[[b]]
        tau_array: pd.Series = arr.shift(periods=tau)
        # With output array of values, attach to DataFrame
        string_difference = b.strip(bodypart)
        df[f'{output_feature_name + string_difference}'] = tau_array

    return df


def attach_train_test_split_col(df, test_col: str, test_pct: float, random_state: int,
                                sort_results_by: Optional[List[str]] = None, copy: bool = False) -> pd.DataFrame:
    """ TODO: Move this somewhere more relevant... it is used in the Basepipeline class
    TODO: For something like cross validation, we would want to allow this to be generated outside the dataset
    :param df:
    :param test_col:
    :param test_pct:
    :return:
    """
    # TODO: med: fix setting with copy warning
    df[test_col] = np.random.binomial(1, test_pct, len(df))

    logger.debug(f"{logging_enhanced.get_current_function()}(): "
                 f"Final test/train split is calculated to be: {test_pct}")

    return df


# Numpy array feature creation functions

def distance(arr1, arr2, p=2) -> (np.ndarray, str):
    """
    Calculates the p'th order distance between two arrays of 2-dimensions (m rows, n columns), where each column
    represents coordinate in 1D.
    Returns float distance between the two arrays.

    :param arr1: (Array)
    :param arr2: (Array)
        arr1/arr2 Example:
            [[5.   2.   3.75]]
            [[20.  15.5  7. ]]
    :param p: order of distance norm
    :returns: (Array)
        Example output: [20.573040611440984]

    """
    check_arg.ensure_type(arr1, np.ndarray)
    check_arg.ensure_type(arr2, np.ndarray)
    check_arg.ensure_numpy_arrays_are_same_shape(arr1, arr2)

    # Execute
    try:
        distance_array = np.sum((arr1 - arr2) ** p, axis=1) ** (1. / p)
    except ValueError as ve:
        # Raises ValueError if array shape is not the same
        err = f'Error occurred when calculating distance between two arrays. ' \
              f'Array 1 = "{arr1}" (shape = "{arr1.shape}"). ' \
              f'Array 2 = "{arr2}" (shape = "{arr2.shape}"). Error raised is: {repr(ve)}.'
        logger.error(err)
        raise ve
    return distance_array, 'avg'


def shifted_distance(arr1, period=1) -> (np.ndarray, str):
    """
    Calculate distance shifted/travelled in given period.
    :param arr1: (Array)
    :param period: int representing the number you want to shift array
    :returns (Array)
    """

    check_arg.ensure_type(arr1, np.ndarray)

    try:
        arr1_shifted = pd.DataFrame(arr1).shift(period).values.reshape((arr1.shape[0], -1))
        movement = np.nan_to_num(distance(arr1.reshape((arr1.shape[0], -1)), arr1_shifted)[0], nan=0)
    except ValueError as ve:
        # Raises ValueError if array shape is not the same
        err = f'Error occurred when calculating shifted distance of period {period}. ' \
              f'Array 1 = "{arr1}" (shape = "{arr1.shape}").'
        logger.error(err)
        raise ve
    return movement, 'avg'


def convex_hull_area(*arrays, dimension=2):
    """
    Return area of convex polynomial formed by coordinates in input array.

    :param arrays : list of ndarrays where each array is of the form bodypart_x, bodypart_y, ...
    :param dimension: Dimension of coordinate system

    :returns array : ndarray containing area of convex polygon formed by joining all the bodyparts.
    """

    [check_arg.ensure_type(arr, np.ndarray) for arr in arrays]

    def calculate_area(points):
        return ConvexHull(np.array(np.split(points, points.shape[0] / dimension))).area

    big_arr = np.concatenate(arrays, axis=1)
    assert big_arr.shape[
               1] % dimension == 0  # We need x,y coordinates for each feature and hence total cols should be even
    return np.apply_along_axis(calculate_area, axis=1, arr=big_arr), 'avg'


def average(*arrays) -> (np.ndarray, str):
    """
    Take the average of n arrays, for int n >= 1
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
    averaged_array = np.mean(np.array(arrays), axis=0)
    return averaged_array, 'avg'


def velocity(arr1) -> (np.ndarray, str):
    pass


def delta_of_array(arr: np.ndarray, action_duration: float = 1.0) -> np.ndarray:
    # Check args
    check_arg.ensure_type(arr, np.ndarray)
    if len(np.shape) != 2 or np.shape[1] != 1:
        raise RuntimeError(
            f'delta_of_array can only handle 2 dimensional arrays with a single columns. Got arr.shape: {arr.shape} instead.')
    arr = arr.flatten()
    delta_array = np.zeros(len(arr))
    delta_array[1:] = arr[:-1] - arr[1:]  # delta at t0 remains 0; Could use NaN but those can reproduce unexpectedly.
    delta_array: np.ndarray = delta_of_array(arr) / action_duration
    return delta_array


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.arccos(np.dot(v1_u, v2_u))


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
    asdf = [averaging_function(*iters_tuple) for iters_tuple in
            itertools.zip_longest(*iterators, fillvalue=float('nan'))]

    return_array = np.array(asdf)

    return return_array


def average_array_into_bins(arr, n_rows_per_bin, average_method: str):
    """"""
    # ARg checking
    valid_avg_methods = {'sum', 'avg', 'average', 'mean', 'first'}
    if average_method not in valid_avg_methods:
        err_invalid_method = f'Invalid method specified: {average_method}'  # TODO: low: improve err msg later
        logger.error(err_invalid_method)
        raise ValueError(err_invalid_method)
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
        # TODO: Use numpy instead?  Tests??
        integrated_val = method(*arr[i: i + n_rows_per_bin])
        integrated_data.append(integrated_val)

    integrated_arr = np.array(integrated_data)
    return integrated_arr


def integrate_df_feature_into_bins(df: pd.DataFrame, map_features_to_bin_methods: dict, n_rows: int,
                                   copy: bool = False) -> pd.DataFrame:
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

def adaptively_filter_dlc_output(in_df: pd.DataFrame, copy=False) -> Tuple[
    pd.DataFrame, List[float]]:  # TODO: implement new adaptive-filter_data for new data pipelineing
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
        col_not_found_err = f'Column named "scorer" not found but should exist (as a result from dibs.read_csv()) // ' \
                            f'All columns: {in_df.columns}'
        logger.error(col_not_found_err)
        raise ValueError(col_not_found_err)  # TODO: should this raise an error?
    scorer_values = np.unique(in_df['scorer'].values)
    if len(scorer_values) != 1:
        err = f'There should be 1 unique scorer value. If there are more than 1, too many values. Value are: {scorer_values}.'
        logger.error(err)
        raise ValueError(err)  # TODO: low: should this raise an error?
    scorer_value: str = scorer_values[0]

    # # Source
    if 'source' in set_in_df_columns:
        if len(scorer_values) != 1:
            err = f'There should be 1 unique "source" value. If there is more than 1, too many values, ' \
                  f'makes no sense to adaptively filter over different datasets.'
            logger.error(err)
            raise ValueError(err)  # TODO: low: should this raise an error?
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
    x_index, y_index, l_index = [], [], []
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
        elif column_suffix == 'coords':  # todo: low: delete this elif. Coords should be dropped with the io.read_csv implementation?
            # Record and check later...likely shouldn't exist anymore since its just a numbered col with no data.
            coords_cols_names.append(col)
        elif col == 'scorer':
            pass  # Ignore 'scorer' column. It tracks the DLC data source.
        elif col == 'source':
            pass  # Keeps track of CSV/h5 source
        elif col == 'frame':
            pass  # Keeps track of frame numbers
        elif col == 'file_source':
            pass
        elif col == 'data_source':
            pass
        else:
            err = f'{inspect.stack()[0][3]}(): An inappropriate column header was found: ' \
                  f'{column_suffix}. Column name = "{col}". ' \
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
    array_data_filtered = np.zeros((data_x.shape[0], (data_x.shape[1]) * 2))

    logger.debug(f'{inspect.stack()[0][3]}(): Computing data threshold to forward fill any sub-threshold (x,y)...')
    percent_filtered_per_bodypart: List = [0. for _ in range(data_likelihood.shape[1])]

    # Loop over data and do adaptive filtering.
    for idx_col_i in tqdm(range(data_likelihood.shape[1]),
                          desc=f'{logging_enhanced.get_current_function()}(): Adaptively filtering DLC columns...',
                          disable=True if config.stdout_log_level.strip().upper() != 'DEBUG' else False):
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
        percent_filtered_per_bodypart[idx_col_i] = np.sum(data_likelihood_col_i < likelihood_threshold) / \
                                                   data_likelihood.shape[0]

        # Note: the slicing below is just slicing the x and y columns.
        for i in range(1, data_likelihood.shape[0] - 1):
            if data_likelihood_col_i[i] < likelihood_threshold:
                array_data_filtered[i, (2 * idx_col_i):(2 * idx_col_i + 2)] = array_data_filtered[i - 1,
                                                                              (2 * idx_col_i):(2 * idx_col_i + 2)]
            else:
                array_data_filtered[i, (2 * idx_col_i):(2 * idx_col_i + 2)] = np.hstack(
                    [data_x[i, idx_col_i], data_y[i, idx_col_i]])

    # ### Adaptive filtering is all done. Clean up and return.
    # # Remove first row in data array (values are all zeroes)
    # array_filtered_data_without_first_row = np.array(array_data_filtered[1:]).astype(np.float)
    array_filtered_data_without_first_row = np.array(array_data_filtered[1:]).astype(np.float)

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

    return df_adaptively_filtered_data, percent_filtered_per_bodypart


def filter_dlc_output(in_df: pd.DataFrame, likelihood_threshold=0.9, copy=False) -> pd.DataFrame:
    """
    This function is a copy of above mentioned adaptive_filter function where we perform linear interpolation instead
    of forward fill. # TODO: This function can be made much more simpler if certain assumptions about input are made.
    :param in_df: (DataFrame) expected: raw DataFrame of DLC results right after reading in using dibs.read_csv().
    :param likelihood_threshold:
    :param copy:

    :returns df_filtered
    """
    # TODO: HIGH: for this function that does not have expected cols (like 'scorer', etc.) it should not fail!
    # Checking args
    check_arg.ensure_type(in_df, pd.DataFrame)
    # Continue
    # # Scorer
    set_in_df_columns = set(in_df.columns)
    if 'scorer' not in set_in_df_columns:
        col_not_found_err = f'Column named "scorer" not found but should exist (as a result from dibs.read_csv()) // ' \
                            f'All columns: {in_df.columns}'
        logger.error(col_not_found_err)
        raise ValueError(col_not_found_err)  # TODO: should this raise an error?
    scorer_values = np.unique(in_df['scorer'].values)
    if len(scorer_values) != 1:
        err = f'There should be 1 unique scorer value. If there are more than 1, too many values. Value are: {scorer_values}.'
        logger.error(err)
        raise ValueError(err)  # TODO: low: should this raise an error?
    scorer_value: str = scorer_values[0]

    # # Source
    if 'source' in set_in_df_columns:
        if len(scorer_values) != 1:
            err = f'There should be 1 unique "source" value. If there is more than 1, too many values, ' \
                  f'makes no sense to adaptively filter over different datasets.'
            logger.error(err)
            raise ValueError(err)  # TODO: low: should this raise an error?
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
    x_index, y_index, l_index = [], [], []
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
        elif column_suffix == 'coords':  # todo: low: delete this elif. Coords should be dropped with the io.read_csv implementation?
            # Record and check later...likely shouldn't exist anymore since its just a numbered col with no data.
            coords_cols_names.append(col)
        elif col == 'scorer':
            pass  # Ignore 'scorer' column. It tracks the DLC data source.
        elif col == 'source':
            pass  # Keeps track of CSV/h5 source
        elif col == 'frame':
            pass  # Keeps track of frame numbers
        elif col == 'file_source':
            pass
        elif col == 'data_source':
            pass
        else:
            err = f'{inspect.stack()[0][3]}(): An inappropriate column header was found: ' \
                  f'{column_suffix}. Column name = "{col}". ' \
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

    array_data_filtered = np.zeros((data_x.shape[0], (data_x.shape[1]) * 3))

    percent_filtered_per_bodypart: List = [0. for _ in range(data_likelihood.shape[1])]

    # Loop over data and do adaptive filtering.
    for idx_col_i in tqdm(range(data_likelihood.shape[1]),
                          desc=f'{logging_enhanced.get_current_function()}(): Filtering DLC columns using interpolation',
                          disable=True if config.stdout_log_level.strip().upper() != 'DEBUG' else False):

        # Change data type to float because its currently string. Is it though?
        data_likelihood_col_i = data_likelihood[:, idx_col_i].astype(np.float)

        # Record percent filtered (for "reasons")
        percent_filtered_per_bodypart[idx_col_i] = np.sum(data_likelihood_col_i < likelihood_threshold) / data_likelihood.shape[0]

        nan_mask = data_likelihood_col_i < likelihood_threshold
        x, y = pd.DataFrame(data_x[:, idx_col_i].astype(np.float)), pd.DataFrame(data_y[:, idx_col_i].astype(np.float))
        x[nan_mask] = np.nan
        y[nan_mask] = np.nan
        array_data_filtered[:, 3*idx_col_i] = x.interpolate(method='linear', limit=500, limit_direction='both').values.reshape(x.shape[0])
        array_data_filtered[:, 3*idx_col_i + 1] = y.interpolate(method='linear', limit=500, limit_direction='both').values.reshape(x.shape[0])
        array_data_filtered[:, 3*idx_col_i + 2] = data_likelihood_col_i

    # Create DataFrame with columns by looping over x/y indices.
    columns_ordered: List[str] = []
    for x_idx, y_idx, l_idx in zip(x_index, y_index, l_index):
        columns_ordered += [map_back_index_to_col_name[x_idx], map_back_index_to_col_name[y_idx], map_back_index_to_col_name[l_idx]]

    # Create frame, replace 'scorer' column. Return.
    df_filtered = pd.DataFrame(array_data_filtered, columns=columns_ordered)
    df_filtered['scorer'] = scorer_value
    # Re-add source, etc
    if source is not None:
        df_filtered['source'] = source
    if file_source is not None:
        df_filtered['file_source'] = file_source
    if data_source is not None:
        df_filtered['data_source'] = data_source

    df_filtered['frame'] = range(len(df_filtered))
    if len(in_df) != len(df_filtered):
        missing_rows_err = f'Input df has {len(df)} rows but output df ' \
                           f'has {len(df_filtered)}. Should be same number.'
        logger.error(missing_rows_err)
        raise ValueError(df_filtered)

    return df_filtered


if __name__ == '__main__':
    pass

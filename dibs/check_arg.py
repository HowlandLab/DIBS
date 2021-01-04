"""
Encapsulate all argument checking functions here.
Since the patterns we use to log and raise exceptions are so frequent and
    consistent, we abstract away and implement and store them here.
"""
from typing import Collection, Optional
import errno
import inspect
import numpy as np
import os
import pandas as pd
import sys

from dibs.logging_dibs import get_caller_function
import dibs

logger = dibs.config.initialize_logger(__file__)


###

def ensure_type(var, *expected_types):
    """
    Checks that input variable `var` matches at least 1 of the expected types.
    If it does match an expected type, function returns immediately without problem.
    If the variables does not match any of the expected types, then a TypeError is raised.
    """
    for t in expected_types:
        if isinstance(var, t):
            return
    type_err = f'Type-checking caller: {get_caller_function()}(): For object (value = {var}), ' \
               f'expected type(s) was {expected_types} but instead found {type(var)}'
    logger.error(type_err)
    raise TypeError(type_err)


def ensure_collection_not_empty(collection):
    """"""
    if len(collection) == 0:
        err = f'Caller: {get_caller_function()}(): Input variable was expected ' \
              f'to be non-empty but was in fact empty. Value = {collection}.'
        logger.error(err)
        raise ValueError(err)


def ensure_columns_in_DataFrame(df: pd.DataFrame, columns: Collection[str]) -> None:
    ensure_type(df, pd.DataFrame)
    set_df_columns = set(df.columns)
    for col in columns:
        if col not in set_df_columns:
            err = f'Caller: {get_caller_function()}(): column named `{col}` was expected to be in DataFrame ' \
                  f'columns but was not found. Actual columns found: {df.columns}.'
            logger.error(err)
            raise ValueError(err)


def ensure_has_valid_chars_for_path(path):
    if has_invalid_chars_in_name_for_a_file(path):
        err = f'Caller: {get_caller_function()}(): Path name has invalid characters. Path = "{path}"'
        logger.error(err)
        raise ValueError(err)


def ensure_is_file(path):
    if not os.path.isfile(path):
        err = f"Caller: {get_caller_function()}(): file path '{path}' was expected to be a file path but was not."
        logger.error(err)
        raise FileNotFoundError(err)


def ensure_is_dir(path):
    if not os.path.isdir(path):
        err = f'Caller: {get_caller_function()}(): submitted file path "{path}" was ' \
              f'expected to be a valid directory but was not.'
        logger.error(err)
        raise NotADirectoryError(err)


def ensure_is_valid_path(path):
    if not is_pathname_valid(path):  # TODO:
        err = f'TODO: elaborate -- has INVALID chars in proposed path: {path}'
        logger.error(err)
        raise ValueError(err)


def ensure_frame_indices_are_integers(df: pd.DataFrame, frame_column_name='frame'):
    df_non_int_frame_indices = df.loc[df[frame_column_name].map(lambda x: x % 1 != 0)]
    num_of_non_integer_frame_indices = len(df_non_int_frame_indices)
    if num_of_non_integer_frame_indices != 0:
        err = f'Found non-integer frame indices which should never happen. Check the ' \
              f'following DataFrame data: {df_non_int_frame_indices.to_string()} ' \
              f'(dtypes={df_non_int_frame_indices.dtypes})'
        logger.error(err)
        raise ValueError(err)


def ensure_numpy_arrays_are_same_shape(*arrays):
    for a in arrays:
        ensure_type(a, np.ndarray)
    # Ensure that at least one array is put in
    if len(arrays) < 1:
        err = f'Caller = "{get_caller_function()}()". Error: 0 arrays were submitted in checkarg.thisfunction'
        logger.error(err)
        raise ValueError(err)
    # Execute
    set_of_shapes = set([arr.shape for arr in arrays])
    if len(set_of_shapes) != 1:
        err = f'Caller = "{get_caller_function()}()". Error: One of the following ' \
              f'arrays does not have the same shape. Set of shapes = "{set_of_shapes}". '
        for arr in arrays:
            err += f'Array = "{arr}" (shape = "{arr.shape}"). '.replace('\n', ' ')
        logger.error(err)
        raise ValueError(err)


###

def has_invalid_chars_in_name_for_a_file(file_name, additional_characters: Collection[str] = None) -> bool:
    """
    Checks if an invalid characters have been included in a potential path. Useful for checking user
    input before attempting to save files. The list of invalid characters
    :param file_name: (str)
    :param additional_characters: (collection of strings)
    :return:
    """
    if additional_characters is not None:
        ensure_type(additional_characters, list, tuple, set)
    # if additional_characters is not None \
    #         and not isinstance(additional_characters, list) \
    #         and not isinstance(additional_characters, tuple) \
    #         and not isinstance(additional_characters, set):
    #     invalid_type_err = f'{inspect.stack()[0][3]}(): Invalid type ' \
    #                    f'found: {type(additional_characters)} (value: {additional_characters})'
    #     logger.error(invalid_type_err)
    #     raise TypeError(invalid_type_err)

    invalid_chars_for_windows_files = {':', '*', '\\', '/', '?', '"', '<', '>', '|'}
    if additional_characters is not None:
        invalid_chars_for_windows_files = invalid_chars_for_windows_files.union(set(additional_characters))

    if not isinstance(file_name, str) or not file_name:
        return True

    union_of_string_and_invalid_chars = set(file_name).intersection(invalid_chars_for_windows_files)
    if len(union_of_string_and_invalid_chars) != 0:
        logger.error(f'Union = {union_of_string_and_invalid_chars}')
        return True

    return False


def is_pathname_valid(pathname: str) -> bool:
    """ Checks if the path name is valid. Useful for checking user inputs.
    Source: https://stackoverflow.com/a/34102855
    Returns: (bool) `True` if the passed pathname is a valid pathname for the current OS;
                    `False` otherwise.
    """
    ERROR_INVALID_NAME = 123  # PLO: Necessary for valid file name checking.
    # If this pathname is either not a string or is but is empty, this pathname is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this path name's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?

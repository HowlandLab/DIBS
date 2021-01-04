"""
TODO: explain submodule
"""
from typing import Any, List, Tuple, Union
import functools
import numpy as np
import pandas as pd
import re

from dibs import config

logger = config.initialize_logger(__file__)


def mean(*args):
    """
    Get the mean average for all arguments for those that are not NaN.
    Provides a solution that gets the average for any N arguments.
    """
    args = [arg for arg in args if arg == arg]  # Remove any 'nan' values
    if len(args) == 0:
        return float('nan')
    return functools.reduce(lambda x, y: x + y, args, 0) / len(args)


def sum_args(*args):
    """
    Get the sum of all arguments for those that are not NaN.
    Provides a solution that gets the sum for any N arguments.
    """
    args = [arg for arg in args if arg == arg]  # Remove any 'nan' values
    if len(args) == 0:
        return float('nan')
    return functools.reduce(lambda x, y: x + y, args, 0)


def first_arg(*args):
    if len(args) <= 0:
        err = f'An invalid set of args submitted. Args = {args}'
        logger.error(err)
        raise ValueError(err)
    return args[0]


def convert_int_from_string_if_possible(s: str):
    """ Converts digit string to integer if possible """
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s) -> List:
    """
    Turn a string into a list of string and number chunks.
        e.g.: input: "z23a" -> output: ["z", 23, "a"]
    """
    return [convert_int_from_string_if_possible(c) for c in re.split('([0-9]+)', s)]


def sort_list_nicely_in_place(list_input: list) -> None:
    """ Sort the given list (in place) in the way that humans expect. """
    if not isinstance(list_input, list):
        raise TypeError(f'argument `l` expected to be of type list but '
                        f'instead found: {type(list_input)} (value: {list_input}).')
    list_input.sort(key=alphanum_key)


def augmented_runlength_encoding(labels: Union[List, np.ndarray]) -> Tuple[List[Any], List[int], List[int]]:
    """
    TODO: med: purpose // purpose unclear
    :param labels: (list or np.ndarray) predicted labels
    :return
        label_list: (list) the label number
        idx: (list) label start index
        lengths: (list) how long each bout lasted for
    """
    label_list, idx_list, lengths_list = [], [], []
    i = 0
    while i < len(labels):
        # 1/3: Record current index
        idx_list.append(i)
        # 2/3: Record current label
        current_label = labels[i]
        label_list.append(current_label)
        # Iterate over i while current label and next label are same
        start_index = i
        while i < len(labels)-1 and labels[i] == labels[i + 1]:
            i += 1
        end_index = i
        # 3/3: Calculate length of repetitions, then record lengths to list
        length = end_index - start_index
        lengths_list.append(length)
        # Increment and continue
        i += 1
    return label_list, idx_list, lengths_list






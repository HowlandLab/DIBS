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






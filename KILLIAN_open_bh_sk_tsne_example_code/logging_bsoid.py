"""
Encapsulate logging for BSOID
"""

from logging.handlers import SMTPHandler  # PLO: to be used potentially later in logging errors to email
import inspect
import logging
import re
import time


# Logger object creation

def preload_logger_with_config_vars(logger_name: str, log_format: str,
                                    stdout_log_level: str = None, file_log_level: str = None,
                                    file_log_file_path: str = None, email_log_level: str = None):
    """
    Create a meta logger. On first call, load up main settings for logger.
    Returns a callable. That returned callable can have a file-specific name loaded
        in so that the name in the logs reflects the file the logger was instantiated inside.
    """
    def argument_loaded_function(current_python_file_name: str, log_format: str = log_format):
        """Load in stored config args"""
        # Check if the name value is in the format string. If possible, switch out the default
        #   logger name for the new filename
        name_regex_result = re.search(r'%\(name\)-?\w*s', log_format)
        if name_regex_result:
            log_format = log_format[:name_regex_result.start()] + \
                         current_python_file_name.ljust(58) + \
                         log_format[name_regex_result.end():]

        final_result = create_generic_logger(
            logger_name=logger_name,
            log_format=log_format,
            stdout_log_level=stdout_log_level,
            file_log_level=file_log_level,
            file_log_file_path=file_log_file_path,
            email_log_level=email_log_level,
        )
        return final_result
    return argument_loaded_function


def create_generic_logger(logger_name: str, log_format: str,
                          stdout_log_level: str = None,
                          file_log_level: str = None, file_log_file_path: str = None,
                          email_log_level: str = None) -> logging.Logger:
    """
    Generic logger instantiation.

    :param logger_name: (str, required) The name of the logger. If referenced in the formatter, this
        name will be used in the logs.
    :param log_format: (str, required): TOO
    :param stdout_log_level: (str, ?) TODO
    :param file_log_level: (str, ?) TODO
    :param file_log_file_path: (str, ?) TODO
    :param email_log_level: TODO:
    Returns logger object
    """
    valid_log_levels = {'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'}
    formatter = logging.Formatter(log_format)
    logger = logging.Logger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG in order to allow ALL messages to be sent to handlers

    # Logging to console
    if stdout_log_level is not None:
        # Check type
        if not isinstance(stdout_log_level, str):
            raise TypeError(f'`stdout_log_level` was expected to be str but found {type(stdout_log_level)}')
        # Check valid log level
        stdout_log_level = stdout_log_level.upper()
        if stdout_log_level not in valid_log_levels:
            raise ValueError(f'Invalid log level submitted for `std_log_level` (value: {stdout_log_level})')
        # Continue if no errors
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stdout_log_level.upper())
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Logging to file
    if file_log_level is not None:
        # Check file level type and value validity
        if not isinstance(file_log_level, str):
            raise TypeError(f'`file_log_level` was expected to be str but instead '
                            f'found: {type(file_log_level)} (value: {file_log_level}).')
        # Check file log level
        file_log_level = file_log_level.upper()
        if file_log_level not in valid_log_levels:
            raise ValueError(f'')
        # Check filepath
        if not isinstance(file_log_file_path, str):
            raise TypeError('argument `file_log_path` was expected to be type str but instead found: '
                            f'{type(file_log_file_path)} (value: {file_log_file_path}).')
        # Check that file path exists up to the folder level
        ## TODO: low

        # Continue if no errors
        file_handler = logging.FileHandler(file_log_file_path)
        file_handler.setLevel(file_log_level.upper())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Logging to email  # TODO: low: instantiate email logging
    if email_log_level:
        ## SMTP vars: mailhost, fromaddr, toaddrs, subject, credentials=None, secure=None, timeout=5.0
        # smtp_handler = SMTPHandler()
        # smtp_handler.setLevel(email_log_level.upper())
        # smtp_handler.setFormatter(formatter)
        # logger.addHandler(smtp_handler)
        pass
    return logger


def log_entry_exit(logger=None):
    """
    Meta decorator. If used as deco., then must have argument of logger object for it to be useful.
    """
    def decorator(func):
        def function_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            if logger:
                logger.debug(f'Now entering: {func.__qualname__}().')
            result = func(*args, *kwargs)
            end_time = time.perf_counter()
            if logger:
                logger.debug(f'Now exiting: {func.__qualname__}(). Time spent in '
                             f'function: {round(end_time-start_time, 1)} seconds')
            return result
        return function_wrapper
    return decorator


# Accessory functions

def get_current_function() -> str:
    """
    Get the string name of the current function in which this function is called.
    """
    return inspect.stack()[1][3]


def get_caller_function() -> str:
    """
    Get the string name of the function which calls the current function the interpreter is in.
    """
    return inspect.stack()[2][3]


# Example functions (not actively in use but useful to have)

def decorator_example(func):
    # Example functions to show how to make a decorator
    def function_wrapper(*args, **kwargs):
        print(f'args: {args} / kwargs: {kwargs}')
        result = func(*args, **kwargs)
        print(f'result: {result}')
        return result

    return function_wrapper


def log_function_decorator(decorator_arg=None):
    """ An example of a decorator that takes an optional arg """
    def decorator(func):
        def function_wrapper(*args, **kwargs):
            result = create_generic_logger(*args, **kwargs)
            return result
        return function_wrapper
    return decorator


if __name__ == '__main__':
    pass

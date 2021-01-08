"""
Every function in this file is an entire runtime sequence (app) encapsulated. Expect nothing to be returned.
"""
from typing import List
import inspect
import os
import time

from dibs import config, logging_dibs, streamlit_app

logger = config.initialize_logger(__name__)


# Command-line functions

def clear_output_folders(*args, **kwargs) -> None:
    """
    For each folder specified below (magic variables be damned),
        delete everything in that folder except for the .placeholder file and any sub-folders there-in.
    """
    raise NotImplementedError(f'TODO: review this function since folder layouts have changed')  # TODO: review this function since folder layouts have changed
    # Choose folders to clear (currently set as magic variables in function below)
    folders_to_clear: List[str] = [config.OUTPUT_PATH,
                                   config.GRAPH_OUTPUT_PATH,
                                   config.VIDEO_OUTPUT_FOLDER_PATH,
                                   config.FRAMES_OUTPUT_PATH, ]
    # Loop over all folders to empty
    for folder_path in folders_to_clear:
        # Parse all files in current folder_path, but exclusive placeholders, folders
        valid_files_to_delete = [file_name for file_name in os.listdir(folder_path)
                                 if file_name != '.placeholder'
                                 and not os.path.isdir(os.path.join(folder_path, file_name))]
        # Loop over remaining files (within current folder iter) that are to be deleted next
        for file in valid_files_to_delete:
            file_to_delete_full_path = os.path.join(folder_path, file)
            try:
                os.remove(file_to_delete_full_path)
                logger.debug(f'{inspect.stack()[0][3]}(): Deleted file: {file_to_delete_full_path}')
            except PermissionError as pe:
                logger.warning(f'{inspect.stack()[0][3]}(): Could not delete file: {file_to_delete_full_path} / '
                               f'{repr(pe)}')
            except Exception as e:
                unusual_err = f'An unusual error was detected: {repr(e)}'
                logger.error(unusual_err)
                raise e

    return None


def streamlit(**kwargs) -> None:
    """ Streamlit code here. Currently this is the first and only iteration of streamlit apps, but
    who knows how many will be created in the future. """
    streamlit_app.home(**kwargs)


# Sample function

def sample_runtime_function(sleep_secs=3, *args, **kwargs):
    """ Sample function that takes n seconds to run """
    logger.debug(f'{logging_dibs.get_current_function()}(): '
                 f'Doing sample runtime execution for {sleep_secs} seconds.')
    time.sleep(sleep_secs)
    return




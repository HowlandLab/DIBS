#!/usr/bin/env python3
"""
Command-line interface for B-SOiD

TODO: Commands to implement:
    - clear logs
    - clear output folder (+ prompt for confirm)
"""
from typing import Union
import argparse
import streamlit as st  # Do not remove this

import dibs
from dibs.pipeline import *  # This line is required for Streamlit to load Pipeline objects. Do not delete. For a more robust solution, consider: https://rebeccabilbro.github.io/module-main-has-no-attribute/

logger = dibs.config.initialize_logger(__name__)


########################################################################################################################

dibs_runtime_description = 'DIBS command line utility. Do DIBS stuff. Expand on this later.'

map_command_to_func = {
    'bitcheck': dibs.app.print_if_system_is_64_bit,
    'buildone': dibs.app.buildone,
    'checktiming': dibs.streamlit_app.checking_file_session_timings,
    'gridsearch': dibs.app.tsnegridsearch,
    'streamlit': dibs.app.streamlit,
    'test': lambda *args, **kwargs: print(args, kwargs),
    'trybuild': dibs.app.trybuild,
    # 'buildandrunlegacy': dibs.main_LEGACY.test_function_to_build_then_run_py,
    # 'clean': dibs.app.clear_output_folders,  # TODO: review clear output folders function for
    # 'cleanoutput': dibs.app.clear_output_folders,
    # 'newbuild': dibs.app.build_classifier_new_pipeline,
    'sample': dibs.app.sample,
}


########################################################################################################################

def parse_args() -> argparse.Namespace:
    """
    Instantiate arguments that will be parsed from the command-line here.
    Regarding HOW these commands will be carried out, implement that elsewhere.
    """
    # Instantiate parser, add arguments as expected on command-line
    parser = argparse.ArgumentParser(description=dibs_runtime_description)
    parser.add_argument(f'command', help=f'HELP: TODO: command. Valid commands: '
                                         f'{[""+x for x in list(map_command_to_func.keys())]}')  # TODO: easy: does it need to coerce into a list at all?
    parser.add_argument('-p', name='pipeline_path', help=f'HELP: TODO: PIPELINE LOC', default=None)
    parser.add_argument('-n', help=f'Name of pipeline', default="OneTwoThree")
    # TODO: add commands, sub-commands as necessary

    # Parse args, return
    args: argparse.Namespace = parser.parse_args()

    # TODO: uncomment below later

    logger.debug(f'ARGS: {args}')
    logger.debug(f'args.command = {args.command}')
    # logger.debug(f'args.p = {args.p}')

    return args


def parse_args_return_kwargs() -> dict:
    # Instantiate parser, add arguments as expected on command-line
    parser = argparse.ArgumentParser(description=dibs_runtime_description)
    parser.add_argument(f'command', help=f'HELP: TODO: command. Valid commands: '
                                         f'{[""+x for x in list(map_command_to_func.keys())]}')  # TODO: easy: does it need to coerce into a list at all?

    parser.add_argument('-n', '--name', help=f'Name of pipeline', default=None)
    parser.add_argument('-p', '--pipeline_path', help=f'Path to an existing Pipeline', default=None)

    # Parse args, return
    args: argparse.Namespace = parser.parse_args()

    logger.debug(f'ARGS: {args}')
    logger.debug(f'args.command = {args.command}')

    kwargs = {
        'command': args.command,
        'pipeline_path': args.pipeline_path,
        'name': args.name,
    }

    return kwargs


def execute_command(**kwargs) -> None:
    try:
        # Execute command
        map_command_to_func[kwargs['command']](**kwargs)
    except BaseException as e:
        err_text = f'An exception occurred during runtime. See the following exception: {repr(e)}'
        logger.error(err_text, exc_info=True)
        raise e


### Main execution #####################################################################################################

def main():
    ### Parse args
    args: dict = parse_args_return_kwargs()
    # logger.debug(f'args: {args}')
    # logger.debug(f'args.command: {args.command}')
    ### Do command
    execute_command(**args)


if __name__ == '__main__':
    main()

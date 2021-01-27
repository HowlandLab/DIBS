#!/usr/bin/env python3
"""
Command-line interface for B-SOiD

TODO: Commands to implement:
    - clear logs
    - clear output folder (+ prompt for confirm)
"""

import argparse

import dibs
from dibs.pipeline import *  # This line is required for Streamlit to load Pipeline objects. Do not delete. For a more robust solution, consider: https://rebeccabilbro.github.io/module-main-has-no-attribute/

logger = dibs.config.initialize_logger(__name__)


########################################################################################################################

dibs_runtime_description = 'DIBS command line utility. Do DIBS stuff. Expand on this later.'

map_command_to_func = {

    # 'clean': dibs.app.clear_output_folders,  # TODO: review clear output folders function for
    # 'cleanoutput': dibs.app.clear_output_folders,
    # 'buildandrunlegacy': dibs.main_LEGACY.test_function_to_build_then_run_py,
    # 'newbuild': dibs.app.build_classifier_new_pipeline,
    'streamlit': dibs.streamlit_app.start_app,
    'test': lambda *args, **kwargs: print(args, kwargs),
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
    parser.add_argument('-p', help=f'HELP: TODO: PIPELINE LOC')
    # TODO: add commands, sub-commands as necessary

    # Parse args, return
    args: argparse.Namespace = parser.parse_args()

    # TODO: uncomment below later
    logger.debug(f'ARGS: {args}')
    logger.debug(f'args.command = {args.command}')
    logger.debug(f'args.p = {args.p}')

    return args


def execute_command(args: argparse.Namespace) -> None:
    kwargs = {}

    if args.p:
        kwargs['pipeline_path'] = args.p

        logger.debug(f'main.py: arg.p parsed as: {args.p}')

    return map_command_to_func[args.command](**kwargs)


### Main execution #####################################################################################################

def main():
    ### Parse args
    args = parse_args()

    print(f'args: {args}')
    print(f'args.command: {args.command}')

    ### Do command
    execute_command(args)

    ### End
    pass


if __name__ == '__main__':
    main()

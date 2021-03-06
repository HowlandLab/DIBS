"""

streamlit api: https://docs.streamlit.io/en/stable/api.html
Number formatting: https://python-reference.readthedocs.io/en/latest/docs/str/formatting.html
    Valid formatters: %d %e %f %g %i
More on formatting: https://pyformat.info/

Developer's note: ignore 120 char limit during development
"""
from matplotlib import pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits.mplot3d import Axes3D  # Despite being "unused", this import MUST stay in for 3d plotting to work.
from traceback import format_exc as get_traceback_string
from typing import Dict, List

import matplotlib
import numpy as np
import os
import pandas as pd
import random
import re
import seaborn as sns
import streamlit as st
import sys
import time
import traceback
# import easygui
# import plotly
# import tkinter as tk
# from tkinter import filedialog
# from mttkinter import mtTkinter as tk

from dibs import check_arg, config, io, logging_enhanced, pipeline, streamlit_session_state, base_pipeline, pipeline_pieces
from dibs.io import clean_string, save_to_folder


# Module settings
logger = config.initialize_logger(__name__)
matplotlib_axes_logger.setLevel('ERROR')
raw, engineered, scaled = 'Raw', 'Engineered', 'Classifier-fed'

##### Instantiate names for buttons, options that can be changed on the fly but logic below stays the same #####
webpage_head_title = 'DIBS'
valid_video_extensions = {'avi', 'mp4', }
# Variables for buttons, drop-down menus, and other things
start_new_project_option_text, load_existing_project_option_text = 'Create new', 'Load existing'
text_bare_pipeline, text_dibs_data_pipeline = 'Create bare pipeline', 'Create a pipeline with pre-loaded training and test data (according to DIBS specs)'
text_half_dibs_data_pipeline = 'DEBUG OPTION: Create a pipeline with HALF of the pre-loaded training and test data (according to DIBS specs). Useful for new users to reduce training times for TSNE param optimization'  # TODO: med: temporary option. Delete on release.
pipeline_options = {  # Key=Pipeline name + optional description. Value=Pipeline class
    # 'PipelineCHBO: the Change Blindness Odor Test pipeline': pipeline.PipelineCHBO,
    # 'PipelineEPM: Elevated Plus Maze': pipeline.PipelineEPM,
    'PipelineHowland: a pipeline aimed at generalized behaviour recognition': pipeline.PipelineHowland,
    # 'PipelinePrime': pipeline.PipelinePrime,
    'PipelineMimic: a pipeline that mimics the B-SOiD implementation for Open Field': pipeline.PipelineMimic,
    # 'PipelineTim: A novel feature set attempt at behaviour segmentation': pipeline.PipelineTim,
    'PipelineKitchenSink: a debug pipeline for too many features': pipeline.PipelineKitchenSink,
    # 'PipelineIBNS'
    'PipelineIBNS: CHBO object exploration pipeline': pipeline.PipelineIBNS_first_one,
    'PipelineISBN_second_with_time_shifting': pipeline.PipelineIBNS_second_with_time_shifting,
}
valid_layouts = {'centered', 'wide'}
training_data_option, predict_data_option = 'Training Data', 'Predict Data'
key_iteration_page_refresh_count = 'key_iteration_page_refresh_count'  # A debugging effort. Remove when done debugging.
checkbox_show_extra_text = 'Show additional descriptive text'  # TODO: low: update text here
# Set keys for objects (mostly buttons) for streamlit components that need some form of persistence.
key_selected_layout, initial_sidebar_state = 'key_selected_layout', 'initial_sidebar_state'
key_pipeline_path = 'key_pipeline_path'  # <- for saving pipe path when loaded??? ????
key_open_pipeline_path = 'key_open_pipeline_path'  # For selecting a pipeline using hack dialog box
default_n_seconds_sleep = 'default_n_seconds_wait_until_auto_refresh'
key_button_change_feature_engineering = 'key_button_change_feature_engineering'
key_button_show_adv_pipeline_information = 'key_button_show_more_pipeline_information'
key_button_see_rebuild_options = 'key_button_see_model_options'
key_button_see_advanced_options = 'key_button_see_advanced_options'
key_button_change_info = 'key_button_change_info'
key_button_rebuild_model = 'key_button_rebuild_model'
key_button_rebuild_model_confirmation = 'key_button_rebuild_model_confirmation'
key_button_add_new_data = 'key_button_add_new_data'
key_button_menu_remove_data = 'key_button_menu_remove_data'
key_button_update_description = 'key_button_update_description'
key_button_add_train_data_source = 'key_button_add_train_data_source'
key_button_add_predict_data_source = 'key_button_add_predict_data_source'
key_button_view_confusion_matrix = 'key_button_view_confusion_matrix'
key_button_review_assignments = 'key_button_update_assignments'
key_button_view_assignments_distribution = 'key_button_view_assignments_distribution'
key_button_view_assignments_clusters = 'key_button_view_assignments_clusters'
key_button_save_assignment = 'key_button_save_assignment'
key_button_show_example_videos_options = 'key_button_show_example_videos_options'
key_button_create_new_example_videos = 'key_button_create_new_example_videos'
key_button_menu_label_entire_video = 'key_button_menu_label_entire_video'
key_button_export_training_data = 'key_button_export_training_data'
# key_button_export_train_data_raw = 'key_button_export_data_raw'
key_button_export_train_data_engineered = 'key_button_export_data_engineered'
key_button_export_train_data_engineered_scaled = 'key_button_export_data_engineered_scaled'
key_button_export_predict_data = 'key_button_export_predict_data'
key_button_export_predict_data_raw = 'key_button_export_data_raw'
key_button_export_predict_data_engineered = 'key_button_export_data_engineered'
key_button_export_predict_data_engineered_scaled = 'key_button_export_data_engineered_scaled'
### Page variables data & default values ###
streamlit_persistence_variables = {  # Instantiate default variable values here.
    key_selected_layout: 'centered',  # Valid values include: "centered" and "wide"
    initial_sidebar_state: 'collapsed',  # Valid values include: "expanded" and "collapsed"
    key_pipeline_path: '',  #  TODO: med: review usage. Could be strong than passing paths between funcs?
    key_open_pipeline_path: config.DIBS_BASE_PROJECT_PATH,
    key_iteration_page_refresh_count: 0,
    default_n_seconds_sleep: 3,
    checkbox_show_extra_text: False,
    key_button_change_feature_engineering: False,
    key_button_show_adv_pipeline_information: False,
    key_button_see_rebuild_options: False,
    key_button_see_advanced_options: False,
    key_button_change_info: False,
    key_button_rebuild_model: False,
    key_button_rebuild_model_confirmation: False,
    key_button_add_new_data: False,
    key_button_add_train_data_source: False,
    key_button_add_predict_data_source: False,
    key_button_menu_remove_data: False,
    key_button_update_description: False,
    key_button_view_confusion_matrix: False,
    key_button_review_assignments: False,
    key_button_view_assignments_distribution: False,
    key_button_view_assignments_clusters: False,
    key_button_save_assignment: False,
    key_button_show_example_videos_options: False,
    key_button_create_new_example_videos: False,
    key_button_menu_label_entire_video: False,
    key_button_export_training_data: False,
    # key_button_export_train_data_raw: False,
    key_button_export_train_data_engineered: False,
    key_button_export_train_data_engineered_scaled: False,
    key_button_export_predict_data: False,
    key_button_export_predict_data_raw: False,
    key_button_export_predict_data_engineered: False,
    key_button_export_predict_data_engineered_scaled: False,
}
# TODO: propagate file path thru session var?
session = {}

##### Page layout #####


def start_app(**kwargs):
    """
    The designated home page/entry point when Streamlit is used.
    -------------
    kwargs:

        pipeline_path : str
        A path to an existing pipeline file which will be loaded by default
        on page load. If this kwarg is not specified, the config.ini
        value will be checked (via dibs.config), and that file path, if
        present, will be used. If that config.ini key/value pair is not in
        use, then no default path will be specified and it will be entirely
        up to the user to fill out.

    """
    ### Set up session variables
    global session
    session = streamlit_session_state.get(**streamlit_persistence_variables)
    # matplotlib.use('TkAgg')  # For allowing graphs to pop out as separate windows. Dev note: Streamlit does not play nice with multithreaded instances like the Tkinter window pop-up. Sometimes crashes Streamlit.
    matplotlib.use('Agg')  # For allowing graphs to render onto page
    session[key_iteration_page_refresh_count] = session[key_iteration_page_refresh_count] + 1

    # Set page config
    st.set_page_config(page_title=webpage_head_title,
                       page_icon=':shark:',  # TODO: low: settle on an emoji later
                       layout=session[key_selected_layout],
                       initial_sidebar_state='collapsed')

    # Load up pipeline if specified on command line or specified in config.ini
    pipeline_file_path: str = kwargs.get('pipeline_path', '')
    if pipeline_file_path:
        check_arg.ensure_is_file(pipeline_file_path)
    else:
        # If not specified on command line, check config.ini path and use as default if possible
        if config.default_pipeline_file_path and os.path.isfile(config.default_pipeline_file_path):
            pipeline_file_path = config.default_pipeline_file_path
        else:
            pipeline_file_path = session[key_pipeline_path]

    ### SIDEBAR ###
    # st.sidebar.markdown(f'## App Settings')
    # st.sidebar.markdown(f'----------------')
    # st.sidebar.markdown(f'### Iteration: {session[key_iteration_page_refresh_count]}')  # Debugging effort
    # st.sidebar.markdown('------')
    is_checked = st.sidebar.checkbox(checkbox_show_extra_text, value=True)  # TODO: low: remove value=True after debugging is done.
    session[checkbox_show_extra_text] = is_checked

    selected_layout = st.sidebar.selectbox('Select a page layout', options=[session[key_selected_layout].title()] + [x.title() for x in ["Wide", "Centered"] if x != session[key_selected_layout].title()])
    if selected_layout.lower() != session[key_selected_layout]:
        session[key_selected_layout] = selected_layout.lower()
        st.experimental_rerun()

    ### MAIN ###
    header(pipeline_file_path)


def header(pipeline_file_path):
    """
    Start of the streamlit app page for drawing
    """
    # # TODO: review use of columns
    # col1, col2, col3 = st.beta_columns(3)
    # with col1:
    #     st.header("A cat")
    #     st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)
    # with col2:
    #     st.header("A dog")
    #     st.image("https://static.streamlit.io/examples/dog.jpg", use_column_width=True)
    # with col3:
    #     st.header("An owl")
    #     st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)
    # # TODO: review use of expanders
    # st.line_chart({"data": [1, 5, 2, 6, 2, 1]})
    # with st.beta_expander("See explanation"):
    #     st.write("""
    #         The chart above shows some numbers I picked for you.
    #         I rolled actual dice for these, so they're *guaranteed* to
    #         be random.""")
    #     st.image("https://static.streamlit.io/examples/dice.jpg")

    ######################################################################################
    logger.debug('    < Start of Streamlit page >    ')
    ### SIDEBAR ###

    # st.sidebar.markdown(f'Checkbox checked: {is_checked}')

    ### MAIN ###
    st.markdown(f'# DIBS Streamlit app')

    st.markdown('------------------------------------------------------------------------------------------')
    is_pipeline_loaded = False

    ## Start/open project using drop-down menu ##
    start_select_option = st.selectbox(
        label='Start a new project or load an existing one?',
        options=('', start_new_project_option_text, load_existing_project_option_text),
        key='StartProjectSelectBox',
        index=2 if os.path.isfile(pipeline_file_path) else 0
    )
    st.markdown('')
    try:
        # Option 1/2: Start new project
        if start_select_option == start_new_project_option_text:
            st.markdown(f'## Create new project pipeline')
            st.markdown('')

            # Input parameters for new Pipeline
            radio_select_create_method = st.radio('Would you like to pre-load data?', [text_bare_pipeline, text_dibs_data_pipeline, text_half_dibs_data_pipeline])
            st.markdown('*Note: the above decision is not final. Data can be added and removed at any time in a project.*')
            # st.markdown('')
            text_input_new_project_name = st.text_input('Enter a name for your project pipeline. Please use only letters, numbers, and underscores.')
            input_path_to_pipeline_dir = st.text_input('Enter a path to a folder where the new project pipeline will be stored. Press Enter when done.', value=config.PIPELINE_OUTPUT_PATH)
            button_project_info_submitted_is_clicked = st.button('Submit', key='SubmitNewProjectInfo')

            if button_project_info_submitted_is_clicked:
                # Error checking first
                if check_arg.has_invalid_chars_in_name_for_a_file(text_input_new_project_name):
                    char_err = ValueError(f'Project name has invalid characters present. Re-submit project pipeline name. {text_input_new_project_name}')
                    logger.error(char_err)
                    st.error(char_err)
                    st.stop()
                if not os.path.isdir(input_path_to_pipeline_dir):
                    dir_err = f'The following (in double quotes) is not a valid directory: "{input_path_to_pipeline_dir}"'
                    logger.error(dir_err)
                    st.error(NotADirectoryError(dir_err))
                    st.stop()

                # If OK: create default pipeline, save, continue
                p = base_pipeline.BasePipeline(text_input_new_project_name)
                save_to_folder(p, input_path_to_pipeline_dir)
                if radio_select_create_method == text_dibs_data_pipeline:
                    with st.spinner('Instantiating pipeline with data now...'):
                        p = p.add_train_data_source(config.DEFAULT_TRAIN_DATA_DIR)
                        p = p.add_predict_data_source(config.DEFAULT_TEST_DATA_DIR)
                elif radio_select_create_method == text_half_dibs_data_pipeline:
                    with st.spinner('Instantiating pipeline with data now...'):
                        all_files = [os.path.join(config.DEFAULT_TRAIN_DATA_DIR, file) for file in os.listdir(config.DEFAULT_TRAIN_DATA_DIR)]
                        half_files = all_files[:len(all_files)//2]
                        p = p.add_train_data_source(*half_files)
                        p = p.add_predict_data_source(config.DEFAULT_TEST_DATA_DIR)
                save_to_folder(p, input_path_to_pipeline_dir)
                pipeline_file_path = os.path.join(input_path_to_pipeline_dir, pipeline.generate_pipeline_filename( text_input_new_project_name))
                session[key_pipeline_path] = pipeline_file_path
                st.balloons()
                st.success(f"""
Success! Your new project pipeline has been saved to disk to the following path: 

{os.path.join(input_path_to_pipeline_dir, f'{text_input_new_project_name}.pipeline')}

""".strip())
                n_secs_til_refresh = session[default_n_seconds_sleep]
                st.info(
                    f'The page will automatically refresh with your new pipeline in {n_secs_til_refresh} seconds...')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                time.sleep(n_secs_til_refresh)
                st.experimental_rerun()

        # Option 2/2: Load existing project
        elif start_select_option == load_existing_project_option_text:
            logger.debug(f'Open LOAD EXISTING option')
            st.markdown('## Load existing project pipeline')

            existing_pipeline_files: List[str] = \
                [config.default_pipeline_file_path_or_name] + [file for file in os.listdir(config.PIPELINE_OUTPUT_PATH)]

            input_text_path_to_pipeline_file = os.path.join(
                config.PIPELINE_OUTPUT_PATH,
                st.selectbox(label=f'Select directory name corresponding to videos you want to see (you would have specified this when creating the videos)', options=existing_pipeline_files)
            )
            # # Code frament of potential file selector...not yet working
            # input_text_path_to_pipeline_file = st_file_selector(pl)
            # pl = st.empty()
            #
            # session[key_open_pipeline_path] = st_file_selector(
            #     st_placeholder=pl,
            #     path=session[key_open_pipeline_path],
            #     label=f'Input path')
            #
            # st.text(f'> Selected \'{session[key_open_pipeline_path]}\'')
            # logger.debug(f'LOAD: session[key_open_pipeline_path] = {session[key_open_pipeline_path]}')
            # input_text_path_to_pipeline_file = session[key_open_pipeline_path]

            ############ Original, working implementation below
            # input_text_path_to_pipeline_file = clean_string(st.text_input(
            #     'Enter full path to existing project pipeline file',
            #     value=pipeline_file_path,  # TODO: remove this line later, or change to a config default?
            #     key='text_input_load_existing_pipeline'
            # ))

            # Do checks on pipeline load
            if input_text_path_to_pipeline_file:
                # Error checking first
                if not os.path.isfile(input_text_path_to_pipeline_file) or not input_text_path_to_pipeline_file.endswith('.pipeline'):
                    err = f'DIBS Pipeline file was not found. User submitted path: {input_text_path_to_pipeline_file}'
                    logger.error(err)
                    st.error(FileNotFoundError(err))
                    st.stop()
                # If OK: load project, continue
                pipeline_file_path = input_text_path_to_pipeline_file
                if session[key_pipeline_path] != pipeline_file_path:
                    session[key_pipeline_path] = pipeline_file_path
                logger.debug(f'Attempting to open: {pipeline_file_path}')
                p = io.read_pipeline(pipeline_file_path)
                logger.info(f'Streamlit: successfully opened: {pipeline_file_path}')
                st.success('Pipeline loaded successfully.')
                is_pipeline_loaded = True

        # Option: no (valid) selection made. Wait for user to select differently.
        else:
            return
    except Exception as e:
        # In case of error, show error and do not continue
        st.markdown('An unexpected error occurred. See below:')
        st.info(f'Traceback: {get_traceback_string()}')
        st.error(e)
        st.info(f'Stack trace for error: {str(traceback.extract_stack())}')
        logger.error(f'{repr(e)} // {str(traceback.extract_stack())}')
        return

    if is_pipeline_loaded:
        logger.debug(f"Leaving home: pipeline_file_path = {pipeline_file_path}")
        st.markdown('----------------------------------------------------------------------------------------------')
        if not os.path.isfile(pipeline_file_path):
            err = f'Pipeline file path got lost along the way. Path = {pipeline_file_path}'
            st.error(FileNotFoundError(err))
            st.stop()

        # HACKS: Should have been a separate streamlit dialog.
        feature_engineering_class = type(p._feature_engineerer)
        # Show extra pipeline text if necessary
        if session[checkbox_show_extra_text]:
            feature_engineering_info = feature_engineering_class.__doc__ if str(feature_engineering_class.__doc__).strip() else f'No documentation detected for {feature_engineering_class.__name__}'
            st.info(f'Current feature engineering set loaded: {feature_engineering_class.__name__}\n       {feature_engineering_info}')
        st.markdown('')

        button_change_feature_engineering_set = st.button('Toggle: Change feature engineering set?', key=key_button_change_feature_engineering)
        if button_change_feature_engineering_set:
            session[key_button_change_feature_engineering] = not session[key_button_change_feature_engineering]
        if session[key_button_change_feature_engineering]:
            selected_feature_engineering = st.selectbox('Select a feature engineering set', options=[''] + list(config.valid_feature_engineerers))
            st.markdown('')
            if selected_feature_engineering:  # After selecting a Pipeline implementation...
                p.set_params(FEATURE_ENGINEERER=(selected_feature_engineering, dict()))
                save_to_folder(p, os.path.dirname(pipeline_file_path))
                st.markdown('You have changed the engineered feature set!')

        show_pipeline_info(p, pipeline_file_path)


def show_pipeline_info(p: pipeline.BasePipeline, pipeline_path):

    """  """
    # logger.debug(f'{logging_enhanced.get_current_function()}(): Starting. pipeline_path = {pipeline_path}')  # Debugging effort

    ### SIDEBAR ###

    ### MAIN PAGE ###
    st.markdown(f'## Pipeline basic information')
    st.markdown(f'- Name: **{p.name}**')
    st.markdown(f'- Description: **{p.description}**')
    st.markdown(f'- File location: **{pipeline_path}**')
    # st.markdown(f'- Total number of training data sources: **{len(p.training_data_sources)}**')
    # st.markdown(f'- Total number of predict data sources: **{len(p.predict_data_sources)}**')
    st.markdown(f'- Is the model built: **{p.is_built}**')

    ### Menu button: show more info
    button_show_advanced_pipeline_information = st.button(f'Toggle: show advanced info', key=key_button_show_adv_pipeline_information)
    if button_show_advanced_pipeline_information:
        session[key_button_show_adv_pipeline_information] = not session[key_button_show_adv_pipeline_information]
    if session[key_button_show_adv_pipeline_information]:
        st.markdown(f'- Training data sources ({len(p.training_data_sources)} total sources):')
        if len(p.training_data_sources) > 0:
            for s in p.training_data_sources: st.markdown(f'- - **{s}**')
        else:
            st.markdown(f'- - **None**')
        st.markdown(f'- Predict data sources ({len(p.predict_data_sources)} total sources):')
        if len(p.predict_data_sources) > 0:
            for s in p.predict_data_sources:
                st.markdown(f'- - **{s}**')
        else:
            st.markdown(f'- - **None**')

        st.markdown(f'- Number of data points in training data set: '
                    f'**{len(p.df_features_train_raw)//p.average_over_n_frames if not p.is_built else len(p.df_features_train_scaled)}**')

        # TODO: Right now this displays really poorly.  Figure out a way to fix it.
        st.markdown(f'{p._embedder.__class__.__name__} params:')
        st.markdown(f'{p._embedder.params_as_string()}')

        st.markdown(f'{p._clusterer.__class__.__name__} params:')
        st.markdown(f'{p._clusterer.params_as_string()}')

        st.markdown(f'{p._clf.__class__.__name__} params:')
        st.markdown(f'{p._clf.params_as_string()}')

        # st.markdown(f'- Perplexity % of training data (perp / data): {round(p.tsne_perplexity_relative_to_num_data_points, 5)*100}%')
        acc_pct = f'{p.accuracy_score * 100.}%' if p.accuracy_score >= 0. else 'N/A'
        st.markdown(f'- Accuracy (with test fraction at {p.test_train_split_pct*100.}% of total training data): **{acc_pct}**')
        st.markdown(f'Model Feature Names:')
        for feat in p.all_engineered_features:
            st.markdown(f'- - {feat}')
        st.markdown(f'Results')
        st.markdown(f'- Total unique behaviours clusters: **{len(p.unique_assignments) if p.is_built else "[Not Available]"}**')
        st.markdown(f'- Unique behaviours assignments: {p.unique_assignments}')
        cross_val_decimals_round = 2
        cross_val_score_text = f'- Median cross validation score: **{round(float(np.median(p.cross_val_scores)), cross_val_decimals_round) if p.is_built else None}**' + \
                               f' (literal scores: {sorted([round(x, cross_val_decimals_round) for x in list(p.cross_val_scores)])})' if p.is_built else ''
        st.markdown(f'{cross_val_score_text}')
        if p.is_built:
            st.markdown('')
            st.markdown(f'- Build time: {p.seconds_to_build} seconds')

    ###

    # Model check before displaying actions that could further change pipeline state.
    if p.is_in_inconsistent_state:
        st.markdown('')
        st.info(fr"""
The pipeline is detected to be in an inconsistent state. 

Some common causes include adding/deleting training data or changing model 
parameters without subsequently rebuilding the model.

We recommend that you rebuild the model to avoid future problems.

Dev details:

{p.get_inconsistent_state_repr()}
""".strip())

    # # TODO: for below commented-out: add a CONFIRM button to confirm model re-build, then re-instate

    st.markdown('------------------------------------------------------------------------------------------------')

    return show_actions(p, pipeline_path)


def show_actions(p: pipeline.BasePipeline, pipeline_file_path):
    """ Show basic actions that we can perform on the model """
    ### SIDEBAR ###

    ### MAIN PAGE ###
    if not os.path.isfile(pipeline_file_path):
        st.error(f'An unexpected error occurred. Your pipeline file path was lost along the way. Currently, your pipeline file path reads as: "{pipeline_file_path}"')
        st.stop()
    st.markdown(f'## Actions')

    ################################# CHANGE PIPELINE INFORMATION ###############################################
    st.markdown(f'### Pipeline information')

    ### Change pipeline description
    button_update_description = st.button(f'Toggle: Change project description', key_button_update_description)
    if button_update_description:
        session[key_button_update_description] = not session[key_button_update_description]
    if session[key_button_update_description]:
        text_input_change_desc = st.text_input(f'(WORK IN PROGRESS) Change project description here', value=p.description)
        if text_input_change_desc != p.description:
            p.set_description(text_input_change_desc)
            save_to_folder(p, os.path.dirname(pipeline_file_path))
            session[key_button_update_description] = False
            wait_seconds = session[default_n_seconds_sleep]
            st.success(f'Pipeline description has been changed!')
            st.info(f'This page will refresh automatically to reflect your changes in {wait_seconds} seconds.')
            time.sleep(wait_seconds)
            st.experimental_rerun()
        for i in range(4):
            st.markdown('')
    ### End: Change pipeline description
    st.markdown('')
    # TODO: low: add a "change save location" option?

    ####################################### MODEL BUILDING #############################################
    st.markdown(f'## Model building & information')

    ### Menu button: adding new data ###
    button_add_new_data = st.button('Toggle: Add new data to model', key_button_add_new_data)
    if button_add_new_data:  # Click button, flip state
        session[key_button_add_new_data] = not session[key_button_add_new_data]
    if session[key_button_add_new_data]:  # Now check on value and display accordingly
        st.markdown(f'### Do you want to add data that will be used to train the model, or '
                    f'data that the model will evaluate?')
        # 1/2: Button for adding data to training data set
        button_add_train_data_source = st.button('-> Add new data for training the model', key=key_button_add_train_data_source)
        if button_add_train_data_source:
            session[key_button_add_train_data_source] = not session[key_button_add_train_data_source]
            session[key_button_add_predict_data_source] = False  # Close the menu for adding prediction data
        if session[key_button_add_train_data_source]:
            # # New implementation via tkinter (NOT WORKING YET - THREAD SAFE PROBS)

            # p = p.add_train_data_source(*file_paths).save(os.path.dirname(pipeline_file_path))
            # st.success(f'Success! Refresh page to see changes')
            # session[key_button_add_train_data_source] = False
            # st.stop()
            # st.success(f'Success! The following files have been added to the pipeline as training data: '
            #            f'{", ".join([os.path.split(x)[-1] for x in file_paths])}. Refresh the page to see the changes')
            # session[key_button_add_new_data] = False
            # st.stop()
            ############

            # Original implementation below: dont delete yet!
            st.markdown(f"_Input a file path below to data which will be used to train the model._\n\n_You may input a file path or a path to a directory. If a directory is entered, all CSVs directly within will be added as training data_")
            if config.DEFAULT_TRAIN_DATA_DIR:
                input_new_data_source = st.selectbox('Select dataset to add: ', options=['']+[file for file in os.listdir(config.DEFAULT_TRAIN_DATA_DIR)])
                if input_new_data_source:
                    input_new_data_source = os.path.join(config.DEFAULT_TRAIN_DATA_DIR, input_new_data_source)
            else:
                input_new_data_source = st.text_input('Past absolute path to data to add: ')
            if input_new_data_source:
                # Check if file exists
                if not os.path.isfile(input_new_data_source) and not os.path.isdir(input_new_data_source):
                    st.error(FileNotFoundError(f'File not found: {input_new_data_source}. Data not added to pipeline.'))
                elif os.path.isdir(input_new_data_source) and len([file for file in os.listdir(input_new_data_source) if file.endswith('csv')]) == 0:
                    err = f'Zero CSV files were found in the submitted folder: {input_new_data_source}'
                    st.error(err)
                # Add to pipeline, save
                else:
                    p = p.add_train_data_source(input_new_data_source)
                    save_to_folder(p, os.path.dirname(pipeline_file_path))
                    session[key_button_add_train_data_source] = False  # Reset menu to collapsed state
                    session[key_button_add_new_data] = False
                    n = session[default_n_seconds_sleep]
                    st.balloons()
                    st.success(f'New training data added to pipeline successfully! Pipeline has been saved to: "{pipeline_file_path}".')  # TODO: finish statement. Add in suggestion to refresh page.
                    st.info(f'This page will refresh automatically in {n} seconds')
                    time.sleep(n)
                    st.experimental_rerun()
            st.markdown('')

        # 2/2: Button for adding data to prediction set
        button_add_predict_data_source = st.button('-> Add data to be evaluated by the model', key=key_button_add_predict_data_source)
        if button_add_predict_data_source:
            session[key_button_add_predict_data_source] = not session[key_button_add_predict_data_source]
            session[key_button_add_train_data_source] = False  # Close the menu for adding training data
        if session[key_button_add_predict_data_source]:
            st.markdown(f'TODO: add in new predict data')
            input_new_predict_data_source = st.text_input(f'Input a file path below to a new data source which will be analyzed by the model.')
            if input_new_predict_data_source:
                # Check if file exists
                if not os.path.isfile(input_new_predict_data_source) and not os.path.isdir(input_new_predict_data_source):
                    st.error(FileNotFoundError(f'File not found: {input_new_predict_data_source}. Data not added to pipeline.'))
                elif os.path.isdir(input_new_predict_data_source) and len([file for file in os.listdir(input_new_predict_data_source) if file.endswith('csv')]) == 0:  # TODO: use .endiswith()
                    err = f'Zero CSV files were found in the submitted folder: {input_new_predict_data_source}'
                    st.error(err)
                # if not os.path.isfile(input_new_predict_data_source):
                #     st.error(FileNotFoundError(f'File not found: {input_new_predict_data_source}. '
                #                                f'No data was added to pipeline prediction data set.'))
                else:
                    p = p.add_predict_data_source(input_new_predict_data_source)
                    save_to_folder(p, os.path.dirname(pipeline_file_path))
                    session[key_button_add_predict_data_source] = False  # Reset add predict data menu to collapsed state
                    session[key_button_add_new_data] = False  # Reset add menu to collapsed state
                    n_wait_secs = session[default_n_seconds_sleep]
                    st.success(f'New prediction data added to pipeline successfully! Pipeline has been saved.')
                    st.info(f'This page will refresh with your new changes in {n_wait_secs} seconds.')
                    time.sleep(n_wait_secs)
                    st.experimental_rerun()
        st.markdown('')
        st.markdown('')
        st.markdown('')
    ### End of menu for adding data ###

    ### Menu button: removing data ###
    button_remove_data = st.button('Toggle: Remove data from model', key_button_menu_remove_data)
    if button_remove_data:
        session[key_button_menu_remove_data] = not session[key_button_menu_remove_data]
    if session[key_button_menu_remove_data]:
        select_train_or_predict_remove = st.selectbox('Select which data you want to remove', options=['', training_data_option, predict_data_option])
        if select_train_or_predict_remove == training_data_option:
            select_train_data_to_remove = st.selectbox('Select a source of data to be removed', options=['']+p.training_data_sources)
            if select_train_data_to_remove:
                st.info(f'Are you sure you want to remove the following data from the training data set: {select_train_data_to_remove} ?')
                st.markdown(f'*NOTE: upon removing the data, the model will need to be rebuilt.*')
                confirm = st.button('Confirm')
                if confirm:
                    with st.spinner(f'Removing {select_train_data_to_remove} from training data set...'):
                        p = p.remove_train_data_source(select_train_data_to_remove)
                        save_to_folder(p, os.path.dirname(pipeline_file_path))
                    session[key_button_menu_remove_data] = False
                    n = session[default_n_seconds_sleep]
                    st.balloons()
                    st.success(f'{select_train_data_to_remove} data successfully removed!')
                    st.info(f'The page will refresh shortly, or you can manually refresh the page to see the changes')
                    time.sleep(n)
                    st.experimental_rerun()
                st.markdown('------------------------------------------')

        if select_train_or_predict_remove == predict_data_option:
            select_predict_option_to_remove = st.selectbox('Select a source of data to be removed', options=['']+p.predict_data_sources)
            if select_predict_option_to_remove:
                st.info(f'Are you sure you want to remove the following data from the predicted/analyzed data set: {select_predict_option_to_remove}')
                st.markdown(f'*NOTE: upon removing the data, the model will need to be rebuilt.*')
                confirm = st.button('Confirm')
                if confirm:
                    with st.spinner(f'Removing {select_predict_option_to_remove} from predict data set'):
                        p = p.remove_predict_data_source(select_predict_option_to_remove)
                        save_to_folder(p, os.path.dirname(pipeline_file_path))
                    session[key_button_menu_remove_data] = False
                    st.balloons()
                    st.success(f'{select_predict_option_to_remove} data was successfully removed!')
                    st.info(f'The page will refresh shortly, or you can manually refresh the page to see the changes')
                    time.sleep(session[default_n_seconds_sleep])
                    st.experimental_rerun()
                st.markdown('------------------------------------------')
                st.markdown('')
        st.markdown('')

    ### End of menu for removing data ###

    st.markdown('')

    # AARONT: TODO: How are the config values read from disk? Ever modified?
    ### Menu button: rebuilding model ###
    button_see_rebuild_options = st.button('Toggle: Review Model Parameters & Rebuild Model', key_button_see_rebuild_options)
    if button_see_rebuild_options:  # Click button, flip state
        session[key_button_see_rebuild_options] = not session[key_button_see_rebuild_options]
    if session[key_button_see_rebuild_options]:  # Now check on value and display accordingly
        # Current params
        st.markdown('')
        st.info(f'''
Current model params:

Embedder: {p._embedder.__class__.__name__}
Embedder params: {p._embedder.get_params()}

Clusterer: {p._clusterer.__class__.__name__}
Clusterer params: {p._clusterer.get_params()}

CLF: {p._clf.__class__.__name__}
CLF params: {p._clf.get_params()}
''')
        st.markdown('')
        st.markdown('## Model Parameters')
        st.markdown(f'### General parameters')
        input_video_fps = st.number_input(f'Video FPS of input data', value=float(p.video_fps), min_value=0., max_value=500., step=1.0, format='%.2f')
        input_average_over_n_frames = st.slider('Select number of frames to average over', value=p.average_over_n_frames, min_value=1, max_value=10, step=1)
        st.markdown(f'By averaging features over **{input_average_over_n_frames}** frame at a time, it is effectively averaging features using a **{round(input_average_over_n_frames / config.VIDEO_FPS * 1_000)}ms** window.')
        st.markdown(f'*By averaging over larger windows, the model can provide better generalizability, but using smaller windows is more likely to find more minute actions*')

        if session[checkbox_show_extra_text]:
            st.info('Increasing the averaging frame will help reduce jitter when labeling,'
                    ' but it will also smooth out any nuanced behaviours that may be short in duration')

        # # TODO: Low/Med: implement variable feature selection
        # st.markdown(f'### Select features')
        # st.multiselect('select features', p.all_features, default=p.all_features)  # TODO: develop this feature selection tool!
        # st.markdown('---')

        st.markdown('### Gaussian Mixture Model Parameters')
        slider_gmm_n_components = st.slider(f'GMM Components (number of clusters)', value=p._clusterer.n_components, min_value=2, max_value=40, step=1)
        st.markdown(f'_You have currently selected __{slider_gmm_n_components}__ clusters_')

        # Extra info: GMM components
        if session[checkbox_show_extra_text]:
            st.info('Increasing the maximum number of GMM components will increase the maximum number of behaviours that'
                    ' are labeled in the final output.')

        st.markdown('')
        # TODO: low: add GMM: probability = True
        # TODO: low: add: GMM: n_jobs = -2

        ### Other model info ###
        st.markdown('### Other model information')
        # TODO: Remove?
        ## input_cross_validation_k = st.number_input(f'Set K for K-fold cross validation', value=int(p.cross_validation_k), min_value=2, format='%i')  # TODO: low: add max_value= number of data points (for k=n)?
        # TODO: med/high: add number input for % holdout for test/train split
        # Hack solution: specify params here so that the variable exists even though advanced params section not opened.

        # Extra info: k-folds

        st.markdown('')
        ### Set up default values for advanced parameters in case the user does not set advanced parameters at all
        # features = p.all_features # TODO: Unused, remove?
        embedder_params, clusterer_params, clf_params = p.get_model_params()

        input_average_over_n_frames = p.average_over_n_frames

        # HACK: TODO: Fix.  Only show the dialog for TSNE
        if p._embedder.__class__.__name__ == 'TSNE':
            # TODO: Streamline the streamlit stuff.  This is temporary
            input_tsne_learning_rate = embedder_params['learning_rate']
            input_tsne_perplexity = embedder_params['make_this_better_perplexity']
            input_tsne_early_exaggeration = embedder_params['early_exaggeration']
            input_tsne_n_components = embedder_params['n_components']
            input_tsne_n_iter = embedder_params['n_iter']

        input_gmm_reg_covar = clusterer_params['reg_covar']
        input_gmm_tol = clusterer_params['tol']
        input_gmm_max_iter = clusterer_params['max_iter']
        input_gmm_n_init = clusterer_params['n_init']

        # TODO: REMOVE TEMP HACKS HERE!!  This whole thing needs to be generalized in some way.
        select_classifier = p._clf.__class__
        input_svm_c = clf_params.get('c')
        input_svm_gamma = clf_params.get('gamma')
        select_rf_n_estimators = clf_params.get('n_estimators')

        ### Advanced Parameters ###
        st.markdown('### Advanced Parameters')
        # TODO: HIGH IMPORTANCE! The advanced parameters should reflect the classifier type being used (SVM vs RF vs something new in the future)
        st.markdown('*Toggle advanced parameters at your own risk. Many variables require special knowledge of ML parameters*')
        button_see_advanced_options = st.button('Toggle: show advanced parameters')
        if button_see_advanced_options:
            session[key_button_see_advanced_options] = not session[key_button_see_advanced_options]
        if session[key_button_see_advanced_options]:
            # st.markdown('### Advanced model options. ')
            st.markdown('### Do not change things here unless you know what you are doing!')
            st.markdown('*Note: If you collapse the advanced options menu, all changes will be lost. To retain advanced parameters changes, ensure that the menu is still open when clicking the "Rebuild" button.*')
            ### See advanced options for model ###

            # TSNE
            # HACK: TODO: Fix.  Only show the dialog for TSNE
            if p._embedder.__class__.__name__ == 'TSNE':

                st.markdown('### Advanced TSNE Parameters')
                if session[checkbox_show_extra_text]:
                    st.info('See the original paper describing this algorithm for more details.'
                            ' Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(Nov), 2579-2605'
                            ' Section 2 includes perplexity.')
                # TODO: med/high: add radio select button for choosing absolute value or choosing ratio value #######################
                input_tsne_perplexity = st.number_input(label=f'TSNE Perplexity', value=input_tsne_perplexity, min_value=0.1, max_value=1000.0, step=10.0)  # TODO: handle default perplexity value (ends up as 0 on fresh pipelines)
                # Extra info: tsne-perplexity
                if session[checkbox_show_extra_text]:
                    st.info('Perplexity can be thought of as a smooth measure of the effective number of neighbors that are considered for a given data point.')  # https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a: "A perplexity is more or less a target number of neighbors for our central point. Basically, the higher the perplexity is the higher value variance has"
                input_tsne_learning_rate = st.number_input(label=f'TSNE Learning Rate', value=input_tsne_learning_rate, min_value=0.01)  # TODO: high is learning rate of 200 really the max limit? Or just an sklearn limit?
                # Extra info: learning rate
                if session[checkbox_show_extra_text]:
                    st.info('TODO: learning rate')  # TODO: low
                input_tsne_early_exaggeration = st.number_input(f'TSNE Early Exaggeration', value=input_tsne_early_exaggeration, min_value=0., step=0.1, format='%.2f')
                # Extra info: early exaggeration
                if session[checkbox_show_extra_text]:
                    st.info('TODO: early exaggeration')  # TODO: low
                input_tsne_n_iter = st.number_input(label=f'TSNE N Iterations', value=input_tsne_n_iter, min_value=config.minimum_tsne_n_iter, max_value=5_000)
                # Extra info: number of iterations
                if session[checkbox_show_extra_text]:
                    st.info('TODO: number of iterations')  # TODO: low
                input_tsne_n_components = st.number_input(f'TSNE N Components/Dimensions', value=input_tsne_n_components, min_value=2, max_value=3, step=1, format='%i')
                # Extra info: number of components (dimensions)
                if session[checkbox_show_extra_text]:
                    st.info('TODO: number of components (dimensions)')  # TODO: low

            st.markdown(f'### Advanced GMM parameters')
            input_gmm_reg_covar = st.number_input(f'GMM "reg. covariance" ', value=input_gmm_reg_covar, format='%f')
            # Extra info: reg covar
            if session[checkbox_show_extra_text]:
                st.info('TODO: reg covar')  # TODO: low
            input_gmm_tol = st.number_input(f'GMM tolerance', value=input_gmm_tol, min_value=1e-10, max_value=50., step=0.1, format='%.2f')
            # Extra info: GMM tolerance
            if session[checkbox_show_extra_text]:
                st.info('TODO: GMM tolerance')  # TODO: low
            input_gmm_max_iter = st.number_input(f'GMM max iterations', value=input_gmm_max_iter, min_value=1, max_value=100_000, step=1, format='%i')
            # Extra info: GMM max iterations
            if session[checkbox_show_extra_text]:
                st.info('TODO: GMM max iterations')  # TODO: low
            input_gmm_n_init = st.number_input(f'GMM "n_init" ("Number of initializations to perform. the best results is kept")  . It is recommended that you use a value of 20', value=input_gmm_n_init, min_value=1, step=1, format="%i")
            # Extra info: GMM number of initializations
            if session[checkbox_show_extra_text]:
                st.info('TODO: GMM number of initializations')  # TODO: low

            st.markdown(f'### Advanced Classifier Parameters')
            select_classifier = st.selectbox('Select a classifier type:', options=[p._clf.__class__.__name__] + [clf_type for clf_type in list(config.valid_classifiers) if clf_type != p._clf.__class__.__name__])
            if select_classifier == 'SVM':
                ### SVM ###
                input_svm_c = st.number_input(f'SVM C', value=input_svm_c, min_value=1e-10, format='%.2f')
                if session[checkbox_show_extra_text]:
                    st.info('TODO: explain SVM "C" parameter')  # TODO: low
                input_svm_gamma = st.number_input(f'SVM gamma', value=input_svm_gamma, min_value=1e-10, format='%.2f')
                if session[checkbox_show_extra_text]:
                    st.info('TODO: explain SVM "gamma" parameter')  # TODO: low
            elif select_classifier == 'RANDOMFOREST':
                select_rf_n_estimators = st.number_input('Random Forest N estimators', value=p._clf.n_estimators, min_value=1, max_value=1_000, format='%i')
                if session[checkbox_show_extra_text]:
                    st.info('TODO: explain RF # of trees and roughly optimal #')
        ### End of Show Advanced Params Section

        st.markdown('')

        st.markdown('### Rebuilding Model')
        st.markdown(f'*Note: changing the above parameters without rebuilding the model will have no effect.*')

        # Save above info & rebuild model
        # st.markdown('## Rebuild model with new parameters above?')
        button_rebuild_model = st.button('I want to rebuild model with new parameters', key_button_rebuild_model)
        if button_rebuild_model: session[key_button_rebuild_model] = not session[key_button_rebuild_model]
        if session[key_button_rebuild_model]:  # Rebuild model button was clicked
            st.markdown('Are you sure?')
            button_confirmation_of_rebuild = st.button('Confirm', key_button_rebuild_model_confirmation)
            if button_confirmation_of_rebuild:
                session[key_button_rebuild_model_confirmation] = not session[key_button_rebuild_model_confirmation]
            if session[key_button_rebuild_model_confirmation]:  # Rebuild model confirmed.
                try:
                    with st.spinner('Building model. This could take a couple minutes...'):

                        # HACK: TODO: Fix.  Only show the dialog for TSNE
                        if p._embedder.__class__.__name__ == 'TSNE':
                            _hack_embedder_params = (
                                'TSNE', {
                                    'perplexity': input_tsne_perplexity,
                                    'learning_rate': input_tsne_learning_rate,
                                    'early_exaggeration': input_tsne_early_exaggeration,
                                    'n_iter': input_tsne_n_iter,
                                    'n_components': input_tsne_n_components,
                                }
                            )
                        else:
                            _hack_embedder_params = (
                                'PrincipalComponents', dict() # TODO: Allow params in GUI
                            )
                        model_vars = {
                            # General opts
                            'video_fps': input_video_fps,
                            'average_over_n_frames': input_average_over_n_frames,
                            # 'cross_validation_k': input_cross_validation_k, # TODO: Remove?

                            # TODO: Generalize this dict of params so that we can dynamically set algos
                            # Something like: {type_of_thing1: (selected_thing, selected_params_for_thing)}
                            'EMBEDDER': _hack_embedder_params,
                            'CLUSTERER': (
                                'GMM', {
                                    'n_components': slider_gmm_n_components,
                                    'reg_covar': input_gmm_reg_covar,
                                    'tol': input_gmm_tol,
                                    'max_iter': input_gmm_max_iter,
                                    'n_init': input_gmm_n_init,
                                }
                            ),
                            'CLF':
                                ('RANDOMFOREST', {
                                    # 'classifier_type': select_classifier,
                                    'n_estimators': select_rf_n_estimators,
                                })
                            # 'SVM': {
                            #     'svm_c': input_svm_c,
                            #     'svm_gamma': input_svm_gamma,
                            # }

                        }

                        # TODO: HIGH: make sure that model parameters are put into Pipeline before build() is called.
                        p = p.set_params(**model_vars)
                        if not os.path.isdir(os.path.dirname(pipeline_file_path)):
                            st.error(f'UNEXPECTED ERROR: pipeline file DIRECTORY parsed as: {os.path.dirname(pipeline_file_path)}')
                            st.stop()
                        # Build, save at each intermediate step, in case of errors later.
                        p = p.build(pipeline_file_path=os.path.dirname(pipeline_file_path))
                        session[key_button_rebuild_model_confirmation] = False
                    st.balloons()
                    session[key_button_see_rebuild_options] = False
                    session[key_button_rebuild_model_confirmation] = False
                    n = session[default_n_seconds_sleep]
                    st.success(f'Model was successfully re-built! This page will auto-refresh in {n} seconds.')
                    time.sleep(n)
                    st.experimental_rerun()
                except Exception as e:
                    traceback_string = '        \n'.join(traceback.format_exc().split('\n'))
                    err = f'UNEXPECTED EXCEPTION WHEN BUILDING PIPELINE: {repr(e)}.' \
                          f'    \n'\
                          f'    \nTraceback:' \
                          f'    \n{traceback_string}.'
                    logger.error(err, exc_info=True)
                    st.error(err)
                    st.stop()

    ### End of rebuild model section

    ### Menu button: re-colour clusters ###

    ### End of menu for re-colour clusters ###

    st.markdown('--------------------------------------------------------------------------------------------------')

    return see_model_diagnostics(p, pipeline_file_path)


def see_model_diagnostics(p: pipeline.BasePipeline, pipeline_file_path):
    ######################################### MODEL DIAGNOSTICS ########################################################

    ### SIDEBAR

    ### MAIN
    st.markdown(f'## Model Diagnostics')

    ### View PCA plot for selected features (hopefully explaining some feature viability?)
    # TODO: med/high

    ###

    ### View confusion matrix for test-data
    button_view_confusion_matrix = st.button(f'Toggle: View confusion matrix')
    if button_view_confusion_matrix:
        session[key_button_view_confusion_matrix] = not session[key_button_view_confusion_matrix]
    if session[key_button_view_confusion_matrix]:
        if p._clf_is_built:
            st.pyplot(p.plot_confusion_matrix())
        else:
            st.info(f'Confusion matrix not available since the classifier has not been built yet.')
        if p.is_in_inconsistent_state:
            st.warning(f'The model was detected to be in an inconsistent state, so the current confusion matrix may not reflect changes in the model done AFTER build.')

    ### View Histogram for assignment distribution
    st.markdown(f'*This section is a work-in-progress. Opening a graph in this section is very volatile and there is high chance that by opening a graph then streamlit will crash. A fix is actively being worked-on!*')
    st.markdown(f'### View distribution of assignments')
    button_view_assignments_distribution = st.button(f'Toggle: View assignment distribution')
    if button_view_assignments_distribution:
        session[key_button_view_assignments_distribution] = not session[key_button_view_assignments_distribution]
    if session[key_button_view_assignments_distribution]:
        if p.is_built:
            # matplotlib.use('Agg')  # <- Hopefully this fixes crashes; no guarantees. TODO: med: review this line later.
            # fig, ax = p.get_plot_svm_assignments_distribution()
            # st.pyplot(fig)
            # matplotlib.use('TkAgg')

            ### Show bar chart of counts of behaviours
            # AARONT: TODO: Add 2 scales to a single histogram
            st.markdown(f'** Graph below: behaviour counts **')
            df_assignment_histogram = p.df_features_train_scaled[p.clf_assignment_col_name].value_counts()
            st.bar_chart(df_assignment_histogram)
            st.markdown('')
            ### Show bar chart of % behaviour occurrence
            st.markdown(f'** Graph below: behaviour counts histogram **')
            df_assignment_histogram = p.df_features_train_scaled[p.clf_assignment_col_name].value_counts(normalize=True)
            st.bar_chart(df_assignment_histogram)
            st.markdown('')
            ### Show bar chart of % behaviour occurrence
            st.markdown(f'** Graph below: Transition matrix for behaviours **')
            st.pyplot(p.plot_transition_matrix_heatmap())
        else:
            st.info('There are no assignment distributions available for display because '
                    'the model is not currently built.')
        st.markdown('')

    # View GMM/clustering plot
    st.markdown(f'### See GMM distributions according to TSNE-reduced feature dimensions')  # TODO: low: phrase better
    gmm_button = st.button('Toggle: graph of cluster/assignment distribution')  # TODO: low: phrase this button better?
    if gmm_button: session[key_button_view_assignments_clusters] = not session[key_button_view_assignments_clusters]
    if session[key_button_view_assignments_clusters]:
        if p.is_built:
            azim, elevation = 15, 110
            if p._embedder.n_components == 3:
                azim = st.slider('azimuth slider', value=azim, min_value=-90, max_value=90)
                elevation = st.slider('elevation slider', value=elevation, min_value=1, max_value=180)
            try:  # Debugging try/catch. Remove when both 3-d and 2-d graphs show and don't crash the app.
                fig, ax = p.plot_clusters_by_assignments(show_now=False, azim_elev=(azim, elevation))
                # st.graphviz_chart(fig)
                # x = plotly.tools.mpl_to_plotly(fig)
                # st.plotly_chart(x)
                st.write(fig)
            except ValueError as ve:
                st.error(f'Cannot plot cluster distribution since the model is not currently built. ({repr(ve)}')
        else:
            st.info('A 3d plot of the cluster distributions could not be created because '
                    'the model is not built. ')
    ###

    ### View histogram(s) that show distribution of lengths of all behaviours
    # TODO: med

    ###

    ### View transition matrix of actions
    # TODO: low

    ###

    st.markdown('--------------------------------------------------------------------------------------------------')

    return review_behaviours(p, pipeline_file_path)


def review_behaviours(p: pipeline.BasePipeline, pipeline_file_path):
    """"""
    # Debugging effort
    if not os.path.isfile(pipeline_file_path):
        st.error(FileNotFoundError(f'An unexpected error occurred. Your pipeline file path was lost along the way. Currently, your pipeline file path reads as: "{pipeline_file_path}"'))

    ####################################################################################################################
    ### SIDEBAR

    ### MAIN

    ## Review Behaviour Example Videos ##
    st.markdown(f'## Behaviour clustering review')

    # AARONT: TODO: Here we give options based on prefix and join path.  This is a temp solution to cut down on number of vids being shown and allow to look at the vids just generated. Long term solution would be to only show vids that are valid for the currently loaded model and clean generated vids periodically (maybe constantly).
    ### Section: create drop-down menu to review videos
    ex_video_dirs: List[str] = [directory for directory in os.listdir(config.EXAMPLE_VIDEOS_OUTPUT_PATH) if os.path.isdir(os.path.join(config.EXAMPLE_VIDEOS_OUTPUT_PATH, directory))]
    ex_video_dirs_selection = st.selectbox(label=f'Select directory name corresponding to videos you want to see (you would have specified this when creating the videos)', options=ex_video_dirs)

    if ex_video_dirs_selection:
        ex_video_dir = os.path.join(config.EXAMPLE_VIDEOS_OUTPUT_PATH, ex_video_dirs_selection)
        example_videos_file_list: List[str] = [video_file_name for video_file_name in os.listdir(ex_video_dir) if video_file_name.split('.')[-1] in valid_video_extensions]  # # TODO: low/med: add user intervention on default path to check?
        # TODO: HACK: Relies on the string 'assignment_##' (number number) for sorting the video examples. If the file format ever changes will break.
        videos_dict: Dict[str: str] = {**{'': ''}, **{video_file_name: os.path.join(ex_video_dir, video_file_name) for video_file_name in example_videos_file_list}}

        video_selection: str = st.selectbox(label=f"Select video to view. Total videos found in example videos folder ({config.EXAMPLE_VIDEOS_OUTPUT_PATH}): {len(videos_dict)-1}", options=list(videos_dict.keys()))
        if video_selection:
            try:
                st.video(get_video_bytes(videos_dict[video_selection]))
            except FileNotFoundError as fe:
                st.error(FileNotFoundError(f'No example behaviour videos were found at this time. Try '
                                           f'generating them at check back again after. // '
                                           f'DEBUG INFO: path checked: {config.EXAMPLE_VIDEOS_OUTPUT_PATH} // {repr(fe)}'))
    ### End section: create drop-down menu to review videos

    st.markdown('')
    st.markdown('')

    ### Create new example videos ###
    button_create_new_ex_videos = st.button(f'Toggle: Create new example videos for each behaviour', key=key_button_show_example_videos_options)
    if button_create_new_ex_videos:
        session[key_button_show_example_videos_options] = not session[key_button_show_example_videos_options]
    if session[key_button_show_example_videos_options]:
        st.markdown(f'Fill in variables for making new example videos of behaviours. Does this line of text need to be altered or even removed?')
        data_source_type = st.selectbox(f'Select type of data to use (training data is usually what you want)', options=['training', 'predict'])
        ex_video_dir_name = clean_string(st.text_input(f'File name prefix. This prefix will help differentiate between example video sets.'))
        number_input_output_fps = st.number_input(f'Output FPS for example videos', value=8., min_value=1., step=1., format='%.2f')
        number_input_max_examples_of_each_behaviour = st.number_input(f'Maximum number of videos created for each behaviour', value=5, min_value=1)
        # TODO: AARON: Number min rows crashed the app when trying to adjust from streamlit.
        number_input_min_rows = st.number_input(f'Number of rows of data required for a detection to occur', value=2, min_value=1, max_value=100, step=1)
        number_input_frames_leadup = st.number_input(f'Number of rows of data that precede and succeed target behaviour', value=2, min_value=0)

        st.markdown('')

        # Get video paths for training data.
        ### Create new example videos button
        st.markdown('#### When the variables above are filled out, press the "Confirm" button below to create new example videos')
        st.markdown('')
        button_create_new_ex_videos = st.button('Confirm', key=key_button_create_new_example_videos)
        if button_create_new_ex_videos:
            is_error_detected = False
            ### Check for errors (display as many errors as necessary for redress)
            # File name prefix check
            if check_arg.has_invalid_chars_in_name_for_a_file(ex_video_dir_name):
                is_error_detected = True
                invalid_name_err_msg = f'Invalid file name submitted. Has invalid char. Prefix="{ex_video_dir_name}"'
                st.error(ValueError(invalid_name_err_msg))
            # Continue if good.
            if not is_error_detected:
                with st.spinner('Creating new videos...'):
                    p = p.make_behaviour_example_videos(
                        data_source_type,
                        ex_video_dir_name,
                        min_rows_of_behaviour=number_input_min_rows,
                        max_examples=number_input_max_examples_of_each_behaviour,
                        output_fps=number_input_output_fps,
                        num_frames_buffer=number_input_frames_leadup,
                    )
                st.success(f'Example videos created!')  # TODO: low: improve message
                session[key_button_show_example_videos_options] = False
                n = session[default_n_seconds_sleep]
                st.info(f'This page will automatically refresh in {n} seconds.')
                time.sleep(n)
                st.experimental_rerun()
        st.markdown('--------------------------------------------------------------------------------------')

    ### End section: create new example videos ###

    ### Review labels for behaviours ###
    button_review_assignments_is_clicked = st.button('Toggle: review behaviour/assignments labels', key=key_button_review_assignments)
    if button_review_assignments_is_clicked:  # Click button, flip state
        session[key_button_review_assignments] = not session[key_button_review_assignments]
    if session[key_button_review_assignments]:  # Depending on state, set behaviours to assignments
        if not p.is_built:
            st.info('The model has not been built yet, so there are no labeling options available.')
        else:
            ### View all assignments
            st.markdown(f'#### All changes entered save automatically. After all changes, refresh page to see changes.')
            for assignment_a in p.unique_assignments:
                # st.markdown(f'Debug info. Current assignment = {assignment_a}')
                session[str(assignment_a)] = p.get_assignment_label(assignment_a)
                existing_behaviour_label = p.get_assignment_label(assignment_a)
                # st.markdown(f'Debug info: Current label = {existing_behaviour_label}')
                existing_behaviour_label = existing_behaviour_label if existing_behaviour_label else ''
                text_input_new_label = st.text_input(f'Add behaviour label to assignment # {assignment_a}', value=existing_behaviour_label, key=f'key_new_behaviour_label_{assignment_a}')
                if text_input_new_label != existing_behaviour_label:
                    # st.markdown(f'Attempting to save label ({text_input_new_label}) to pipeline (found at: {pipeline_file_path})')
                    # st.markdown(f'Debug statement: pipe file path = {os.path.dirname(pipeline_file_path)}')
                    if not os.path.isfile(pipeline_file_path):
                        st.error(f'ERROR FOUND: The following path was not detected to be a file: {pipeline_file_path}')
                    if not os.path.isdir(os.path.dirname(pipeline_file_path)):
                        st.error(f'ERROR FOUND: The following path was not detected to be a directory: {os.path.isdir(os.path.dirname(pipeline_file_path))}')
                    # assert os.path.isdir(os.path.dirname(pipeline_file_path))
                    # Execute
                    p = p.set_label(assignment_a, text_input_new_label)
                    save_to_folder(p, os.path.dirname(pipeline_file_path))

    ### End section: review labels for behaviours ###

    st.markdown('')

    ###

    return results_section(p, pipeline_file_path)


def results_section(p: pipeline.BasePipeline, pipeline_file_path):
    if not os.path.isfile(pipeline_file_path):
        st.error(f'An unexpected error occurred. Your pipeline file path was lost along the way. '
                 f'Currently, your pipeline file path reads as: "{pipeline_file_path}"')
    st.markdown('---------------------------------------------------------------------------------------------')
    st.markdown(f'## Create results')

    # AARONT: TODO: This sounds like it should be labelling all the DATASETS for us.  Add functionality to produce all final labeled datasets.
    #               (This seems to be what export is for?  Revisit)
    # AARONT: TODO: Update to use a default output video dir, use all dataset videos and make one video for each (of training or of predict)
    #               and then to allow viewing the labelled videos.
    ### Label an entire video ###
    button_menu_label_entire_video = st.button('Toggle: Use pipeline model to label to entire video', key=key_button_menu_label_entire_video)
    if button_menu_label_entire_video: session[key_button_menu_label_entire_video] = not session[key_button_menu_label_entire_video]
    if session[key_button_menu_label_entire_video]:
        st.markdown('')
        # AARONT: TODO: After other video section is changed, revisit this.
        selected_data_source = st.selectbox('Select a data source to use as the labels set for specified video:', options=['']+p.training_data_sources+p.predict_data_sources)
        input_video_to_label = st.text_input('Input path to corresponding video which will be labeled:', value=f'{config.STREAMLIT_DEFAULT_VIDEOS_FOLDER}')
        st.markdown('')
        st.markdown('')
        input_new_video_name = st.text_input('Enter a file name for the labeled video output:')
        output_folder = st.text_input('Enter a directory into which the labeled video will be saved:', value=config.OUTPUT_PATH)
        # TODO: med/high: add FPS option for video out
        button_create_labeled_video = st.button('Create labeled video')
        if button_create_labeled_video:
            error_detected = False
            if not os.path.isfile(input_video_to_label):
                error_detected = True
                st.error(f'Path to video not found: {input_video_to_label}')
            if not selected_data_source:
                error_detected = True
                st.error('An invalid data source was selected. Please change the data source and try again.')
            if check_arg.has_invalid_chars_in_name_for_a_file(input_new_video_name):
                error_detected = True
                st.error(f'Invalid characters for new video name detected. Please review name: {input_new_video_name}')
            if error_detected:
                st.stop()
            else:
                with st.spinner('(WIP) Creating labeled video now. This could take a few minutes...'):
                    p.make_video(video_to_be_labeled_path=input_video_to_label,
                                 data_source=selected_data_source,
                                 video_name=input_new_video_name,
                                 output_dir=output_folder)
                st.balloons()
                st.success('Success! Video was created at: TODO: get video out path')  # todo: MED
        st.markdown('---------------------------------------------------------------------------------------')

    return export_data(p, pipeline_file_path)


def export_data(p: pipeline.BasePipeline, pipeline_file_path):
    ### Sidebar
    # TODO: HIGH: ensure that all export buttons work! Test them!
    ### Main
    st.markdown('---------------------------------------------------------------------------------------------------')
    st.markdown(f'## Exporting data')

    # Export training data
    button_export_training_data = st.button(f'Toggle: export training data', key=key_button_export_training_data)
    if button_export_training_data:
        session[key_button_export_training_data] = not session[key_button_export_training_data]
    if session[key_button_export_training_data]:

        training_data_selec = st.selectbox('Select data set to save', ('', raw, engineered, scaled))
        if training_data_selec:

            raw_data_save_location = st.text_input('Input a file into which the data will be saved. Example: "/usr/home/my_raw_data.csv" (without the quotation marks)', value=config.OUTPUT_PATH)
            error_detected_raw = False
            if raw_data_save_location:
                if os.path.isdir(raw_data_save_location):
                    st.warning(f'File path submitted was of a folder. Submit a final path to a file.')
                    error_detected_raw = True
                    st.stop()
                if not os.path.isabs(raw_data_save_location):
                    st.warning(f'File path is not absolute. ')
                    error_detected_raw = True
                    st.stop()
                if not check_arg.is_pathname_valid(raw_data_save_location):
                    st.error(f'Bad path input. Check again. Path detected: "{raw_data_save_location}"')
                    error_detected_raw = True
                    st.stop()

            button_export_data_raw_confirmed = st.button('Export') if not error_detected_raw else False
            if button_export_data_raw_confirmed:
                if training_data_selec == raw:
                    df_train = p.df_features_train_raw
                elif training_data_selec == engineered:
                    df_train = p.df_features_train
                elif training_data_selec == scaled:
                    df_train = p.df_features_train_scaled
                else:
                    err = f'Invalid data selected ({training_data_selec}).'
                    logger.error(err)
                    raise ValueError(err)
                st.info(f'Saving file to "{raw_data_save_location}"...')
                with st.spinner(f'Trying to save training (raw) data to: {raw_data_save_location}...'):
                    df_train.to_csv(raw_data_save_location, index=None)
                st.success(f'Data file saved successfully to {raw_data_save_location}')
                st.info(f'This page will refresh automatically in {session[default_n_seconds_sleep]} seconds.')
                session[key_button_export_training_data] = False
                time.sleep(session[default_n_seconds_sleep])
                st.experimental_rerun()
        st.markdown('')
        st.markdown('----------------------------------------------')

    ### Export predict data
    button_export_predict_data = st.button(f'Toggle: export predicted data', key=key_button_export_predict_data)
    if button_export_predict_data:
        session[key_button_export_predict_data] = not session[key_button_export_predict_data]
    if session[key_button_export_predict_data]:

        predic_data_type_selec = st.selectbox('Select data set to save', ('', raw, engineered, scaled))
        if predic_data_type_selec:

            predic_data_save_location = st.text_input('Input a file path into which the data will be saved. Example: "/usr/home/my_raw_data.csv" (without the quotation marks)', value=config.OUTPUT_PATH)
            error_detected = False
            if predic_data_save_location:
                if os.path.isdir(predic_data_save_location):
                    st.warning(f'File path submitted was of a folder. Submit a final path to a file.')
                    error_detected = True
                    st.stop()
                if not os.path.isabs(predic_data_save_location):
                    st.warning(f'File path is not absolute. ')
                    error_detected = True
                    st.stop()
                if not check_arg.is_pathname_valid(predic_data_save_location):
                    st.error(f'Bad path input. Check again. Path detected: "{predic_data_save_location}"')
                    error_detected = True
                    st.stop()

            button_export_data_raw_confirmed = st.button('Export') if not error_detected else False
            if button_export_data_raw_confirmed:
                if predic_data_type_selec == raw:
                    df_predict = p.df_features_predict_raw
                elif predic_data_type_selec == engineered:
                    df_predict = p.df_features_predict
                elif predic_data_type_selec == scaled:
                    df_predict = p.df_features_predict_scaled
                else:
                    logging_enhanced.log_then_raise(f'Invalid data selected ({predic_data_type_selec}).', logger, ValueError)
                # st.info(f'Saving file to "{predic_data_save_location}"...')
                with st.spinner(f'Saving predict ({predic_data_type_selec}) data to: {predic_data_save_location}...'):
                    df_predict.to_csv(predic_data_save_location, index=None)
                st.success(f'Data file saved successfully to {predic_data_save_location}')
                st.info(f'This page will refresh automatically in {session[default_n_seconds_sleep]} seconds.')
                session[key_button_export_predict_data] = False
                time.sleep(session[default_n_seconds_sleep])
                st.experimental_rerun()
        st.markdown('')
        st.markdown('----------------------------------------------')


        # st.markdown('')

    ### End of menu for exporting predict data

        # # st.markdown('')
        # st.markdown(f'### Export training data section')
        # # 1/3: Export training data: raw
        # button_export_data_raw = st.button('Toggle: export raw train data', key=key_button_export_train_data_raw)
        # if button_export_data_raw:
        #     session[key_button_export_train_data_raw] = not session[key_button_export_train_data_raw]
        # if session[key_button_export_train_data_raw]:
        #     st.markdown('----------------------------------------------------------------------------------')
        #     st.markdown(f'#### Export raw training data')
        #     if len(p.df_features_train_raw) == 0:
        #         st.info(f'No raw training data available. Load up data first, then exporting that data will be available.')
        #     else:
        #         raw_data_save_location = st.text_input('Input a file into which the data will be saved. Example: "/usr/home/my_raw_data.csv" (without the quotation marks)', value=config.OUTPUT_PATH)
        #         error_detected_raw = False
        #         if raw_data_save_location:
        #             if os.path.isdir(raw_data_save_location):
        #                 st.error(f'File path submitted was of a folder. Submit a final path to a file.')
        #                 error_detected_raw = True
        #             if not os.path.isabs(raw_data_save_location):
        #                 st.error(f'File path is not absolute. ')
        #                 error_detected_raw = True
        #             if not check_arg.is_pathname_valid(raw_data_save_location):
        #                 st.error(f'Bad path input. Check again. Path detected: "{raw_data_save_location}"')
        #                 error_detected_raw = True
        #
        #         button_export_data_raw_confirmed = st.button('Export') if not error_detected_raw else False
        #         if button_export_data_raw_confirmed:
        #             st.info(f'Saving file to "{raw_data_save_location}"...')
        #             logger.info(f'Trying to save training (raw) data to: {raw_data_save_location}...')
        #             p.df_features_train_raw.to_csv(raw_data_save_location, index=None)
        #             session[key_button_export_training_data] = False
        #             session[key_button_export_train_data_raw] = False
        #             time.sleep(session[default_n_seconds_sleep])
        #             st.experimental_rerun()
        #     st.markdown('--------------------------------------------------------------------------------')
        #     st.markdown('')
        #     st.markdown('')
        #
        # # 2/3: Export training data: engineered
        # button_export_train_data_engineered = st.button('Toggle: export engineered training data', key=key_button_export_train_data_engineered)  # TODO: low: ensure button string well formed
        # if button_export_train_data_engineered:
        #     session[key_button_export_train_data_engineered] = not session[key_button_export_train_data_engineered]
        # if session[key_button_export_train_data_engineered]:
        #     st.markdown('--------------------------------------------------------------------------------')
        #     st.markdown(f'#### Export engineered training data')
        #     data_save_location = st.text_input('Input a full file path into which the data will be saved', value=config.OUTPUT_PATH)  # TODO: low: improve message string
        #     err_detected = False
        #     if data_save_location:
        #         if not os.path.isabs(data_save_location):
        #             warning_msg_is_abs = f'Save location ({data_save_location}) was not an absolute path.'
        #             st.warning(warning_msg_is_abs)
        #             logger.warning(warning_msg_is_abs)
        #             st.stop()
        #             err_detected = True
        #         if os.path.isdir(data_save_location):
        #             warning_msg_dir = f'Save location ({data_save_location}) is a directory and should not be. Specify an output file instead.'
        #             st.warning(warning_msg_dir)
        #             err_detected = True
        #         # If no errors, present save confirmation button
        #         button_export_data_confirmed = st.button('Export') if not err_detected else False
        #         if button_export_data_confirmed:
        #             st.info(f'Saving file to "{data_save_location}"')
        #             p.df_features_train.to_csv(data_save_location, index=None)
        #             session[key_button_export_train_data_engineered] = False
        #             session[key_button_export_training_data] = False
        #             st.success(f'Data saved to "{data_save_location}"')
        #             time.sleep(session[default_n_seconds_sleep])
        #             st.experimental_rerun()
        #     st.markdown('--------------------------------------------------------------------------------')
        #     st.markdown('')
        #     st.markdown('')
        #
        # # 3/3: Export training data: engineered, scaled, and directly used for model training
        # button_export_train_data_engineered_scaled = st.button('Toggle: export classifier-fed data', key=key_button_export_train_data_engineered_scaled)  # TODO: low: ensure button string well formed
        # if button_export_train_data_engineered_scaled:
        #     session[key_button_export_train_data_engineered_scaled] = not session[key_button_export_train_data_engineered_scaled]
        # if session[key_button_export_train_data_engineered_scaled]:
        #     st.markdown('--------------------------------------------------------------------------------')
        #     st.markdown(f'#### Export classifier-fed data')
        #     # TODO: check implementation
        #     data_scaled_save_location = st.text_input(f'Input a full file path into which the classifier-fed data will be saved', value=config.OUTPUT_PATH)
        #     err_scaled_data_save = False
        #     if data_scaled_save_location:
        #         if not os.path.isabs(data_scaled_save_location):
        #             err_scaled_data_save = True
        #             warning_msg_is_abs_scaled = f''
        #             st.info(warning_msg_is_abs_scaled)
        #         if os.path.isdir(data_scaled_save_location):
        #             err_scaled_data_save = True
        #         button_export_data_scaled_confirm = st.button('Export' if not err_scaled_data_save else False)
        #         if button_export_data_scaled_confirm:
        #             st.info(f'Saving file to "{data_scaled_save_location}"')
        #             p.df_features_train_scaled.to_csv(data_scaled_save_location, index=None)
        #             session[key_button_export_training_data] = False
        #             session[key_button_export_train_data_engineered_scaled] = False
        #             st.success(f'Data saved to "{data_scaled_save_location}"')
        #             time.sleep(session[default_n_seconds_sleep])
        #             st.experimental_rerun()
        #     st.markdown('--------------------------------------------------------------------------------')
        #     st.markdown('')
        #     st.markdown('')


    ### End of menu for exporting training data

    # st.markdown('')

    ### Menu: Export predict data



    # button_export_predict_data = st.button('Toggle: export predict data', key=key_button_export_predict_data)
    # if button_export_predict_data:
    #     session[key_button_export_predict_data] = not session[key_button_export_predict_data]
    # if session[key_button_export_predict_data]:
    #     st.markdown(f'### Export predict data')
    #
    #     selected_data = st.selectbox('labelgoeshere', (raw, engineered, scaled))
    #
    #
    #     # 1/2: export raw data
    #     button_export_data_raw = st.button('Toggle: export raw predict data', key=key_button_export_predict_data_raw)
    #     if button_export_data_raw:
    #         session[key_button_export_predict_data_raw] = not session[key_button_export_predict_data_raw]
    #     if session[key_button_export_predict_data_raw]:
    #         st.markdown(f'')
    #     # TODO: add a button
    #     # 2/2: Export engineered feature data but with labels
    #     button_export_predict_data_engineered = st.button('Toggle: export engineered, predicted predict data', key=key_button_export_predict_data_engineered)
    #     if button_export_predict_data_engineered:
    #         session[key_button_export_predict_data_engineered] = not session[key_button_export_predict_data_engineered]
    #     if session[key_button_export_predict_data_engineered]:
    #         st.markdown(f'')
    #
    #     # Export engineered, scaled feature data
    #     button_export_predict_data_engineered_scaled = st.button('Toggle: export `classifier-fed` data', key=key_button_export_predict_data_engineered_scaled)
    #     if button_export_predict_data_engineered_scaled:
    #         session[key_button_export_predict_data_engineered_scaled] = not session[key_button_export_predict_data_engineered_scaled]
    #     if session[key_button_export_predict_data_engineered_scaled]:
    #         st.markdown(f'')

    return display_footer(p, pipeline_file_path)


def display_footer(p: pipeline.BasePipeline, pipeline_file_path, *args, **kwargs):
    """ Footer of Streamlit page """
    # TODO: low: consider: add link to GitHub page and/or credits?
    st.markdown('')
    logger.debug('   < End of streamlit page >   ')
    return p


# Accessory functions #

def get_video_bytes(path_to_video):
    check_arg.ensure_is_file(path_to_video)
    with open(path_to_video, 'rb') as video_file:
        video_bytes = video_file.read()
    return video_bytes


# Misc.

def st_file_selector(st_placeholder, label='', path='.'):
    """
    TODO: NOTE: THIS FUNCTION DOES NOT CURRENTLY WORK!!! wip
    """
    # get base path (directory)
    logger.debug(f'st_file_selector(): label = {label} / path = {path}')
    initial_base_path = '.' if not path else path
    # If initial base path is a file, get the directory
    base_path = initial_base_path if os.path.isdir(initial_base_path) else os.path.dirname(initial_base_path)

    base_path = '.' if not base_path else base_path
    logger.debug(f'st_file_selector(): base_path finally resolves to: {base_path}')

    # list files in base path directory
    files: List[str] = ['.', '..', ] + os.listdir(base_path)
    # logger.debug(f'st_file_selector(): files list: {files} ')

    # Create select box
    selected_file = st_placeholder.selectbox(label=label, options=files, key=base_path+str(random.randint(0, 1000)))
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    logger.debug(f'st_file_selector(): ')

    if selected_file == '.':
        logger.debug(f'st_file_selector(): selected_file = {selected_file}')
        return selected_path
    if selected_file == '..':
        logger.debug(f'st_file_selector(): SELECTED PATH = {selected_path}')
        selected_path = st_file_selector(st_placeholder=st_placeholder,
                                         path=os.path.dirname(base_path),
                                         label=label)
    if os.path.isdir(selected_path):
        logger.debug(f'st_file_selector(): SELECTED PATH = {selected_path}')
        selected_path = st_file_selector(st_placeholder=st_placeholder,
                                         path=selected_path,
                                         label=label)
    return selected_path


def example_of_value_saving():
    """ A toy function for showing patterns of saving button presses/values """
    session_state = streamlit_session_state.get(**{'TestButton1': False, 'TestButton2': False})
    st.markdown("# [Title]")
    button1_is_clicked = st.button('Test Button 1', 'TestButton1')
    st.markdown(f'Pre button1: Button 1 session state: {session_state["TestButton1"]}')
    if button1_is_clicked:
        session_state['TestButton1'] = not session_state['TestButton1']
    if session_state['TestButton1']:
        st.markdown(f'In button1: Button 1 session state: {session_state["TestButton1"]}')
        button2_is_clicked = st.button('Test Button 2', 'TestButton2')
        if button2_is_clicked:
            session_state['TestButton2'] = not session_state['TestButton2']
        if session_state['TestButton2']:
            st.markdown('----------------------------')
            st.markdown('Button 2 pressed')


def checking_file_session_timings():
    space1 = st.empty()
    space2 = st.empty()
    iters = 1000
    global session
    session = streamlit_session_state.get(**streamlit_persistence_variables)
    start1 = time.perf_counter()
    for i in range(iters):
        x = session[checkbox_show_extra_text]
        st.markdown(x)
    end1 = time.perf_counter()

    start2 = time.perf_counter()
    y = session[checkbox_show_extra_text]
    for i in range(iters):
        x = y
        st.markdown(x)
    end2 = time.perf_counter()
    space1.write(f'File checking time: {end1-start1}')
    space2.write(f'file checkimt time 2: {end2-start2}')
    pass


# Main: only use for debugging. It will be deleted later.

if __name__ == '__main__':
    # Note: this import only necessary when running streamlit onto this file specifically rather than
    #   calling `streamlit run main.py streamlit`
    DIBS_project_path = os.path.dirname(os.path.dirname(__file__))
    if DIBS_project_path not in sys.path:
        sys.path.insert(0, DIBS_project_path)
    start_app()
    # checking_file_session_timings()

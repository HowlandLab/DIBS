"""
Every function in this file is an entire runtime sequence (app) encapsulated. Expect nothing to be returned.
"""
from typing import List
import inspect
import itertools

import os
import time

from dibs import config, logging_enhanced, pipeline, streamlit_app


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
    """
    Entry point for the Streamlit companion app.
    """
    # streamlit_app.header(**kwargs)
    streamlit_app.start_app(**kwargs)


def tsnegridsearch():
    # Param section -- MAGIC VARIABLES GO HERE
    perplexity_fracs = [0.001, ]
    exaggerations = [500, ]
    learn_rates = [100, ]
    percent_epm_train_files_to_cluster_on = 0.1
    assert 0 < percent_epm_train_files_to_cluster_on <= 1.0


    pipeline_implementation = pipeline.PipelineHowland  # Another option includes dibs.pipeline.PipelineMimic
    num_gmm_clusters_aka_num_colours = 7  # Sets the number of clusters that GMM will try to label

    ### Diagnostics parameters (graphing) ###
    show_cluster_graphs_in_a_popup_window = False  # Set to False to display graphs inline
    graph_dimensions = (10, 10)  # length x width.

    # Auto-generate the product between all possible parameters
    kwargs_product = [{
        'tsne_perplexity': perplexity_i,
        'tsne_early_exaggeration': early_exaggeration_j,
        'tsne_learning_rate': learning_rate_k,
        'gmm_n_components': num_gmm_clusters_aka_num_colours,
        'tsne_n_components': 2,  # n-D dimensionality reduction

        'cross_validation_k': 2,
        'cross_validation_n_jobs': 2,
        'rf_n_jobs': 2,
        'tsne_n_jobs': 2,
    } for learning_rate_k, early_exaggeration_j, perplexity_i in itertools.product(
        learn_rates,
        exaggerations,
        [f'lambda self: self.num_training_data_points * {f}' for f in perplexity_fracs],
    )]
    pipeline_names_by_index = [f'Pipeline_{i}' for i in range(len(kwargs_product))]
    # print('Number of parameter permutations:', len(kwargs_product))

    # Queue up which data files will be added to each Pipeline
    all_files = [os.path.join(config.DEFAULT_TRAIN_DATA_DIR, file) for file in
                 os.listdir(config.DEFAULT_TRAIN_DATA_DIR)]
    train_data = half_files = all_files[:int(len(all_files) * percent_epm_train_files_to_cluster_on)]
    # print(train_data)  # Uncomment this line to see which exact data files are added to the Pipeline

    # Create list of pipelines with all of the different combinations of parameters inserted
    pipelines_ready_for_building = [pipeline_implementation(name, **kwargs).add_train_data_source(*train_data) for
                                    name, kwargs in zip(pipeline_names_by_index, kwargs_product)]

    # The heavy lifting/processing is done here
    results_current_time = time.strftime("%Y-%m-%d_%HH%MM")
    print(f'Start time: {results_current_time}')
    start_time = time.perf_counter()
    print(len(pipelines_ready_for_building))
    for i, p_i in enumerate(pipelines_ready_for_building):
        print(f'START {i}: Pipeline={p_i.name} / Frac={p_i._tsne_perplexity}')
        try:
            pipelines_ready_for_building[i] = p_i.build()
        except Exception as e:
            info = f'PerpRaw={p_i._tsne_perplexity}/Perp={p_i.tsne_perplexity}/EE={p_i.tsne_early_exaggeration}/LR={p_i.tsne_learning_rate}/GMM-N={p_i.gmm_n_components}'
            err = f'app.{logging_enhanced.get_current_function()}(): an unexpected exception occurred when building many pipelines to get good graphs. Info is as follows: {info}. Exception is: {repr(e)}'
            logger.error(err)
        else:
            # Save graph to file
            perplexity_ratio_i, perplexity_i, learning_rate_i, early_exaggeration_i = p_i.tsne_perplexity_relative_to_num_data_points, p_i.tsne_perplexity, p_i.tsne_learning_rate, p_i.tsne_early_exaggeration

            title = f"Perp ratio: {round(perplexity_ratio_i, 5)} / Perp: {perplexity_i} / EE: {early_exaggeration_i} / LearnRate: {learning_rate_i} "

            p_i.plot_clusters_by_assignments(
                title=title,
                fig_file_prefix=f'{results_current_time}__{p_i.name}__',
                show_now=False, save_to_file=True, figsize=graph_dimensions,
                s=0.4 if show_cluster_graphs_in_a_popup_window else 1.5,
            )

        print('--------------------------\n\n')
    end_time = time.perf_counter()
    print(f'Total compute time: {round((end_time - start_time) / 60, 2)} minutes.')
    done_time = time.strftime("%Y-%m-%d_%HH%MM")
    print(f'Done job at: {done_time}')


# Sample function

def sample_runtime_function(sleep_secs=3, *args, **kwargs):
    """ Sample function that takes n seconds to run. Used for debugging. """
    logger.debug(f'{logging_enhanced.get_current_function()}(): '
                 f'Doing sample runtime execution for {sleep_secs} seconds.')
    time.sleep(sleep_secs)
    return




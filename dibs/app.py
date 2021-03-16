"""
Every function in this file is an entire runtime sequence (app) encapsulated. Expect nothing to be returned.
"""
from typing import List
import inspect
import itertools

import os
import sys
import time

from dibs import config, logging_enhanced, pipeline


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

    # streamlit_app.start_app(**kwargs)
    pass


def tsnegridsearch():
    # Param section -- MAGIC VARIABLES GO HERE
    perplexity_fracs = [1e-4, 1e-3, 5e-3, 1e-2, ]
    perplexity_fracs = [1e-2, 1e-1, 0.2,0]
    perplexities = list(range(500, 750, 50))
    exaggerations = list(range(50, 1100, 100))[::-1]
#     exaggerations = [200, ]
    learn_rates = list(range(100, 1_000, 200))
#     learn_rates = [100, ]

    # tsne_n_iters = [1_000, 1_500, 2_000, ]
    tsne_n_iters = [1_000, ]
    percent_epm_train_files_to_cluster_on = 1.0
    assert 0 < percent_epm_train_files_to_cluster_on <= 1.0


    pipeline_implementation = pipeline.PipelineHowland  # Another option includes dibs.pipeline.PipelineMimic
    num_gmm_clusters_aka_num_colours = 7  # Sets the number of clusters that GMM will try to label

    ### Diagnostics parameters (graphing) ###
    show_cluster_graphs_in_a_popup_window = False  # Set to False to display graphs inline
    graph_dimensions = (15, 15)  # length x width.
    max_cores_per_pipe = 8
    # Auto-generate the product between all possible parameters
    kwargs_product = [{
        'tsne_perplexity': perplexity_i,
        'tsne_early_exaggeration': early_exaggeration_j,
        'tsne_learning_rate': learning_rate_k,
        'gmm_n_components': num_gmm_clusters_aka_num_colours,
        'tsne_n_components': 2,  # n-D dimensionality reduction
        'tsne_n_iter': tsne_n_iter,
        'cross_validation_k': max_cores_per_pipe,
        'cross_validation_n_jobs': max_cores_per_pipe,
        'rf_n_jobs': max_cores_per_pipe,
        'tsne_n_jobs': max_cores_per_pipe,
    } for learning_rate_k, early_exaggeration_j, perplexity_i, tsne_n_iter in itertools.product(
        learn_rates,
        exaggerations,
        perplexities,
        #[f'lambda self: self.num_training_data_points * {f}' for f in perplexity_fracs],
        tsne_n_iters,
    )]
    pipeline_names_by_index = [f'Pipeline_{i}' for i in range(len(kwargs_product))]
    logger.info(f'Number of parameter permutations: {len(kwargs_product)}')

    # Queue up which data files will be added to each Pipeline
    all_files = [os.path.join(config.DEFAULT_TRAIN_DATA_DIR, file) for file in os.listdir(config.DEFAULT_TRAIN_DATA_DIR)]
    train_data = all_files[:int(len(all_files) * percent_epm_train_files_to_cluster_on)]

    # print(train_data)  # Uncomment this line to see which exact data files are added to the Pipeline

    # Create list of pipelines with all of the different combinations of parameters inserted
    # pipelines_ready_for_building = [pipeline_implementation(name, **kwargs).add_train_data_source(*(train_data.copy())) for name, kwargs in zip(pipeline_names_by_index, kwargs_product)]

    # The heavy lifting/processing is done here
    print(f'# of combinations: {len(pipeline_names_by_index)}')
    print(f'Start time: {time.strftime("%Y-%m-%d_%HH%MM")}')

    start_time = time.perf_counter()
    for i, kwargs_i in enumerate(kwargs_product):
        results_current_time = time.strftime("%Y-%m-%d_%HH%MM")
        p_i: pipeline.BasePipeline = pipeline_implementation(f'{pipeline_names_by_index[i]}_{results_current_time}', **kwargs_i).add_train_data_source(*(train_data.copy()))
        print(f'Start build for pipeline idx {i} -- Frac={p_i._tsne_perplexity}')
        try:
            p_i = p_i.build(skip_accuracy_score=True)
        except Exception as e:
            info = f'PerpRaw={p_i._tsne_perplexity}/Perp={p_i.tsne_perplexity}/EE={p_i.tsne_early_exaggeration}/LR={p_i.tsne_learning_rate}/GMM-N={p_i.gmm_n_components}'
            err = f'app.{logging_enhanced.get_current_function()}(): an unexpected exception occurred when building many pipelines to get good graphs. Info is as follows: {info}. Exception is: {repr(e)}'
            logger.error(err)
        else:
            # Save graph to file
            perplexity_ratio_i, perplexity_i, learning_rate_i, early_exaggeration_i = p_i.tsne_perplexity_relative_to_num_data_points, p_i.tsne_perplexity, p_i.tsne_learning_rate, p_i.tsne_early_exaggeration

            title = f"Perp ratio: {round(perplexity_ratio_i, 5)} / Perp: {perplexity_i} / EE: {early_exaggeration_i} / LearnRate: {learning_rate_i} / #data={p_i.num_training_data_points}"

            p_i.plot_clusters_by_assignments(
                title=title,
                fig_file_prefix=f'{results_current_time}__{p_i.name}__',
                show_now=False, save_to_file=True, figsize=graph_dimensions,
                s=0.4 if show_cluster_graphs_in_a_popup_window else 1.5,
            )
        print('--------------------------\n\n')
    end_time = time.perf_counter()
    print(f'Total compute time: {round((end_time - start_time) / 60, 2)} minutes.')
    print(f'Total jobs completed: {len(pipeline_names_by_index)}')
    done_time = time.strftime("%Y-%m-%d_%HH%MM")
    print(f'Done job at: {done_time}')


def print_if_system_is_64_bit():
    print(f'This system is detected to be 64-bit: {sys.maxsize > 2**32}')


# Sample function

def sample_runtime_function(sleep_secs=3, *args, **kwargs):
    """ Sample function that takes n seconds to run. Used for debugging. """
    logger.debug(f'{logging_enhanced.get_current_function()}(): '
                 f'Doing sample runtime execution for {sleep_secs} seconds.')
    time.sleep(sleep_secs)
    return




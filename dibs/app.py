"""
Every function in this file is an entire runtime sequence (app) encapsulated. Expect nothing to be returned.
"""
from typing import List
import inspect
import itertools
import os
import time
import sys

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
    max_cores_per_pipe = 8
    num_gmm_clusters_aka_num_colours = 7  # Sets the number of clusters that GMM will try to label
    pipeline_implementation = pipeline.PipelineHowland  # Another option includes dibs.pipeline.PipelineMimic
    graph_dimensions = (15, 15)  # length x width.
    show_cluster_graphs_in_a_popup_window = False  # Set to False to display graphs inline

    # perplexity_fracs = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    perplexities = [200, 400, 600, 800]
    # perplexities = [200, ]
    exaggerations = [200, 400, 600, 800, 1000]
    # learn_rates = [100, 200, 400, ]
    learn_rates = [500, 1000, 1500]
    # tsne_n_iters = [1_000, 2000, 3000]
    tsne_n_iters = [1000, ]
    percent_epm_train_files_to_cluster_on = 1.0

    assert 0 < percent_epm_train_files_to_cluster_on <= 1.0

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
        'svm_n_jobs': max_cores_per_pipe,
    } for learning_rate_k, early_exaggeration_j, perplexity_i, tsne_n_iter in itertools.product(
        learn_rates,
        exaggerations,
        perplexities,
        # [f'lambda self: self.num_training_data_points * {f}' for f in perplexity_fracs],
        tsne_n_iters,
    )]
    pipeline_names_by_index = [f'Pipeline_{i}' for i in range(len(kwargs_product))]
    logger.info(f'Number of parameter permutations: {len(kwargs_product)} (starting runtime at: {time.strftime("%Y-%m-%d_%HH%MM")})')

    # Queue up which data files will be added to each Pipeline
    all_files = [os.path.join(config.DEFAULT_TRAIN_DATA_DIR, file) for file in os.listdir(config.DEFAULT_TRAIN_DATA_DIR)]
    train_data = all_files[:int(len(all_files) * percent_epm_train_files_to_cluster_on)]

    # print(train_data)  # Uncomment this line to see which exact data files are added to the Pipeline

    # Create list of pipelines with all of the different combinations of parameters inserted
    # pipelines_ready_for_building = [pipeline_implementation(name, **kwargs).add_train_data_source(*(train_data.copy())) for name, kwargs in zip(pipeline_names_by_index, kwargs_product)]

    # The heavy lifting/processing is done here
    logger.info(f'# of combinations: {len(pipeline_names_by_index)}')
    logger.debug(f'Start time: {time.strftime("%Y-%m-%d_%HH%MM")}')
    successful_builds = 0
    start_time = time.perf_counter()
    for i, kwargs_i in enumerate(kwargs_product):
        start_build = time.perf_counter()
        results_current_time = time.strftime("%Y-%m-%d_%HH%MM")
        p_i: pipeline.BasePipeline = pipeline_implementation(f'{pipeline_names_by_index[i]}_{results_current_time}', **kwargs_i).add_train_data_source(*(train_data.copy()))
        logger.debug(f'Start build for pipeline idx {i} ({i+1} of {len(kwargs_product)})  -- Frac={p_i._tsne_perplexity}')
        try:
            p_i = p_i.build(skip_accuracy_score=True)
        except Exception as e:
            info = f'PerpRaw={p_i._tsne_perplexity}/Perp={p_i.tsne_perplexity}/' \
                   f'EE={p_i.tsne_early_exaggeration}/LR={p_i.tsne_learning_rate}/GMM-N={p_i.gmm_n_components}'
            err = f'Unexpected exception::{__name__}.{logging_enhanced.get_current_function()}(): ' \
                  f'an unexpected exception occurred when building many pipelines to get good graphs. ' \
                  f'Info is as follows: {info}. Exception is: {repr(e)}.'
            try:
                diag = p_i.diagnostics()
            except:
                pass
            else:
                err += f'Diagnostics: {diag}'

            logger.error(err)
            logger.debug(f'Failed to build iteration {(i+1)/len(kwargs_product)}')
        else:
            # Save graph to file
            perplexity_ratio_i, perplexity_i, learning_rate_i, early_exaggeration_i = p_i.tsne_perplexity_relative_to_num_data_points, p_i.tsne_perplexity, p_i.tsne_learning_rate, p_i.tsne_early_exaggeration

            graph_title = f"Perp ratio: {round(perplexity_ratio_i, 4)} / " \
                          f"Perp: {perplexity_i} / EE: {early_exaggeration_i} / " \
                          f"LearnRate: {learning_rate_i} / #data={p_i.num_training_data_points} / tsneNiter={p_i.tsne_n_iter}"
            # Save graph to file
            p_i.plot_clusters_by_assignments(
                title=graph_title,
                fig_file_prefix=f'newweek_{p_i.name}__{results_current_time}__',
                show_now=False, save_to_file=True, figsize=graph_dimensions,
                s=0.4 if show_cluster_graphs_in_a_popup_window else 1.5,
            )
            logger.debug(f'Finished successfully building iteration: {(i+1)/len(kwargs_product)}')
        end_build = time.perf_counter()
        logger.info(f'Time to build: {round(end_build-start_build)} seconds (using {max_cores_per_pipe} cores)')
        successful_builds += 1
        logger.debug('---------------------------------------------\n\n')
    end_time = time.perf_counter()
    logger.info(f'Total compute time: {round((end_time - start_time) / 60, 2)} minutes. Total successful jobs with results: {successful_builds}. Total jobs computed: {len(pipeline_names_by_index)}')
    logger.debug(f'Done job at: {time.strftime("%Y-%m-%d_%HH%MM")}')


def print_if_system_is_64_bit():
    print(f'This system is detected to be 64-bit: {sys.maxsize > 2**32}')


def trybuild(*args, **kwargs):
    # def make_behaviour_example_videos(self, data_source: str, video_file_path: str, file_name_prefix=None, min_rows_of_behaviour=1, max_examples=3, num_frames_buffer=0, output_fps=15):
    vid_data_source = 'EPM-MCE-10DLC_resnet50_Maternal_EPMDec28shuffle1_700000'
    vid_file_path = "C:\\Users\\killian\\projects\\DIBS\\epm-mce-vids-f\\EPM-MCE-10.mp4"
    assert os.path.isfile(vid_file_path)
    # Param section -- MAGIC VARIABLES GO HERE
    max_cores_per_pipe = 8
    perplexities = [400, 400]
    exaggerations = [1000]
    learn_rates = [1000, 1000, 1000, 1000, 1000]
    tsne_n_iters = [250]
    percent_epm_train_files_to_cluster_on = 0.5

    num_gmm_clusters_aka_num_colours = 7  # Sets the number of clusters that GMM will try to label
    pipeline_implementation = pipeline.PipelineHowland  # Another option includes dibs.pipeline.PipelineMimic
    graph_dimensions = (12, 12)  # length x width.
    show_cluster_graphs_in_a_popup_window = False  # Set to False to display graphs inline
    assert 0 < percent_epm_train_files_to_cluster_on <= 1.0

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
        # [f'lambda self: self.num_training_data_points * {f}' for f in perplexity_fracs],
        tsne_n_iters,
    )]
    pipeline_names_by_index = [f'Pipeline_{i}' for i in range(len(kwargs_product))]
    logger.info(f'Number of parameter permutations: {len(kwargs_product)} (starting runtime at: {time.strftime("%Y-%m-%d_%HH%MM")})')

    # Queue up which data files will be added to each Pipeline
    all_files = [os.path.join(config.DEFAULT_TRAIN_DATA_DIR, file) for file in os.listdir(config.DEFAULT_TRAIN_DATA_DIR)]
    train_data = all_files[:int(len(all_files) * percent_epm_train_files_to_cluster_on)]

    # print(train_data)  # Uncomment this line to see which exact data files are added to the Pipeline

    # Create list of pipelines with all of the different combinations of parameters inserted
    # pipelines_ready_for_building = [pipeline_implementation(name, **kwargs).add_train_data_source(*(train_data.copy())) for name, kwargs in zip(pipeline_names_by_index, kwargs_product)]

    # The heavy lifting/processing is done here
    logger.info(f'# of combinations: {len(pipeline_names_by_index)}')
    logger.debug(f'Start time: {time.strftime("%Y-%m-%d_%HH%MM")}')
    successful_builds = 0
    start_time = time.perf_counter()
    for i, kwargs_i in enumerate(kwargs_product):
        start_build = time.perf_counter()
        results_current_time = time.strftime("%Y-%m-%d_%HH%MM")
        p_i: pipeline.BasePipeline = pipeline_implementation(f'{pipeline_names_by_index[i]}_{results_current_time}', **kwargs_i).add_train_data_source(*(train_data.copy()))
        logger.debug(f'Start build for pipeline idx {i} ({i+1} of {len(kwargs_product)})  -- Frac={p_i._tsne_perplexity}')
        # Try to build
        try:
            p_i = p_i.build(skip_accuracy_score=True)
        except Exception as e:
            info = f'PerpRaw={p_i._tsne_perplexity}/Perp={p_i.tsne_perplexity}/' \
                   f'EE={p_i.tsne_early_exaggeration}/LR={p_i.tsne_learning_rate}/GMM-N={p_i.gmm_n_components}'
            err = f'Unexpected exception::{__name__}.{logging_enhanced.get_current_function()}(): ' \
                  f'an unexpected exception occurred when building many pipelines to get good graphs. ' \
                  f'Info is as follows: {info}. Exception is: {repr(e)}.'# Diagnostics: {p_i.diagnostics()}'
            logger.error(err)

            raise RuntimeError(err)
        else:
            # Get info
            perplexity_ratio_i, perplexity_i, learning_rate_i, early_exaggeration_i = p_i.tsne_perplexity_relative_to_num_data_points, p_i.tsne_perplexity, p_i.tsne_learning_rate, p_i.tsne_early_exaggeration

            graph_title = f"Perp ratio: {round(perplexity_ratio_i, 5)} / " \
                          f"Perp: {perplexity_i} / EE: {early_exaggeration_i} / " \
                          f"LearnRate: {learning_rate_i} / #data={p_i.num_training_data_points} / tsneNiter={p_i.tsne_n_iter}"
            # Save graph to file
            p_i.plot_clusters_by_assignments(
                title=graph_title,
                fig_file_prefix=f'standardscaler1{results_current_time}__{p_i.name}__',
                show_now=False, save_to_file=True, figsize=graph_dimensions,
                s=0.4 if show_cluster_graphs_in_a_popup_window else 1.5,
            )
            # p_i.save_to_folder()
            # Try to make example vids
            # p_i.make_behaviour_example_videos(vid_data_source, vid_file_path, file_name_prefix='timtest4', num_frames_buffer=1)  #     def make_behaviour_example_videos(self, data_source: str, video_file_path: str, file_name_prefix=None, min_rows_of_behaviour=1, max_examples=3, num_frames_buffer=0, output_fps=15):

        end_build = time.perf_counter()
        logger.info(f'Time to build: {round(end_build-start_build)} seconds (using {max_cores_per_pipe} cores)')
        successful_builds += 1
        logger.debug('---------------------------------------------\n\n')
    end_time = time.perf_counter()
    logger.info(f'Total compute time: {round((end_time - start_time) / 60, 2)} minutes. Total successful jobs with results: {successful_builds}. Total jobs computed: {len(pipeline_names_by_index)}')
    logger.debug(f'Done job at: {time.strftime("%Y-%m-%d_%HH%MM")}')


def buildone(pipeline_name, **kwargs):
    # Param section -- MAGIC VARIABLES GO HERE
    # vid_data_source = 'EPM-MCE-10DLC_resnet50_Maternal_EPMDec28shuffle1_700000'
    # vid_file_path = "C:\\Users\\killian\\projects\\DIBS\\epm-mce-vids-f\\EPM-MCE-10.mp4"
    # assert os.path.isfile(vid_file_path), f'Video file not found. Path = {vid_file_path}'

    max_cores_per_pipe = 4
    percent_epm_train_files_to_cluster_on = 1.0
    num_gmm_clusters_aka_num_colours = 7  # Sets the number of clusters that GMM will try to label
    pipeline_implementation = pipeline.PipelineHowland  # Another option includes dibs.pipeline.PipelineMimic
    graph_dimensions = (12, 12)  # length x width.
    show_cluster_graphs_in_a_popup_window = False  # Set to False to display graphs inline
    assert 0 < percent_epm_train_files_to_cluster_on <= 1.0

    # Auto-generate the product between all possible parameters
    pipeline_kwargs = {
        'tsne_perplexity': kwargs.get('perplexity', config.TSNE_PERPLEXITY),
        'tsne_early_exaggeration': kwargs.get('early_exaggeration', config.TSNE_EARLY_EXAGGERATION),
        'tsne_learning_rate': kwargs.get('learning_rate', config.TSNE_LEARNING_RATE),
        'gmm_n_components': kwargs.get('gmm_n_components', config.gmm_n_components),
        'tsne_n_components': 2,  # n-D dimensionality reduction
        'tsne_n_iter': kwargs.get('tsne_n_iter', ),
        'cross_validation_k': max_cores_per_pipe,
        'cross_validation_n_jobs': max_cores_per_pipe,
        'rf_n_jobs': max_cores_per_pipe,
        'tsne_n_jobs': max_cores_per_pipe,
    }

    # Queue up which data files will be added to each Pipeline
    all_files_paths = [os.path.join(config.DEFAULT_TRAIN_DATA_DIR, file) for file in os.listdir(config.DEFAULT_TRAIN_DATA_DIR)]
    actual_used_training_data_files_paths = all_files_paths[:int(len(all_files_paths) * percent_epm_train_files_to_cluster_on)]
    # print(train_data)  # Uncomment this line to see which exact data files are added to the Pipeline

    logger.debug(f'{logging_enhanced.get_current_function()}(): Start time: {time.strftime("%Y-%m-%d_%HH%MM")}')
    start_build = time.perf_counter()
    results_current_time = time.strftime("%Y-%m-%d_%HH%MM")
    p_i: pipeline.BasePipeline = pipeline_implementation(f'{pipeline_name}_{results_current_time}', **pipeline_kwargs).add_train_data_source(*actual_used_training_data_files_paths)

    # Try to build
    try:
        p_i = p_i.build(skip_accuracy_score=True)
    except Exception as e:
        info = f'PerpRaw={p_i._tsne_perplexity}/Perp={p_i.tsne_perplexity}/' \
               f'EE={p_i.tsne_early_exaggeration}/LR={p_i.tsne_learning_rate}/GMM-N={p_i.gmm_n_components}'
        err = f'Unexpected exception::{__name__}.{logging_enhanced.get_current_function()}(): ' \
              f'an unexpected exception occurred when building many pipelines to get good graphs. ' \
              f'Info is as follows: {info}. Exception is: {repr(e)}.'  # Diagnostics: {p_i.diagnostics()}'
        logger.error(err)
        raise RuntimeError(err)
    else:
        # Get info
        perplexity_ratio_i, perplexity_i, learning_rate_i, early_exaggeration_i = p_i.tsne_perplexity_relative_to_num_data_points, p_i.tsne_perplexity, p_i.tsne_learning_rate, p_i.tsne_early_exaggeration

        graph_title = f"Perp ratio: {round(perplexity_ratio_i, 5)} / " \
                      f"Perp: {perplexity_i} / EE: {early_exaggeration_i} / " \
                      f"LearnRate: {learning_rate_i} / #data={p_i.num_training_data_points} / tsneNiter={p_i.tsne_n_iter}"
        # Save graph to file
        p_i.plot_clusters_by_assignments(
            title=graph_title,
            fig_file_prefix=f'standardscaler1{results_current_time}__{p_i.name}__',
            show_now=False, save_to_file=True, figsize=graph_dimensions,
            s=0.4 if show_cluster_graphs_in_a_popup_window else 1.5,
        )
        # p_i.save_to_folder()
        # Try to make example vids
        # p_i.make_behaviour_example_videos(vid_data_source, vid_file_path, file_name_prefix='timtest4', num_frames_buffer=1)  #     def make_behaviour_example_videos(self, data_source: str, video_file_path: str, file_name_prefix=None, min_rows_of_behaviour=1, max_examples=3, num_frames_buffer=0, output_fps=15):

    end_build = time.perf_counter()
    build_time_secs = round((end_build-start_build), 2)
    logger.info(f'Time to build: {build_time_secs} seconds (using {max_cores_per_pipe} cores)')
    logger.debug(f'Done job at: {time.strftime("%Y-%m-%d_%HH%MM")}')

"""
Every function in this file is an entire runtime sequence (app) encapsulated. Expect nothing to be returned.
"""
from typing import List
import inspect
import itertools
import os
import time
import sys

from dibs import check_arg, config, logging_enhanced, pipeline, streamlit_app


logger = config.initialize_logger(__name__)


### App

def streamlit(*args, **kwargs) -> None:
    """
    Entry point for the Streamlit companion app.
    """
    # streamlit_app.header(**kwargs)
    streamlit_app.start_app(**kwargs)


### Command-line runnables

def tsnegridsearch(**kwargs):
    # Param section -- MAGIC VARIABLES GO HERE
    percent_epm_train_files_to_cluster_on = 1.0
    max_cores_per_pipe = 8
    num_gmm_clusters_aka_num_colours = 8  # Sets the number of clusters that GMM will try to label
    pipeline_implementation = pipeline.PipelineHowlandLLE  # Another option includes dibs.pipeline.PipelineMimic
    graph_dimensions = (15, 15)  # length x width.
    show_cluster_graphs_in_a_popup_window = False  # Set to False to display graphs inline

    # perplexity_fracs = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    perplexities = [200, 400, 600, 800]
    perplexities = [400, ]
    # perplexities = [200, ]
    exaggerations = [200, 400, 600, 800, 1000]
    exaggerations = [800, ]
    # learn_rates = [100, 200, 400, ]
    learn_rates = [500, 1000, 1500]
    learn_rates = [1000, ]
    # tsne_n_iters = [1_000, 2000, 3000]
    tsne_n_iters = [1000, ]
    # umap_neighbors = [5, 50, 100, ]
    umap_neighbors = [5, ]

    lle_methods = ['standard', 'hessian', 'modified', 'ltsa']
    LLE_n_neighbors = [5, 50, 100, ][::-1]

    assert 0 < percent_epm_train_files_to_cluster_on <= 1.0

    # Auto-generate the product between all possible parameters
    kwargs_product = [{
        'tsne_perplexity': perplexity_i,
        'tsne_early_exaggeration': early_exaggeration_j,
        'tsne_learning_rate': learning_rate_k,
        'tsne_n_iter': tsne_n_iter,

        'umap_n_neighbors': umap_n_neigh,
        'LLE_method': lle_method,
        'LLE_n_neighbors': LLE_n_neighbor,

        'gmm_n_components': num_gmm_clusters_aka_num_colours,
        'tsne_n_components': 2,  # n-D dimensionality reduction

        'cross_validation_k': max_cores_per_pipe,
        'cross_validation_n_jobs': max_cores_per_pipe,
        'rf_n_jobs': max_cores_per_pipe,
        'tsne_n_jobs': max_cores_per_pipe,
        'svm_n_jobs': max_cores_per_pipe,


    } for learning_rate_k, early_exaggeration_j, perplexity_i, tsne_n_iter, umap_n_neigh, lle_method, LLE_n_neighbor in itertools.product(
        learn_rates,
        exaggerations,
        perplexities,
        # [f'lambda self: self.num_training_data_points * {f}' for f in perplexity_fracs],
        tsne_n_iters,
        umap_neighbors,
        lle_methods,
        LLE_n_neighbors,
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
    logger.info(f'Gridsearch: # of combinations: {len(pipeline_names_by_index)}')
    logger.debug(f'Start time: {time.strftime("%Y-%m-%d_%HH%MM")}')
    start_time_str_start = time.strftime("%Y-%m-%d_%HH%MM")
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
            graph_filename_prefix = f'{p_i.name}__LLE_GRIDTEST1__method_{p_i.LLE_method}__LLENEIGHBORS_{p_i.LLE_n_neighbors}__{start_time_str_start}__'
            successful_builds += 1
            # Save graph to file
            perplexity_ratio_i, perplexity_i, learning_rate_i, early_exaggeration_i = p_i.tsne_perplexity_relative_to_num_data_points, p_i.tsne_perplexity, p_i.tsne_learning_rate, p_i.tsne_early_exaggeration

            graph_title = f"Perp ratio: {round(perplexity_ratio_i, 4)} / " \
                          f"Perp: {perplexity_i} / EE: {early_exaggeration_i} / " \
                          f"LearnRate: {learning_rate_i} / #data={p_i.num_training_data_points} / tsneNiter={p_i.tsne_n_iter}"
            # Save graph to file
            p_i.plot_clusters_by_assignments(
                title=graph_title,
                fig_file_prefix=graph_filename_prefix,
                show_now=False, save_to_file=True, figsize=graph_dimensions,
                s=0.4 if show_cluster_graphs_in_a_popup_window else 1.5,
            )
            logger.debug(f'Finished successfully building iteration: {(i+1)/len(kwargs_product)}')
        end_build = time.perf_counter()
        logger.info(f'Time to build: {round(end_build-start_build)} seconds (using {max_cores_per_pipe} cores)')
        logger.debug('---------------------------------------------\n\n')
    end_time = time.perf_counter()
    logger.info(f'Total compute time: {round((end_time - start_time) / 60, 2)} minutes. Total successful jobs with results: {successful_builds}. Total jobs computed: {len(pipeline_names_by_index)}')
    logger.debug(f'Done job at: {time.strftime("%Y-%m-%d_%HH%MM")}')


def buildone(**kwargs):
    """
    Build just one pipeline, save to file.
    :param pipeline_name:
    :param kwargs:
    :return:
    """
    # Param section -- MAGIC VARIABLES GO HERE

    LocallyLinearEmbedding_n_neighbors = 50
    cvae_num_steps = 10_000
    isomap_n_neighbors = 50
    LLE_method = 'modified'  # OR: standard OR hessian OR ltsa
    percent_epm_train_files_to_cluster_on = 1.0

    save_to_folder = True
    save_graph_to_file = True
    skip_accuracy_scoring = True
    max_cores_per_pipe = 5
    pipeline_implementation = pipeline.PipelineHowland  # Another option includes dibs.pipeline.PipelineMimic
    graph_dimensions = (12, 12)  # length x width?
    show_cluster_graphs_in_a_popup_window = False  # Set to False to display graphs inline

    pipeline_name = kwargs['name']

    # vid_data_source = 'EPM-MCE-10DLC_resnet50_Maternal_EPMDec28shuffle1_700000'
    # vid_file_path = "C:\\Users\\killian\\projects\\DIBS\\epm-mce-vids-f\\EPM-MCE-10.mp4"
    # assert os.path.isfile(vid_file_path), f'Video file not found. Path = {vid_file_path}'

    # Arg checking
    assert 0 < percent_epm_train_files_to_cluster_on <= 1.0
    if pipeline_name is None:
        err_type = f'Name was expected to be a string but instead found None (likely due to missing command-line parameter). Try using the -n option.'
        logger.error(err_type)
        raise TypeError(err_type)
    # Auto-generate the product between all possible parameters
    pipeline_kwargs = {
        'LLE_method': LLE_method,
        'cvae_num_steps': cvae_num_steps,
        'tsne_perplexity': kwargs.get('perplexity', config.TSNE_PERPLEXITY),
        'tsne_early_exaggeration': kwargs.get('early_exaggeration', config.TSNE_EARLY_EXAGGERATION),
        'tsne_learning_rate': kwargs.get('learning_rate', config.TSNE_LEARNING_RATE),
        'gmm_n_components': kwargs.get('gmm_n_components', config.gmm_n_components),
        'tsne_n_components': 2,  # n-D dimensionality reduction
        # 'tsne_n_iter': kwargs.get('tsne_n_iter', config.TSNE_N_ITER),
        'tsne_n_iter': kwargs.get('tsne_n_iter', config.TSNE_N_ITER),
        'cross_validation_k': max_cores_per_pipe,
        'cross_validation_n_jobs': max_cores_per_pipe,
        'rf_n_jobs': max_cores_per_pipe,
        'tsne_n_jobs': max_cores_per_pipe,

        'LocallyLinearEmbedding_n_neighbors': LocallyLinearEmbedding_n_neighbors,
        'isomap_n_neighbors': isomap_n_neighbors,
    }

    # Queue up which data files will be added to each Pipeline
    all_files_paths = [os.path.join(config.DEFAULT_TRAIN_DATA_DIR, file) for file in os.listdir(config.DEFAULT_TRAIN_DATA_DIR)]
    actual_used_training_data_files_paths = all_files_paths[:int(len(all_files_paths) * percent_epm_train_files_to_cluster_on)]
    # print(train_data)  # Uncomment this line to see which exact data files are added to the Pipeline

    # Instantiated pipeline
    logger.debug(f'{logging_enhanced.get_current_function()}(): Start time: {time.strftime("%Y-%m-%d_%HH%MM")}')
    start_build = time.perf_counter()
    results_current_time = time.strftime("%Y-%m-%d_%HH%MM")
    p: pipeline.BasePipeline = pipeline_implementation(pipeline_name, **pipeline_kwargs).set_description(f'{pipeline_name}. Built on: {results_current_time}').add_train_data_source(*actual_used_training_data_files_paths)

    # Try to build
    try:
        p = p.build(skip_accuracy_score=skip_accuracy_scoring)
    except Exception as e:
        info = f'PerpRaw={p._tsne_perplexity}/Perp={p.tsne_perplexity}/' \
               f'EE={p.tsne_early_exaggeration}/LR={p.tsne_learning_rate}/GMM-N={p.gmm_n_components}'
        err = f'Unexpected exception::{__name__}.{logging_enhanced.get_current_function()}(): ' \
              f'an unexpected exception occurred when building many pipelines to get good graphs. ' \
              f'Info is as follows: {info}. Exception is: {repr(e)}.'  # Diagnostics: {p_i.diagnostics()}'
        logger.error(err)
        raise RuntimeError(err)

    ### If no exceptions raised, continue
    # Save pipeline to folder
    if save_to_folder:
        p.save_to_folder(config.OUTPUT_PATH)

    # Generate result info
    perplexity_ratio_i, perplexity_i, learning_rate_i, early_exaggeration_i = p.tsne_perplexity_relative_to_num_data_points, p.tsne_perplexity, p.tsne_learning_rate, p.tsne_early_exaggeration
    graph_title = f"Perp ratio: {round(perplexity_ratio_i, 5)} / " \
                  f"Perp: {perplexity_i} / EE: {early_exaggeration_i} / " \
                  f"LearnRate: {learning_rate_i} / #data={p.num_training_data_points} / tsneNiter={p.tsne_n_iter}"

    # Save graph to file
    p.plot_clusters_by_assignments(
        title=graph_title,
        fig_file_prefix=f'{p.name}__cvaenumsteps_{cvae_num_steps}__{results_current_time}__',
        save_to_file=save_graph_to_file,
        figsize=graph_dimensions,
        s=0.4 if show_cluster_graphs_in_a_popup_window else 1.5,
        show_now=False,
    )
    # # Save pipeline to folder
    # p_i.save_to_folder()

    # # Try to make example vids
    # p_i.make_behaviour_example_videos(vid_data_source, vid_file_path, file_name_prefix='somethingsomething', num_frames_buffer=1)  #     def make_behaviour_example_videos(self, data_source: str, video_file_path: str, file_name_prefix=None, min_rows_of_behaviour=1, max_examples=3, num_frames_buffer=0, output_fps=15):

    end_build = time.perf_counter()
    build_time_secs = round((end_build-start_build), 2)
    logger.info(f'Time to build "{pipeline_name}": {build_time_secs} seconds (using {max_cores_per_pipe} cores)')
    logger.debug(f'Done job at: {time.strftime("%Y-%m-%d_%HH%MM")}')


### Utilities

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


def print_if_system_is_64_bit(**kwargs):
    print(f'This system is detected to be 64-bit: {sys.maxsize > 2**32}')


def sample(*args, **kwargs):
    print(f'Args: {args}')
    print(f'kwargs: {kwargs}')

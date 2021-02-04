"""
Functionality for visualizing plots and saving those plots.

Color map information: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

"""
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits.mplot3d import Axes3D  # Despite being "unused", this import MUST stay for 3d plotting to work. PLO!
from typing import Collection, Tuple
import inspect
import os
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib as mpl
# import seaborn as sn

from dibs import check_arg, config
from dibs.logging_enhanced import get_current_function


#####################################################################################################################

logger = config.initialize_logger(__name__)
matplotlib_axes_logger.setLevel('ERROR')


### Generate graphs ###

def plot_assignment_distribution_histogram(assignments: Collection, **kwargs) -> Tuple[object, object]:
    """
    Produce histogram plot of assignments. Useful for seeing lop-sided outcomes.
    :param assignments:
    :param kwargs:
    :return:
    """
    # Arg checking
    if not isinstance(assignments, np.ndarray):
        assignments = np.array(assignments)
    # Kwarg resolution
    histtype = kwargs.get('histtype', 'stepfilled')
    # Do
    unique_assignments = np.unique(assignments)
    R = np.linspace(0, 1, len(unique_assignments))
    colormap = plt.cm.get_cmap("Spectral")(R)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, assignment in enumerate(unique_assignments):
        idx = np.where(assignments == assignment)
        plt.hist(assignments[idx], histtype=histtype, color=colormap[i])

    return fig, ax


def plot_cross_validation_scores(scores, save_to_file=False, fig_file_prefix='classifier_accuracy_score', **kwargs) -> Tuple[object, object]:
    """
        TODO: elaborate
    :param scores:
    :param save_to_file:
    :param fig_file_prefix:
    :param kwargs:
    :return:
    """
    # Parse kwargs
    facecolor, edgecolor = kwargs.get('facecolor', 'w'), kwargs.get('edgecolor', 'k')
    s, c, alpha = kwargs.get('s', 40), kwargs.get('c', 'r'), kwargs.get('alpha', 0.5)
    xlabel, ylabel = kwargs.get('xlabel', 'SVM classifier'), kwargs.get('ylabel', 'Accuracy')
    #
    # TODO: decouple the fig saving and the plotting. Current state is due to legacy.
    fig = plt.figure(facecolor=facecolor, edgecolor=edgecolor)
    fig.suptitle(f"Performance on {config.HOLDOUT_PERCENT * 100} % data")
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    if len(x) != len(scores):
        logger.error(f'len(x) does not equal len(scores). '
                     f'If you see an error next, check the logs! x = {x} / scores = {scores}.')
    if isinstance(x, np.ndarray) and isinstance(scores, np.ndarray):
        logger.debug(f'{get_current_function()}: both inputs are arrays. '
                     f'x.shape = {x.shape} // scores.shape = {scores.shape}')
        if x.shape != scores.shape:
            logger.error(f'{inspect.stack()[0][3]}(): x = {x} // scores = {scores}')
    plt.scatter(x, scores, s=s, c=c, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    if save_to_file:
        fig_file_name = f'{fig_file_prefix}_{config.runtime_timestr}'
        save_graph_to_file(fig, fig_file_name)  # fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}_{timestr}.svg'))

    return fig, ax


def plot_clusters_by_assignment(data: np.ndarray, assignments, save_fig_to_file: bool, fig_file_prefix='train_assignments_and_clustering__', show_now=True, draw_now=False, figsize=None, **kwargs) -> Tuple[object, object]:  # TODO: medium: rename this function
    """
    Plot trained TSNE for EM-GMM assignments
    :param show_now:
    :param data: 2D array, trained_tsne array (i rows, j columns) where j must equal either 2 or 3
    :param assignments: 1D array, EM-GMM assignments
    :param save_fig_to_file:
    :param fig_file_prefix:
    :param show_now: (bool) use draw() instead of show()
    :param kwargs:
        s : float

        marker : str

        alpha : float

        title : str

        azim_elev : 2-Tuple[int, int]



    """
    # TODO: find out why attaching the log entry/exit decorator kills the streamlit rotation app. For now, do not attach
    # Arg checks
    check_arg.ensure_type(data, np.ndarray)
    num_columns = data.shape[1]
    if num_columns not in {2, 3}:
        err = f'{get_current_function()}(): submitted data has an unexpected number of ' \
              f'columns. Expected 2 or 3 columns but instead found {data.shape[1]} (data shape: {data.shape}).'
        logger.error(err)
        raise ValueError(err)
    if figsize is not None:
        check_arg.ensure_type(figsize, tuple)
        # TODO: low: add more figsize checks

    # Parse kwargs
    s = kwargs.get('s', 0.5)
    marker = kwargs.get('marker', 'o')
    alpha = kwargs.get('alpha', 0.8)
    title = kwargs.get('title', 'Assignments by GMM')  # TODO: med: review hiding this param. Maybe push to arg line?
    azim_elev = kwargs.get('azim_elev', (70, 135))
    # Plot graph

    unique_assignments = list(np.unique(assignments))
    R = np.linspace(0, 1, len(unique_assignments))
    colormap = plt.cm.get_cmap("Spectral")(R)
    if num_columns == 2:
        tsne_x, tsne_y = data[:, 0], data[:, 1]
    elif num_columns == 3:
        tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d' if num_columns == 3 else None)

    # Loop over assignments
    for i, assignment in enumerate(unique_assignments):
        # Select data for only assignment i
        idx = np.where(assignments == assignment)
        # Assign to colour and plot
        if num_columns == 2:
            ax.scatter(tsne_x[idx], tsne_y[idx], c=colormap[i], label=assignment, s=s, marker=marker, alpha=alpha)
        elif num_columns == 3:
            ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=colormap[i], label=assignment, s=s, marker=marker, alpha=alpha)
    if num_columns == 2:
        ax.set_xlabel('Dim. 1')
        ax.set_ylabel('Dim. 2')
        plt.legend(ncol=2)
    elif num_columns == 3:
        ax.set_xlabel('Dim. 1')
        ax.set_ylabel('Dim. 2')
        ax.set_zlabel('Dim. 3')
        ax.view_init(*azim_elev)
        plt.legend(ncol=3)

    plt.title(title)

    # Draw now?
    if show_now:
        plt.show()
    elif draw_now:
        plt.draw()
    # Save to graph to file?
    if save_fig_to_file:
        file_name = f'{fig_file_prefix}_{config.runtime_timestr}'  # fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}_{time_str}.svg'))
        save_graph_to_file(fig, file_name)

    return fig, ax


def plot_GM_assignments_in_3d(data: np.ndarray, assignments, save_fig_to_file: bool, fig_file_prefix='train_assignments', show_now=True, **kwargs) -> Tuple[object, object]:  # TODO: medium: rename this function
    """
    Plot trained TSNE for EM-GMM assignments

    :param data: 2D array, trained_tsne array (3 columns)
    :param assignments: 1D array, EM-GMM assignments
    :param save_fig_to_file: (bool)
    :param fig_file_prefix: (str)
    :param show_now: (bool)
    :param show_later: use draw() instead of show()
    """
    # TODO: find out why attaching the log entry/exit decorator kills the streamlit rotation app. For now, do not attach
    check_arg.ensure_type(data, np.ndarray)

    # Parse kwargs
    s = kwargs.get('s', 0.5)
    marker = kwargs.get('marker', 'o')
    alpha = kwargs.get('alpha', 0.8)
    title = kwargs.get('title', 'Assignments by GMM')
    azim_elev = kwargs.get('azim_elev', (70, 135))
    # Plot graph
    unique_assignments = list(np.unique(assignments))
    R = np.linspace(0, 1, len(unique_assignments))
    colormap = plt.cm.get_cmap("Spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Loop over assignments
    for i, assignment in enumerate(unique_assignments):
        # Select data for only assignment i
        idx = np.where(assignments == assignment)
        # Assign to colour and plot
        ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=colormap[i], label=assignment, s=s, marker=marker, alpha=alpha)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(*azim_elev)
    plt.title(title)
    plt.legend(ncol=3)
    # Draw now?
    if show_now:
        plt.show()
    else:
        plt.draw()
    # Save to graph to file?
    if save_fig_to_file:
        file_name = f'{fig_file_prefix}_{config.runtime_timestr}'  # fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}_{time_str}.svg'))
        save_graph_to_file(fig, file_name)

    return fig, ax


### Helper functions ###

def generate_color_map(n_colors: int, map_type='Spectral') -> np.ndarray:
    """
    Generate a deterministic color map of evenly spaced colours according to mapping type.
    Documentation can be found here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    :param map_type:
    :param n_colors: (int) Number of colors to generate
    :return: a 2-d array of shape (n_colors, 4).
        E.g.: If input n = 4, then output will look like below:
        array([[0.61960784, 0.00392157, 0.25882353, 1.        ],
               [0.99346405, 0.74771242, 0.43529412, 1.        ],
               [0.74771242, 0.89803922, 0.62745098, 1.        ],
               [0.36862745, 0.30980392, 0.63529412, 1.        ]])

    """
    space = np.linspace(0, 1, n_colors)
    colormap = plt.cm.get_cmap(map_type)(space)
    return colormap


def save_graph_to_file(figure: object, file_title: str, file_type_extension=config.DEFAULT_SAVED_GRAPH_FILE_FORMAT, alternate_save_path: str = None) -> None:
    """
    :param figure: (object) a figure object. Must have a savefig() function
    :param file_title: (str)
    :param alternate_save_path:
    :param file_type_extension: (str)
    :return:
    """
    if alternate_save_path and not os.path.isdir(alternate_save_path):
        path_not_exists_err = f'Alternate save file path does not exist. Cannot save image to path: {alternate_save_path}.'
        logger.error(path_not_exists_err)
        raise ValueError(path_not_exists_err)
    if not hasattr(figure, 'savefig'):
        cannot_save_input_figure_error = f'Figure is not savable with current interface. ' \
                                         f'Requires ability to use .savefig() method. ' \
                                         f'repr(figure) = {repr(figure)}.'
        logger.error(cannot_save_input_figure_error)
        raise AttributeError(cannot_save_input_figure_error)
    out_path = os.path.join(config.GRAPH_OUTPUT_PATH, f'{file_title}.{file_type_extension}')
    # After type checking, save fig to file
    logger.debug(f'Saving graph to following path: {out_path}')
    figure.savefig(out_path)
    return


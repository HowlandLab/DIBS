"""
Functionality for visualizing plots and saving those plots.
"""
from mpl_toolkits.mplot3d import Axes3D  # Despite being "unused", this import MUST stay for 3d plotting to work. PLO!
from typing import Collection, Tuple
import inspect
import os
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


TM = NotImplementedError('TODO: HIGH: The source of TM has not been determined. Find and fix as such.')  # TODO: low


### New

def plot_GM_assignments_in_3d_new(data: np.ndarray, assignments, show_now=True, **kwargs) -> Tuple[object, object]:
    """
    Plot trained TSNE for EM-GMM assignments.

    :param data: (2-d numpy array) Must be a numpy array with a shape of: (n_rows, 3).
        So currently, for this function at least, the input data MUST be t-sne reduced into 3 dimensions
    :param assignments: 1D array, EM-GMM assignments, used to colour the points. Must be the same length as @data

    The assignment (for @param assignments) at row index i directly correlates to the tsne dimensions (for @param data) at index row i

    :param show_now: (bool) Default True. ___

    """
    # TODO: find out why attaching the log entry/exit decorator kills the streamlit graph-rotation app
    if not isinstance(data, np.ndarray):
        err = f'Expected `data` to be of type numpy.ndarray but instead found: {type(data)} (value = {data}).'
        raise TypeError(err)
    # Parse kwargs
    s = kwargs.get('s', 1.5)
    marker = kwargs.get('marker', 'o')
    alpha = kwargs.get('alpha', 0.8)
    title = kwargs.get('title', 'Drug Labels Embedded in 3 Dimensions based on similarity by tSNE')
    azim_elev = kwargs.get('azim_elev', (70, 135))
    # Plot graph
    unique_assignments = list(np.unique(assignments))
    R = np.linspace(0, 1, len(unique_assignments))
    colormap = plt.cm.get_cmap("nipy_spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Loop over assignments
    for i, g in enumerate(unique_assignments):
        # Select data for only assignment i
        idx = np.where(assignments == g)
        # Assign to colour and plot
        ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=colormap[i],
                   #label=(g if g == 'default' else None),
                   s=s, marker=marker, alpha=alpha)
    ax.set_xlabel('tSNE Dim 1')
    ax.set_ylabel('tSNE Dim 2')
    ax.set_zlabel('tSNE Dim 3')
    ax.view_init(*azim_elev)
    plt.title(title)
    plt.legend(ncol=3)
    # Show now?
    if show_now:
        plt.show()
    else:
        plt.draw()

    return fig, ax


#######################################################################################################################

def plot_tsne_in_3d(data, **kwargs):
    # TODO: HIGH: consider reducing the total data when plotting because, if TONS of data is
    #  plotted in 3d, it can be very laggy when viewing and when especially rotating
    """
    Plot trained tsne
    :param data: trained_tsne TODO: expand desc. and include type
    """
    if not isinstance(data, np.ndarray):
        err = f'data was expected to be of type array but instead found type: {type(data)}.'
        raise TypeError(err)
    # TODO: low: reduce
    # Parse kwargs
    x_label = kwargs.get('x_label', 'Dim. 1')
    y_label = kwargs.get('y_label', 'Dim. 2')
    z_label = kwargs.get('z_label', 'Dim. 3')
    # Produce graph
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_x, tsne_y, tsne_z, s=1, marker='o', alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.view_init(70, 135)
    plt.title('Embedding of the training set by t-SNE')
    plt.show()


def plot_GM_assignments_in_3d_tuple(data: np.ndarray, assignments, save_fig_to_file: bool, fig_file_prefix='train_assignments', show_now=True, **kwargs) -> Tuple[object, object]:  # TODO: medium: rename this function
    """
    Plot trained TSNE for EM-GMM assignments
    :param data: 2D array, trained_tsne array (3 columns)
    :param assignments: 1D array, EM-GMM assignments
    :param save_fig_to_file:
    :param fig_file_prefix:
    :param show_later: use draw() instead of show()
    """
    # TODO: find out why attaching the log entry/exit decorator kills the streamlit rotation app. For now, do not attach.
    if not isinstance(data, np.ndarray):
        err = f'Expected `data` to be of type numpy.ndarray but instead found: {type(data)} (value = {data}).'
        raise TypeError(err)
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
        file_name = "GMM_with_3d_tuple"
        save_graph_to_file(fig, file_name)

    return fig, ax



def save_graph_to_file(figure: object, file_title: str, file_type_extension=".pdf",
                       alternate_save_path: str = None) -> None:
    """
    :param figure: (object) a figure object. Must have a savefig() function
    :param file_title: (str)
    :param alternate_save_path:
    :param file_type_extension: (str)
    :return:
    """
    if alternate_save_path and not os.path.isdir(alternate_save_path):
        path_not_exists_err = f'Alternate save file path does not exist. Cannot save image to path: {alternate_save_path}.'
        raise ValueError(path_not_exists_err)
    if not hasattr(figure, 'savefig'):
        cannot_save_input_figure_error = f'Figure is not savable with current interface. ' \
                                         f'Requires ability to use .savefig() method. ' \
                                         f'repr(figure) = {repr(figure)}.'
        raise AttributeError(cannot_save_input_figure_error)
    # After type checking, save fig to file
    graph_path="./"
    figure.savefig(os.path.join(graph_path, f'{file_title}.{file_type_extension}'))
    return



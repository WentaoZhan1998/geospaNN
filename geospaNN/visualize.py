from .model import nngls
import warnings

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Optional, Tuple

import torch
import numpy as np
import pandas as pd

from scipy.interpolate import griddata, CloughTocher2DInterpolator

def spatial_plot_surface(variable: np.array,
                         coord: np.array,
                         title: Optional[str] = "Variable",
                         save_path: Optional[str] = "./",
                         file_name: Optional[str] = "spatial_surface.png",
                         grid_resolution: Optional[int] = 1000,
                         method: Optional[str] = "CloughTocher",
                         cmap: Optional[str] = 'viridis',
                         show: Optional[bool] = False
                         ) -> None:
    """
    Plots a smooth surface for spatial data.

    The function interpolates scattered data onto a regular grid and generates
    a smooth surface plot. The resulting plot is saved as a PNG file.

    Parameters:
        variable (array-like): Values to plot, corresponding to the coordinates.
        coord (array-like): Coordinates of shape (n, 2) where each row represents a point (x, y).
        title (str, optional): Title of the plot. Defaults to "Variable".
        save_path (str, optional): Directory to save the plot. Defaults to "./".
        file_name (str, optional): Name of the saved plot file. Defaults to "spatial_surface.png".
        grid_resolution (int, optional): Resolution of the interpolation grid.
            Higher values produce finer surfaces. Defaults to 100.
        cmap (str, optional): Colormap to use for the plot. Defaults to 'viridis'.

    Raises:
        ValueError: If the lengths of `variable` and `coord` do not match.

    Returns:
        None: The function saves the plot to a file and does not return any value.

    Example:
        >>> import numpy as np
        >>> coord = np.random.uniform(0, 10, (50, 2))
        >>> variable = np.sin(coord[:, 0]) + np.cos(coord[:, 1])
        >>> spatial_plot_surface(variable, coord, title="Example Plot", grid_resolution=200)
    """
    if len(variable) != len(coord):
        raise ValueError("Length of 'variable' and 'coord' must match.")

    # Create grid for interpolation
    xi = np.linspace(coord[:, 0].min(), coord[:, 0].max(), grid_resolution)
    yi = np.linspace(coord[:, 1].min(), coord[:, 1].max(), grid_resolution)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate data onto the grid

    if method == "CloughTocher":
        interp = CloughTocher2DInterpolator(list(zip(coord[:, 0], coord[:, 1])), variable)
        Z = interp(X, Y)
    elif method in ['linear', 'nearest', 'cubic']:
        Z = griddata(coord, variable, (X, Y), method=method)
    else:
        warnings.warn("No interpolation method provided, use cubic spline as default!")
        Z = griddata(coord, variable, (X, Y), method="cubic")

    # Plot the interpolated surface
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    surface = plt.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
    plt.colorbar(surface, label='Variable')
    plt.title(title)
    plt.xlabel('Coord_X')
    plt.ylabel('Coord_Y')
    plt.savefig(f"{save_path}/{file_name}")
    if show:
        plt.show()

    return fig


class _PDP_estimator(BaseEstimator, RegressorMixin):
    def __init__(self, intValue=0):
        self.intValue = intValue

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def fit(self, X, model):
        self.treshold_ = 1
        self.model = model
        return self

    def _meaning(self, x):
        return 1

    def predict(self, X):
        if isinstance(self.model, nngls):
            return self.model.estimate(torch.from_numpy(X)).reshape(-1).detach().numpy()
        else:
            return self.model(torch.from_numpy(X)).reshape(-1).detach().numpy()


def plot_PDP(model,
             X: torch.tensor,
             names: Optional[list] = [],
             save_path: Optional[str] = "./",
             save: bool = False,
             split: bool = False):
    """Partial dependency plot for model on the data.

    A Partial Dependence Plot (PDP) is a visualization tool used to illustrate the relationship between a selected feature
    and the predicted outcome of a machine learning model, while averaging out the effects of other features.
    This helps to understand the marginal influence of a single feature on the model's predictions in a more interpretable way.

    Parameters:
        model:
            Usually a model in nngls class. Can take model with .() method that take tensor X as input and
            predicted scalar value Y as output. (to implement for more models)
        X:
            nxp array of the covariates.
        names:
            List of names for variable, if not specified, use "variable 1" to "variable p".
        save_path:
            Directory to save the plot. Defaults to "./".
        save:
            Whether to save the PDPs to the working directory. Default False.
        split:
            Whether to return the PDPs as a list of PartialDependenceDisplay object for single variabls or a whole

    Returns:
        A sklearn.inspection.PartialDependenceDisplay object contains PDPs for each variable.
    """
    X = X.detach().numpy()
    p = X.shape[1]
    Est = _PDP_estimator()
    Est.fit(X, model)
    if len(names) != p:
        warnings.warn("length of names does not match columns of X, replace by variable index")
        names = [f"variable {i + 1}" for i in range(p + 1)]

    if not split:
        figures = PartialDependenceDisplay.from_estimator(estimator=Est, X=X, features=[i for i in range(p)],
                                                          feature_names=names,
                                                          percentiles=(0.05, 0.95))
        if save:
            plt.savefig(save_path + names[0] + ".png")

    else:
        figures = []
        for k in range(p):
            res = PartialDependenceDisplay.from_estimator(estimator=Est, X=X, features=[k],
                                                          feature_names=names[k],
                                                          percentiles=(0.05, 0.95))
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                                wspace=0.4, hspace=0.4)
            if save:
                plt.savefig(save_path + names[k] + ".png")
            figures.append(res)

    return figures

def plot_PDP_list(model_list: list,
                  model_names: list,
                  X: torch.tensor,
                  feature_names: Optional[list] = [],
                  save_path: Optional[str] = "./",
                  save: bool = False,
                  split: bool = True):
    """Partial dependency plot for model on the data.

    A Partial Dependence Plot (PDP) is a visualization tool used to illustrate the relationship between a selected feature
    and the predicted outcome of a machine learning model, while averaging out the effects of other features.
    This helps to understand the marginal influence of a single feature on the model's predictions in a more interpretable way.

    Parameters:
        model_list:
            A list of models described in plot_PDP.
        model_names:
            A list of model names for PDP visualization, expect the same length as model_list.
        X:
            nxp array of the covariates for PDP integration.
        feature_names:
            List of names for variable, if not specified, use "variable 1" to "variable p".
        save_path:
            Directory to save the plot. Defaults to "./".
        save:
            Whether to save the PDPs to the working directory. Default False.
        split:
            Whether to save the PDP's for different features seperately or as one figure. The default is True.

    Returns:
        A sklearn.inspection.PartialDependenceDisplay object contains PDPs for each variable.
    """
    l = min(len(model_list), len(model_names))
    if len(model_list) != len(model_names):
        warnings.warn("Number of models different from number of model_names! Use top " + l + " ones.")
    PDP_list = [plot_PDP(model_list[i], X, feature_names) for i in range(l)]
    p = X.shape[1]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(p, 1, figsize=(6, 4 * p))  # Adjust height proportionally

    for i in range(l):
        PDP_list[i].plot(ax=axes, line_kw={"label": model_names[i], "color": colors[i]})

    if split:
        for i, ax in enumerate(axes):
            ax.set_title(f"Partial Dependence for Feature {i + 1}")
            ax.legend()  # Ensure legends show up for each axis

            # Save each axis as a separate figure
            fig_single, ax_single = plt.subplots(figsize=(6, 4))  # Create a new figure
            for line in ax.get_lines():  # Copy lines from the original axis
                ax_single.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())

            ax_single.set_title(ax.get_title())
            ax_single.set_xlabel(ax.get_xlabel())
            ax_single.set_ylabel(ax.get_ylabel())
            ax_single.legend()

            # Save the figure for the current axis
            if save:
                fig_single.savefig(save_path + f"PDP_feature_{i + 1}.png")
            plt.close(fig_single)
    else:
        axes.set_title(f"Partial Dependence for Features")
        axes.legend()  # Ensure legends show up for each axis
        fig.savefig(save_path + f"PDP_features.png")

def plot_log(training_log: list,
             theta: tuple[float, float, float],
             save_path: Optional[str] = "./",
             save: bool = False):
    """Output visualization for NN-GLS training.

    This is a simple visualization of the training log from geospaNN.nngls_train.train()

    Parameters:
        training_log:
                A list contains vectors of validation loss, spatial parameters sigma, phi, and tau.
                Lengths of the vectors must be the same.
        theta:
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.
        save_path:
            Directory to save the plot. Defaults to "./".
        save:
            Whether to save the PDPs to the working directory. Default False.
    """
    epoch = len(training_log["val_loss"])
    training_log["epoch"] = list(range(1, epoch + 1))
    training_log = pd.DataFrame(training_log)

    # Melting the dataframe to make it suitable for seaborn plotting
    training_log_melted = training_log[["epoch", "val_loss"]].melt(id_vars='epoch', var_name='Variable', value_name='Value')
    # Plotting with seaborn
    # Creating two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    sns.lineplot(ax=axes[0], data=training_log_melted, x='epoch', y='Value', hue='Variable', style='Variable', markers=False, dashes=False)

    axes[0].set_title('Validation and prediction loss over Epochs (Log Scale) with Benchmark', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=15)
    axes[0].set_ylabel('Value (Log Scale)', fontsize=15)
    axes[0].set_yscale('log')
    axes[0].legend(prop={'size': 15})
    axes[0].tick_params(labelsize=14)
    axes[0].grid(True)

    # Second plot (sigma, phi, tau)
    kernel_params_melted = training_log[["epoch", "sigma", "phi", "tau"]].melt(id_vars='epoch', var_name='Variable', value_name='Value')
    ground_truth = {'sigma': theta[0], 'phi': theta[1], 'tau': theta[2]}
    sns.lineplot(ax=axes[1], data=kernel_params_melted, x='epoch', y='Value', hue='Variable', style='Variable', markers=False, dashes=False)
    palette = sns.color_palette()
    for i, (param, gt_value) in enumerate(ground_truth.items()):
        axes[1].hlines(y=gt_value, xmin=1, xmax=epoch, color=palette[i], linestyle='--')
    axes[1].set_title('Parameter Values (log) over Epochs with Ground Truth', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=15)
    axes[1].set_ylabel('Value', fontsize=15)
    axes[1].set_yscale('log')
    axes[1].legend(prop={'size': 15})
    axes[1].tick_params(labelsize=14)
    axes[1].grid(True)

    plt.tight_layout()
    if save:
        plt.savefig(save_path + "training_log.png")
'''
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, axes = plt.subplots(p, 1, figsize=(6, 4 * p))  # Adjust height proportionally

# Plot each feature on a separate axis
PDP_truth.plot(ax=axes, line_kw={"label": "Friedman's function", "color": colors[0]})
PDP_NNGLS.plot(ax=axes, line_kw={"label": "NNGLS", "color": colors[1]})
PDP_NN.plot(ax=axes, line_kw={"label": "NN", "color": colors[2]})

for i, ax in enumerate(axes):
    ax.set_title(f"Partial Dependence for Feature {i + 1}")
    ax.legend()  # Ensure legends show up for each axis

    # Save each axis as a separate figure
    fig_single, ax_single = plt.subplots(figsize=(6, 4))  # Create a new figure
    for line in ax.get_lines():  # Copy lines from the original axis
        ax_single.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
    
    ax_single.set_title(ax.get_title())
    ax_single.set_xlabel(ax.get_xlabel())
    ax_single.set_ylabel(ax.get_ylabel())
    ax_single.legend()

    # Save the figure for the current axis
    fig_single.savefig(path + f"PDP_feature_{i + 1}.png")
    plt.close(fig_single) 
'''


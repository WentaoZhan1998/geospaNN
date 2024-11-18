from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Callable, Optional, Tuple

def spatial_plot_surface(variable: np.array,
                         coord: np.array,
                         title: Optional[str] = "Variable",
                         save_path: Optional[str] = "./",
                         file_name: Optional[str] = "spatial_surface.png",
                         grid_resolution: Optional[int] = 1000,
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
    Z = griddata(coord, variable, (X, Y), method='cubic')

    # Plot the interpolated surface
    plt.clf()
    plt.figure(figsize=(8, 6))
    surface = plt.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
    plt.colorbar(surface, label='Variable')
    plt.title(title)
    plt.xlabel('Coord_X')
    plt.ylabel('Coord_Y')
    plt.savefig(f"{save_path}/{file_name}")
    if show:
        plt.show()
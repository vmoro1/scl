# Visualize PDE solutions and predictions.
# Different functions are used depending on the dimensionality 
# of the PDE (both spatial and temporal).

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_1D_spatial_1D_temporal(u, x, t, path):
    """Plot the solution or prediction for a PDE with one spatial dimension and one temporal dimension."""
    u = u.detach().cpu().numpy().T      # shape: (nt, nx) --> (nx, nt)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(u, 
                   origin='lower', 
                   cmap='viridis', 
                   aspect='auto', 
                   interpolation='nearest', 
                   extent=[t.min(), t.max(), x.min(), x.max()])

    # add color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    # ax.set_title("ABC'", fontsize=20)
    ax.set_xlabel('t', fontweight='bold', size=15)
    ax.set_ylabel('x', fontweight='bold', size=15)
    ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def visualize_2D_spatial(u, x, y, path):
    """Plot the solution or prediction for a PDE with two spatial dimensions."""
    u = u.detach().cpu().numpy().T      # shape: (nx, ny) --> (ny, nx)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(u, 
                   origin='lower', 
                   cmap='viridis', 
                   aspect='auto', 
                   interpolation='nearest', 
                   extent=[x.min(), x.max(), y.min(), y.max()])

    # add color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    # ax.set_title("ABC'", fontsize=20)
    ax.set_xlabel('x', fontweight='bold', size=15)
    ax.set_ylabel('y', fontweight='bold', size=15)
    ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def visualize_2D_spatial_1D_temporal():
    """Plot the solution or prediction for a PDE with two spatial dimensions and one temporal dimension."""
    pass
    # TODO: Plot e.g., snapshots from the solution or prediction
    # Check the above and test in test_plot.ipynb

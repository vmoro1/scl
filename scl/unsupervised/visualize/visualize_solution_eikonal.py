# Visualize teh solution of the Eikonal equation.

# NOTE: Two diffuerent ways of plotting is used (they are all the same except for axis location and colormaps). 
# The first three functions vs. the last. The only difference is where the axis are placed. The last function 
# don't transpose the input and use origin=upper in contrast to the first htree funcitons. This is the only 
# difference between the two ways of plotting. Both are correct and display the solutions correctly.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_exact_u(u_exact, x, y, path, suffix=""):
    """Visualize exact solution."""
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    h = ax.imshow(u_exact.T, interpolation='nearest', cmap='coolwarm',
                  extent=[y.min(), y.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel('y', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)

    plt.savefig(f"{path}/exact_pde_solution{suffix}.pdf")
    plt.close()

    return None


def plot_u_diff(u_exact, u_pred, x, y, path, relative_error=False, suffix=""):
    """Visualize abs(u_pred - u_exact)."""

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    if relative_error:
        h = ax.imshow(np.abs(u_exact.T - u_pred.T)/np.abs(u_exact.T), interpolation='nearest', cmap='binary',
                    extent=[y.min(), y.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    else:
        h = ax.imshow(np.abs(u_exact.T - u_pred.T), interpolation='nearest', cmap='binary',
                    extent=[y.min(), y.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    line = np.linspace(x.min(), x.max(), 2)[:,None]

    ax.set_xlabel('y', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)

    plt.savefig(f"{path}/u_diff{suffix}.pdf")
    plt.close()

    return None


def plot_u_pred(u_exact, u_pred, x, y, path, suffix=""):
    """Visualize u_predicted."""

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    u_exact_min = u_exact.min()
    u_exact_max = u_exact.max()

    # colorbar for prediction: set min/max to ground truth solution.
    h = ax.imshow(u_pred.T, interpolation='nearest', cmap='coolwarm',
                  extent=[y.min(), y.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=u_exact_min, vmax=u_exact_max)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    line = np.linspace(x.min(), x.max(), 2)[:,None]

    ax.set_xlabel('y', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)

    plt.savefig(f"{path}/predicted_solution{suffix}.pdf")
    plt.close()

    return None


def visualize(usol, u_pred, f_pred, path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), dpi=100)
    
    ax = axes[0]
    h = ax.imshow(usol, cmap='jet', aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(r'Exact $u(x)$')

    ax = axes[1]
    h = ax.imshow(u_pred, cmap='jet', aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(r'Predicted $u(x)$')

    ax = axes[2]
    h = ax.imshow(np.abs(usol - u_pred), cmap='jet', aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Absolute error')
    
    x_lim = [-1, 1]
    y_lim = [-1, 1]
    ax = axes[3]
    h = ax.imshow(f_pred, interpolation='nearest', cmap='copper',
                extent=[y_lim[0], y_lim[1], x_lim[0], x_lim[1]],
                aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    fig.tight_layout()
    plt.savefig(f"{path}/visualization.pdf")
    plt.close()

    return None
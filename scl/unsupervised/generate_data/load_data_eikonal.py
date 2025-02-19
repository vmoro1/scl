# load data for eikonal equation

import numpy as np
import torch

from ..utils import *


def load_eikonal_data(args):
    if args.run_location == 'local':
        data = torch.load('data/unsupervised/eikonal_gear.pt')
    elif args.run_location == 'horeka':
        data = torch.load('/home/hk-project-test-p0021798/st_ac144859/scl/data/unsupervised/eikonal_gear.pt')
    elif args.run_location == 'bro_cluster':
        data = torch.load('/slurm-storage/vigmor/scl/data/unsupervised/eikonal_gear.pt')
    else:
        raise ValueError('Invalid run location')
        
    u_exact = data['sdf']
    u_exact_1d = u_exact.flatten().reshape(-1, 1) 
    X_ic = data['zero_contours'][:, ::-1].copy()
    grid_size = data['grid_size']
    # mask = data['img']

    x = np.linspace(-1, 1, grid_size[0])
    y = np.linspace(-1, 1, grid_size[1])
    xv, yv = np.meshgrid(x, y)
    X_test = np.hstack((xv.flatten().reshape(-1,1), yv.flatten().reshape(-1,1)))

    x_noboundary = x[1:-1]
    y_noboundary = y[1:-1]
    xv_noboundary, yv_noboundary = np.meshgrid(x_noboundary, y_noboundary)
    X_noboundary = np.hstack((xv_noboundary.flatten().reshape(-1,1), yv_noboundary.flatten().reshape(-1,1)))

    X_train_pde = sample_random(X_noboundary, args.num_collocation_pts)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_bc = np.linspace(-1, 1, 10)
    y_bc = np.linspace(-1, 1, 10)
    xv_bc, yv_bc = np.meshgrid(x_bc, y_bc)

    # each array is one boundary
    bc_x_upper = np.hstack((xv_bc[0,:].reshape(-1, 1), yv_bc[0,:].reshape(-1, 1)))
    bc_x_lower = np.hstack((xv_bc[-1,:].reshape(-1, 1), yv_bc[-1,:].reshape(-1, 1)))
    bc_y_left = np.hstack((xv_bc[:,0].reshape(-1, 1), yv_bc[:,0].reshape(-1, 1)))
    bc_y_right = np.hstack((xv_bc[:,-1].reshape(-1, 1), yv_bc[:,-1].reshape(-1, 1)))
    X_bc = np.vstack((bc_x_upper, bc_x_lower, bc_y_left, bc_y_right))

    return X_train_pde, X_bc, X_ic, X_test, u_exact, x, y, u_exact_1d

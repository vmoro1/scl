# Generate ground truth data (analytical solutions) for the Helmholtz equation.

import numpy as np
import torch

from ..utils import *


def generate_helmholtz_data(args, a1, a2):
    x = np.linspace(-1, 1, args.nx)
    y = np.linspace(-1, 1, args.ny)

    xv, yv = np.meshgrid(x, y)
    X_test = np.hstack((xv.flatten().reshape(-1,1), yv.flatten().reshape(-1,1)))

    x_noboundary = x[1:-1]
    y_noboundary = y[1:-1]
    xv_noboundary, yv_noboundary = np.meshgrid(x_noboundary, y_noboundary)
    X_noboundary = np.hstack((xv_noboundary.flatten().reshape(-1,1), yv_noboundary.flatten().reshape(-1,1)))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    X_train_pde = sample_random(X_noboundary, args.num_collocation_pts)

    u_exact = np.sin(a1 * np.pi * xv) * np.sin(a2 * np.pi * yv) 
    u_exact_1d = u_exact.reshape(-1, 1)

    bc_left = np.stack((xv[:,0], yv[:,0]), axis=1)
    bc_right = np.stack((xv[:,-1], yv[:,-1]), axis=1)
    bc_upper = np.stack((xv[0,:], yv[0,:]), axis=1)
    bc_lower = np.stack((xv[-1,:], yv[-1,:]), axis=1)
    bc = np.vstack((bc_left, bc_right, bc_upper, bc_lower))

    bc_u_left = np.sin(a1 * np.pi * bc_left[:,0]) * np.sin(a2 * np.pi * bc_left[:,1])
    bc_u_right = np.sin(a1 * np.pi * bc_right[:,0]) * np.sin(a2 * np.pi * bc_right[:,1])
    bc_u_upper = np.sin(a1 * np.pi * bc_upper[:,0]) * np.sin(a2 * np.pi * bc_upper[:,1])
    bc_u_lower = np.sin(a1 * np.pi * bc_lower[:,0]) * np.sin(a2 * np.pi * bc_lower[:,1])
    bc_u = np.vstack((bc_u_left, bc_u_right, bc_u_upper, bc_u_lower)).reshape(-1, 1)

    return X_train_pde, bc, bc_u, X_test, u_exact, x, y, u_exact_1d


def generate_bc(nx, ny, a1_vec, a2_vec):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xv, yv = np.meshgrid(x, y)
    
    bc_left = np.stack((xv[:,0], yv[:,0]), axis=1)
    bc_right = np.stack((xv[:,-1], yv[:,-1]), axis=1)
    bc_upper = np.stack((xv[0,:], yv[0,:]), axis=1)
    bc_lower = np.stack((xv[-1,:], yv[-1,:]), axis=1)
    bc = np.vstack((bc_left, bc_right, bc_upper, bc_lower))

    a1_vec = a1_vec.cpu().numpy().reshape(-1, 1)
    a2_vec = a2_vec.cpu().numpy().reshape(-1, 1)
    a2_vec = np.array(a2_vec).reshape(-1, 1)
    
    bc_x = bc[:, 0].reshape(-1, 1)
    bc_y = bc[:, 1].reshape(-1, 1)
    
    bc_u = np.sin(np.pi * a1_vec @ bc_x.T) * np.sin(np.pi * a2_vec @ bc_y.T)
    bc_u = np.expand_dims(bc_u, axis=-1)

    # N = bc_x.shape[0]
    # a1_vec = a1_vec.reshape(-1,1,1)
    # a1_vec = np.tile(a1_vec, (1, N, 1))
    # a2_vec = a2_vec.reshape(-1,1,1)
    # a2_vec = np.tile(a2_vec, (1, N, 1))
    # bc_u_alternative = np.sin(np.pi * a1_vec * bc_x) * np.sin(np.pi * a2_vec * bc_y)
    # assert (bc_u == bc_u_alternative).all()
    
    bc_u = torch.tensor(bc_u).float()
    return bc_u

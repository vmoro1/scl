# Generate ground truth data (analytical solutions) for the convection and reaction-diffusion equations.
# It can also generate data for reaction and diffusion only but the data is not used in the experiments.

import numpy as np
import torch.fft

from ..utils import *


def generate_data(args, nu, beta, rho):

    x = np.linspace(0, 2*np.pi, args.num_x_pts, endpoint=False).reshape(-1, 1)      # not inclusive
    t = np.linspace(0, 1, args.num_t_pts).reshape(-1, 1)
    X, T = np.meshgrid(x, t)                                                        # all the X grid points T times, all the T grid points X times
    X_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))                # all data collected 

    # data without initial and boundary condition
    t_noinitial = t[1:]
    x_noboundary = x[1:]   # remove boundary at x=0
    X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
    X_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

    # sample collocation points only from the interior (where the PDE is enforced)
    X_train_pde = sample_random(X_noinitial_noboundary, args.num_collocation_pts)

    # get analytical solutiuon (i.e. targets)
    if 'convection' in args.system or 'diffusion' in args.system:
        u_exact_1d = convection_diffusion(args.u0_str, nu, beta, args.source, args.num_x_pts, args.num_t_pts)
        G = np.full(X_train_pde.shape[0], float(args.source))
    elif 'rd' in args.system:
        u_exact_1d = reaction_diffusion_discrete_solution(args.u0_str, nu, rho, args.num_x_pts, args.num_t_pts)
        G = np.full(X_train_pde.shape[0], float(args.source))
    elif 'reaction' in args.system:
        u_exact_1d = reaction_solution(args.u0_str, rho, args.num_x_pts, args.num_t_pts)
        G = np.full(X_train_pde.shape[0], float(args.source))
    else:
        raise ValueError('System not specified.')   

    u_exact = u_exact_1d.reshape(-1, 1)                
    exact_solution = u_exact.reshape(len(t), len(x))    # exact solution on the (x,t) grid

    x_initial = np.hstack((X[0:1,:].T, T[0:1,:].T))     # initial condition at t=0
    u_initial = exact_solution[0:1,:].T                 # u(x, t) at t=0
    bc_lb = np.hstack((X[:,0:1], T[:,0:1]))             # boundary condition at x = 0 (lower boundary), and t = [0, 1]
    u_bc_lb = exact_solution[:,0:1]                     # u(0, t)

    # generate the other boundary condition, now at x=2pi
    t = np.linspace(0, 1, args.num_t_pts).reshape(-1, 1)
    x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
    bc_ub = np.hstack((x_bc_ub, t))

    u_train_initial = u_initial       # just the initial condition
    X_train_initial = x_initial       # (x,t) for initial condition

    return X_train_initial, u_train_initial, X_train_pde, bc_lb, bc_ub, X_test, u_exact, x, t, exact_solution, u_exact_1d, G


def generate_data_time_generalization(args, nu, beta, rho, t_max=1):

    num_t_pts = int(t_max * args.num_t_pts)                                              # need more points for longer time intervall
    x = np.linspace(0, 2*np.pi, args.num_x_pts, endpoint=False).reshape(-1, 1)      # not inclusive
    t = np.linspace(0, t_max, num_t_pts).reshape(-1, 1)
    X, T = np.meshgrid(x, t)                                                        # all the X grid points T times, all the T grid points X times
    X_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))                # all data collected 

    # data without initial and boundary condition
    t_noinitial = t[1:]
    x_noboundary = x[1:]   # remove boundary at x=0
    X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
    X_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

    # sample collocation points only from the interior (where the PDE is enforced)
    X_train_pde = sample_random(X_noinitial_noboundary, args.num_collocation_pts)

    # get analytical solutiuon (i.e. targets)
    if 'convection' in args.system or 'diffusion' in args.system:
        u_exact_1d = convection_diffusion(args.u0_str, nu, beta, args.source, args.num_x_pts, num_t_pts, t_max)
        G = np.full(X_train_pde.shape[0], float(args.source))
    elif 'rd' in args.system:
        u_exact_1d = reaction_diffusion_discrete_solution(args.u0_str, nu, rho, args.num_x_pts, num_t_pts, t_max)
        G = np.full(X_train_pde.shape[0], float(args.source))
    elif 'reaction' in args.system:
        u_exact_1d = reaction_solution(args.u0_str, rho, args.num_x_pts, num_t_pts, t_max)
        G = np.full(X_train_pde.shape[0], float(args.source))
    else:
        raise ValueError('System not specified.')   

    u_exact = u_exact_1d.reshape(-1, 1)                
    exact_solution = u_exact.reshape(len(t), len(x))    # exact solution on the (x,t) grid

    x_initial = np.hstack((X[0:1,:].T, T[0:1,:].T))     # initial condition at t=0
    u_initial = exact_solution[0:1,:].T                 # u(x, t) at t=0
    bc_lb = np.hstack((X[:,0:1], T[:,0:1]))             # boundary condition at x = 0 (lower boundary), and t = [0, 1]
    u_bc_lb = exact_solution[:,0:1]                     # u(0, t)

    # generate the other boundary condition, now at x=2pi
    t = np.linspace(0, t_max, num_t_pts).reshape(-1, 1)
    x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
    bc_ub = np.hstack((x_bc_ub, t))

    u_train_initial = u_initial       # just the initial condition
    X_train_initial = x_initial       # (x,t) for initial condition

    return X_train_initial, u_train_initial, X_train_pde, bc_lb, bc_ub, X_test, u_exact, x, t, exact_solution, u_exact_1d, G


def function(u0: str):
    """Initial condition, string --> function. Must be appropriate for 
    the chosen PDE system (sin for convection and gauss for reaction and
      reaction-diffusion)."""

    if u0 == 'sin(x)':
        u0 = lambda x: np.sin(x)
    elif u0 == 'sin(pix)':
        u0 = lambda x: np.sin(np.pi*x)
    elif u0 == 'sin^2(x)':
        u0 = lambda x: np.sin(x)**2
    elif u0 == 'sin(x)cos(x)':
        u0 = lambda x: np.sin(x)*np.cos(x)
    elif u0 == '0.1sin(x)':
        u0 = lambda x: 0.1*np.sin(x)
    elif u0 == '0.5sin(x)':
        u0 = lambda x: 0.5*np.sin(x)
    elif u0 == '10sin(x)':
        u0 = lambda x: 10*np.sin(x)
    elif u0 == '50sin(x)':
        u0 = lambda x: 50*np.sin(x)
    elif u0 == '1+sin(x)':
        u0 = lambda x: 1 + np.sin(x)
    elif u0 == '2+sin(x)':
        u0 = lambda x: 2 + np.sin(x)
    elif u0 == '6+sin(x)':
        u0 = lambda x: 6 + np.sin(x)
    elif u0 == '10+sin(x)':
        u0 = lambda x: 10 + np.sin(x)
    elif u0 == 'sin(2x)':
        u0 = lambda x: np.sin(2*x)
    elif u0 == 'tanh(x)':
        u0 = lambda x: np.tanh(x)
    elif u0 == '2x':
        u0 = lambda x: 2*x
    elif u0 == 'x^2':
        u0 = lambda x: x**2
    elif u0 == 'gauss':
        x0 = np.pi
        sigma = np.pi/4
        u0 = lambda x: np.exp(-np.power((x - x0)/sigma, 2.)/2.)
    return u0


def reaction(u, rho, dt):
    """ du/dt = rho*u*(1-u)
    """
    factor_1 = u * np.exp(rho * dt)
    factor_2 = (1 - u)
    u = factor_1 / (factor_2 + factor_1)
    return u


def diffusion(u, nu, dt, IKX2):
    """ du/dt = nu*d2u/dx2
    """
    factor = np.exp(nu * IKX2 * dt)
    u_hat = np.fft.fft(u)
    u_hat *= factor
    u = np.real(np.fft.ifft(u_hat))
    return u


def reaction_solution(u0: str, rho, nx=256, nt=100, t_max=1):
    L = 2*np.pi
    T = t_max
    dx = L/nx
    dt = T/nt
    x = np.arange(0, 2*np.pi, dx)
    t = np.linspace(0, T, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = function(u0)
    u0 = u0(x)

    u = reaction(u0, rho, T)

    u = u.flatten()
    return u


def reaction_diffusion_discrete_solution(u0 : str, nu, rho, nx = 256, nt = 100, t_max=1):
    """ Computes the discrete solution of the reaction-diffusion PDE using
        pseudo-spectral operator splitting.
    Args:
        u0: initial condition
        nu: diffusion coefficient
        rho: reaction coefficient
        nx: size of x-tgrid
        nt: number of points in the t grid
    Returns:
        u: solution
    """
    L = 2*np.pi
    T = t_max
    dx = L/nx
    dt = T/nt
    x = np.arange(0, L, dx) # not inclusive of the last point
    t = np.linspace(0, T, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    u = np.zeros((nx, nt))

    IKX_pos = 1j * np.arange(0, nx/2+1, 1)
    IKX_neg = 1j * np.arange(-nx/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = function(u0)
    u0 = u0(x)

    u[:,0] = u0
    u_ = u0
    for i in range(nt-1):
        u_ = reaction(u_, rho, dt) 
        u_ = diffusion(u_, nu, dt, IKX2)
        u[:,i+1] = u_

    u = u.T
    u = u.flatten()
    return u

def convection_diffusion(u0: str, nu, beta, source=0, xgrid=256, nt=100, t_max=1):
    """Calculate the u solution for convection/diffusion, assuming PBCs.
    Args:
        u0: Initial condition
        nu: viscosity coefficient
        beta: wavespeed coefficient
        source: q (forcing term), option to have this be a constant
        xgrid: size of the x grid
    Returns:
        u_vals: solution
    """

    N = xgrid
    h = 2 * np.pi / N
    x = np.arange(0, 2*np.pi, h) # not inclusive of the last point
    t = np.linspace(0, t_max, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = function(u0)
    u0 = u0(x)

    G = (np.copy(u0)*0)+source # G is the same size as u0

    IKX_pos =1j * np.arange(0, N/2+1, 1)
    IKX_neg = 1j * np.arange(-N/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    uhat0 = np.fft.fft(u0)
    nu_factor = np.exp(nu * IKX2 * T - beta * IKX * T)
    A = uhat0 - np.fft.fft(G)*0 # at t=0, second term goes away
    uhat = A*nu_factor + np.fft.fft(G)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))

    u_vals = u.flatten()
    return u_vals

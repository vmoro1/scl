# SCL'(M) for training a parametric solution for the reaction-difusion equation (parametric dependence thorugh nu and rho)
# Want to find NN f(x,t,nu,rho) that approximates solution everywhere for all (nu,rho) in the domain
# NOTE: pde_param refers to the tuple of PDE parameters (nu, rho) that defines the reaction-diffusion PDE.

# NOTE that the PDE loss is over (x,t,nu,rho) while the data loss (IC + BC) is only over (nu,rho) with fixed (x,t), i.e., the 
# sampling for the PDE samples (x,t,nu,rho) and the sampling for teh data loss only samples nu,rho. IT is very easy to do 
# the data loss for the 3D case as well.

# NOTE: In association with the worst case loss we sometimes use adversarial (or adv) as part of variable names 
# (e.g., refering to the samples as samples_adv) du to the connection between worst-case and adversarial optimization.

# TODO: If it doesn't work, revert to prevoius specific MH sampler

import sys
import os
import argparse
from datetime import datetime
import pickle
import itertools

sys.path.append('.') 

import numpy as np
import torch

from scl.unsupervised.csl_problem import ConstrainedStatisticalLearningProblem
from scl.unsupervised.solver import SimultaneousPrimalDual
from scl.unsupervised.model import MLP

from scl.unsupervised.langevin_monte_carlo import *
from scl.unsupervised.metropolis_hastings import *
from scl.unsupervised.generate_data.generate_data_convection_rd import *
from scl.unsupervised.utils import set_seed
from scl.unsupervised.visualize_solution import *


parser = argparse.ArgumentParser(description='SCL(M) for parametric solution')

parser.add_argument('--system', type=str, default='rd')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--num_collocation_pts', type=int, default=100, help='Number of collocation points.')
parser.add_argument('--epochs', type=int, default=200000, help='Number of epochs to train for.')   
parser.add_argument('--lr_primal', type=float, default=1e-3, help='Learning rate for primal variables (NN parameters).')
parser.add_argument('--lr_dual', type=float, default=1e-4, help='Learning rate for dual variables (lambdas).')
parser.add_argument('--eps', nargs='+', default=[1e-2], help='Tolerances for the constraints.')
parser.add_argument('--use_primal_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the primal variables.')
parser.add_argument('--use_dual_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the dual variables.')

parser.add_argument('--use_mh_sampling', action=argparse.BooleanOptionalAction, default=True, help='Whether to use Metropolis-Hastings sampling (or Langevin Monte Carlo) to sample in order tom compute worst-case losses.')

# arguments for MH sampling of IC, BC and data loss
parser.add_argument('--sampler_batch_size_data', type=int, default=1, help='Batch size for speeding up MH/LMC sampling.')
parser.add_argument('--steps_data', type=int, default=50, help='Number of steps for sampling.')
parser.add_argument('--n_samples_data', type=int, default=25, help='Number of samples to use.')

# arguments for MH sampling of PDE loss
parser.add_argument('--sampler_batch_size_pde', type=int, default=20, help='Batch size for speeding up MH/LMC sampling.')
parser.add_argument('--steps_pde', type=int, default=250, help='Number of steps for sampling.')
parser.add_argument('--n_samples_pde', type=int, default=2500, help='Number of samples to use.')
parser.add_argument('--sigma_x', type=float, default=0.5, help='Standard deviation for the Gaussian proposal in the 3D MH sampling.')
parser.add_argument('--sigma_t', type=float, default=0.1, help='Standard deviation for the Gaussian proposal in the 3D MH sampling.')
parser.add_argument('--sigma_nu', type=float, default=1, help='Standard deviation for the Gaussian proposal in the MH sampling.')
parser.add_argument('--sigma_rho', type=float, default=1, help='Standard deviation for the Gaussian proposal in the MH sampling.')

parser.add_argument('--num_x_pts', type=int, default=256, help='Number of points in the x grid.')
parser.add_argument('--num_t_pts', type=int, default=100, help='Number of points in the time grid.')
parser.add_argument('--domain_nu', nargs='+', default=[1, 5], help='Domain for nu (value that the d^2u/dx^2 term).')
parser.add_argument('--domain_rho', nargs='+', default=[1, 5], help='Domain for rho (value that the u*(1-u) term).')
parser.add_argument('--nu_test', nargs='+', default=[1.0, 2.5, 5.0], help='nu for testing after training.')
parser.add_argument('--rho_test', nargs='+', default=[1.0, 2.5, 5.0], help='rho for testing after training.')

parser.add_argument('--u0_str', default='gauss', help='str argument for initial condition (sin(x) or similiar for convection else gauss)')
parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions of layers of the NN (except the first layer).') 
parser.add_argument('--source', default=0, type=float, help="Source term. Not used.")

parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True, help='Visualize the solution and prediction of the model.')
parser.add_argument('--plot_diagnostics', action=argparse.BooleanOptionalAction, default=True, help='Plot diagnostics of the model.')
parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval_every', type=int, default=100, help='Evaluate the model every n epochs.')
    

class SCL_M_ParametricSolution(ConstrainedStatisticalLearningProblem):
    def __init__(
            self,
            args, 
            model, 
            X_train_initial, 
            u_train_initial,
            bc_lb, 
            bc_ub, 
            device):

        self.model = model
        self.device = device
        eps = [float(eps) for eps in args.eps]

        # data
        self.x_initial = torch.tensor(X_train_initial[:, 0:1], requires_grad=True).float().to(device)
        self.t_initial = torch.tensor(X_train_initial[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)
        self.u_initial = torch.tensor(u_train_initial, requires_grad=True).float().to(device)

        self.samples_pde = []
        # self.samples_BC = []
        # self.samples_IC = []
        self.samples_data = []

        domain_x = [0, 2*np.pi]
        domain_t = [0, 1]
        domain_nu = [float(args.domain_nu[0]), float(args.domain_nu[1])]
        domain_rho = [float(args.domain_rho[0]), float(args.domain_rho[1])]

        # sampling/worst-case loss
        covariance_matrix_pde = torch.tensor([[args.sigma_x**2, 0, 0, 0], [0, args.sigma_t**2, 0, 0], [0, 0, args.sigma_nu**2, 0], [0, 0, 0, args.sigma_rho**2]], device=device, dtype=torch.float32)
        covariance_matrix_data = torch.tensor([[args.sigma_nu**2, 0], [0, args.sigma_rho**2]], device=device, dtype=torch.float32)

        if args.use_mh_sampling:
            self.sampler_pde = MH_Gaussian_4D(self, domain_x, domain_t, domain_nu, domain_rho, args.steps_pde, args.n_samples_pde, self.pde_loss_per_sample, covariance_matrix_pde, device, args.sampler_batch_size_pde)
            # self.sampler_BC = MH_Gaussian_2D(self, domain_nu, domain_rho, args.steps_data, args.n_samples_data, self.BC_loss_per_pde_param, covariance_matrix_data, device, args.sampler_batch_size_data)
            # self.sampler_IC = MH_Gaussian_2D(self, domain_nu, domain_rho, args.steps_data, args.n_samples_data, self.IC_loss_per_pde_param, covariance_matrix_data, device, args.sampler_batch_size_data)
            self.sampler_data = MH_Gaussian_2D(self, domain_nu, domain_rho, args.steps_data, args.n_samples_data, self.data_loss_per_pde_param, covariance_matrix_data, device, args.sampler_batch_size_data)
        else:
            raise NotImplementedError

        self.objective_function = self.worst_case_data_loss
        self.constraints = [self.worst_case_pde_loss]
        self.rhs = eps  

        super().__init__()

    def forward(self, x, t, pde_param):
        """Make a forward pass throught the model. The model takes (x,t, pde_param) as input and predicts the value of the funcion, u."""
        N = x.shape[0]                                                         # number of points
        pde_param = pde_param.reshape(1, -1).repeat(N, 1)                      # shape of pde_param: (2,) --> (N, 2)
        pde_param = pde_param.float().to(self.device)
        u = self.model(torch.cat([x, t, pde_param], dim=1))
        return u

    def forward_data(self, x, t, pde_param):
        """Make a forward pass through the model. The model takes (x,t, pde_param) as input and predicts the value of the funcion, u."""
        N = x.shape[1]                                                                 # number of points
        pde_param = pde_param.unsqueeze(1).repeat(1, N, 1).float().to(self.device)     # shape of pde_param: (batch size, 1) --> (batch size, N, 1)
        u = self.model(torch.cat([x, t, pde_param], dim=-1))
        return u
    
    def forward_pde(self, x, t, pde_param):
        u = self.model(torch.cat([x, t, pde_param], dim=1))
        return u

    def pde_residual(self, x, t, pde_param):
        """ Autograd for calculating the PDE residual, i.e., if the model satisfies the PDE"""
        u = self.forward_pde(x, t, pde_param)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]

        nu = pde_param[:, 0:1]
        rho = pde_param[:, 1:2]

        residual = u_t - nu*u_xx - rho*u + rho*u**2
        return residual

    def boundary_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """Takes derivative of model at the boundary. Used to satisfy periodic boundary condition (BC)."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_x, u_ub_x
    
    # def BC_loss_per_pde_param(self, pde_param):
    #     """Computes the boundary condition loss. per point refers to that we don't 
    #     average over pde_param."""
    #     # expand first dim of boundary tensors based on number of pde params
    #     n = pde_param.shape[0]          # number of different pde params
    #     t_bc_lb_exp = self.t_bc_lb.unsqueeze(0).repeat(n, 1, 1).requires_grad_(True)
    #     x_bc_lb_exp = self.x_bc_lb.unsqueeze(0).repeat(n, 1, 1).requires_grad_(True)
    #     t_bc_ub_exp = self.t_bc_ub.unsqueeze(0).repeat(n, 1, 1).requires_grad_(True)
    #     x_bc_ub_exp = self.x_bc_ub.unsqueeze(0).repeat(n, 1, 1).requires_grad_(True)

    #     u_pred_lb = self.forward_data(x_bc_lb_exp, t_bc_lb_exp, pde_param)
    #     u_pred_ub = self.forward_data(x_bc_ub_exp, t_bc_ub_exp, pde_param)
    #     loss_boundary = torch.mean((u_pred_lb - u_pred_ub) ** 2, dim=-2)

    #     nu = pde_param[:, 0]
    #     if (nu != 0).any():
    #         u_pred_lb_x, u_pred_ub_x = self.boundary_derivatives(u_pred_lb, u_pred_ub, x_bc_lb_exp, x_bc_ub_exp)
    #         loss_boundary += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2, dim=-2)

    #     return loss_boundary
    
    # def worst_case_BC_loss(self):
    #     pde_params_adv = self.sampler_BC()
    #     self.samples_BC.append(pde_params_adv)   
    #     worst_case_loss = self.BC_loss_per_pde_param(pde_params_adv)
    #     worst_case_loss = torch.mean(worst_case_loss)
    #     return worst_case_loss
    
    # def IC_loss_per_pde_param(self, pde_param):
    #     """Computes the initial condition loss. per point refers to that we don't 
    #     average over pde_param."""
    #     # expand first dim of boundary tensors based on number of pde params
    #     n = pde_param.shape[0]          # number of different pde params
    #     t_initial_exp = self.t_initial.unsqueeze(0).repeat(n, 1, 1)
    #     x_initial_exp = self.x_initial.unsqueeze(0).repeat(n, 1, 1)

    #     u_pred_initial = self.forward_data(x_initial_exp, t_initial_exp, pde_param)
    #     loss_initial = torch.mean((self.u_initial - u_pred_initial) ** 2, dim=-2)   # mean over the IC points
    #     return loss_initial
    
    # def worst_case_IC_loss(self):
    #     pde_params_adv = self.sampler_IC()
    #     self.samples_IC.append(pde_params_adv)   
    #     worst_case_loss = self.IC_loss_per_pde_param(pde_params_adv)
    #     worst_case_loss = torch.mean(worst_case_loss)
    #     return worst_case_loss
    
    def data_loss_per_pde_param(self, nu, rho):
        """Computes the data loss. The data loss is computed for points on the 
        boundary and for the initial condition."""
        pde_param = torch.cat([nu, rho], dim=1)

        # expand first dim of boundary tensors based on number of pde params
        n = pde_param.shape[0]          # number of different pde params
        t_initial_exp = self.t_initial.unsqueeze(0).repeat(n, 1, 1)
        x_initial_exp = self.x_initial.unsqueeze(0).repeat(n, 1, 1)
        t_bc_lb_exp = self.t_bc_lb.unsqueeze(0).repeat(n, 1, 1).requires_grad_(True)
        x_bc_lb_exp = self.x_bc_lb.unsqueeze(0).repeat(n, 1, 1).requires_grad_(True)
        t_bc_ub_exp = self.t_bc_ub.unsqueeze(0).repeat(n, 1, 1).requires_grad_(True)
        x_bc_ub_exp = self.x_bc_ub.unsqueeze(0).repeat(n, 1, 1).requires_grad_(True)

        # prediction for initial condition, lower boundary, and upper boundary
        u_pred_initial = self.forward_data(x_initial_exp, t_initial_exp, pde_param)
        u_pred_lb = self.forward_data(x_bc_lb_exp, t_bc_lb_exp, pde_param)
        u_pred_ub = self.forward_data(x_bc_ub_exp, t_bc_ub_exp, pde_param)

        loss_initial = torch.mean((self.u_initial - u_pred_initial) ** 2, dim=-2)
        loss_boundary = torch.mean((u_pred_lb - u_pred_ub) ** 2, dim=-2)

        nu = pde_param[:, 0]
        if (nu != 0).any():
            u_pred_lb_x, u_pred_ub_x = self.boundary_derivatives(u_pred_lb, u_pred_ub, x_bc_lb_exp, x_bc_ub_exp)
            loss_boundary += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2, dim=-2)

        loss = loss_initial + loss_boundary
        return loss
    
    def worst_case_data_loss(self):
        pde_param_adv = self.sampler_data()
        self.samples_data.append(pde_param_adv)    
        worst_case_loss = self.data_loss_per_pde_param(pde_param_adv[:,0:1], pde_param_adv[:,1:2])
        worst_case_loss = torch.mean(worst_case_loss)
        return worst_case_loss
    
    def pde_loss_per_sample(self, x, t, pde_param):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE."""
        x.requires_grad_(True)
        t.requires_grad_(True)
        residual_pde = self.pde_residual(x, t, pde_param)
        loss_pde = residual_pde ** 2
        return loss_pde 
    
    def worst_case_pde_loss(self):
        samples = self.sampler_pde()
        self.samples_pde.append(samples)    # store samples

        x = samples[:,0:1]
        t = samples[:,1:2]
        pde_param = samples[:,2:4]

        worst_case_loss = self.pde_loss_per_sample(x, t, pde_param)
        worst_case_loss = torch.mean(worst_case_loss)
        return worst_case_loss
    
    def predict(self, X, pde_param):
        """Make predictions. Used during evaluation. """
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        pde_param = torch.tensor(pde_param).float()

        self.model.eval()
        u = self.forward(x, t, pde_param)
        u = u.detach().cpu().numpy()

        return u


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert args.system == 'rd'

    nu_test = [float(nu) for nu in args.nu_test]
    rho_test = [float(rho) for rho in args.rho_test]
    beta = 0

    # layers of NN
    layers = [int(item) for item in args.layers.split(',')]

    # get data by solving the PDE
    params_test = list(itertools.product(nu_test, rho_test))    # all combinations of nu and rho
    data_test = {}
    for nu, rho in params_test:
        X_train_initial, u_train_initial, X_train_pde, bc_lb, bc_ub, X_test, u_exact, x, t, exact_solution, u_exact_1d, G = generate_data(args, nu, beta, rho)
        data_pde_param = {'X_train_initial': X_train_initial, 'u_train_initial': u_train_initial, 'X_train_pde': X_train_pde, 'bc_lb': bc_lb, 'bc_ub': bc_ub, 'X_test': X_test, 'u_exact': u_exact, 'x': x, 't': t, 'exact_solution': exact_solution, 'u_exact_1d': u_exact_1d, 'G': G}
        data_test[(nu, rho)] = data_pde_param
    
    # add first layer with correct number of input nodes
    layers.insert(0, X_train_initial.shape[-1] + 2)       # + 2 for nu and rho 

    # set set for reproducibility
    set_seed(args.seed)

    # define constrained learning problem
    model = MLP(layers, 'tanh').to(device)

    # X_train_initial, u_train_initial, bc_lb, bc_ub are independent of beta so we don't need them for different (nu, rho)
    constrained_pinn = SCL_M_ParametricSolution(
        args, 
        model,  
        X_train_initial, 
        u_train_initial, 
        bc_lb, 
        bc_ub, 
        device)

    optimizers = {'primal_optimizer': 'Adam',
                  'use_primal_lr_scheduler': args.use_primal_lr_scheduler,
                'dual_optimizer': 'Adam',
                'use_dual_lr_scheduler': args.use_dual_lr_scheduler}
    
    solver = SimultaneousPrimalDual(
        constrained_pinn, 
        optimizers, 
        args.lr_primal, 
        args.lr_dual, 
        args.epochs, 
        args.eval_every, 
        data_test)

    # solve constrained learning problem
    solver.solve(constrained_pinn)

    # print final values for diagnostics
    print('Final diagnostics:')
    num_dual_variables = len(constrained_pinn.lambdas)
    for i in range(num_dual_variables):
        print(f'Dual variable {i}: ', solver.state_dict['dual_variables'][f'dual_variable_{i}'][-1])
        print(f'Slack_{i}: ', solver.state_dict['slacks'][f'slack_{i}'][-1])
    print('Approximate duality gap: ', solver.state_dict['approximate_duality_gap'][-1])
    print('Approximate relative duality gap: ', solver.state_dict['aproximate_relative_duality_gap'][-1])
    print('Lagrangian: ', solver.state_dict['Lagrangian'][-1])
    print('Primal value: ', solver.state_dict['primal_value'][-1])

    # visualize PINN solution and plot diagnostics
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    suffix = f'scl_m_parametric_solution_rd'
    dir_name = date_time_string  + '_' + suffix

    path_save = f'saved/rd/{dir_name}'

    if args.plot_diagnostics:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        solver.plot(path_save)
    
    if args.save_model:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        torch.save(constrained_pinn.model.state_dict(), f"{path_save}/model.pt")
        torch.save(solver.primal_optimizer.state_dict(), f'{path_save}/primal_optimizer.pt')
        torch.save(solver.dual_optimizer.state_dict(), f'{path_save}/dual_optimizer.pt')

    # make predictions
    preds = {}
    for pde_param in data_test.keys():
        X_test = data_test[pde_param]['X_test']
        u_pred = constrained_pinn.predict(X_test, pde_param)
        preds[pde_param] = u_pred

    if args.visualize:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        
        for pde_param in data_test.keys():
            u_pred_pde_param = preds[pde_param]
            u_pred_pde_param = u_pred_pde_param.reshape(len(t), len(x))
            exact_solution_pde_param = data_test[pde_param]['exact_solution']
            u_exact_1d_pde_param = data_test[pde_param]['u_exact_1d']

            plot_exact_u(exact_solution_pde_param, x, t, path=path_save, label_x='x', label_y='t', suffix=f'_{pde_param}', flip_axis_plotting=True)
            plot_u_diff(exact_solution_pde_param, u_pred_pde_param, x, t, path=path_save, label_x='x', label_y='t', suffix=f'_{pde_param}', flip_axis_plotting=True)
            plot_u_pred(u_exact_1d_pde_param, u_pred_pde_param, x, t, path=path_save, label_x='x', label_y='t', suffix=f'_{pde_param}', flip_axis_plotting=True)
    


    # save arguments
    with open(f'{path_save}/args.txt', 'a') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')

    # test model and print metrics
    for pde_param in data_test.keys():
        u_exact_pde_param = data_test[pde_param]['u_exact']
        u_pred_pde_param = preds[pde_param]
        u_pred_pde_param = u_pred_pde_param.reshape(-1,1)
        error_u_relative = np.linalg.norm(u_exact_pde_param - u_pred_pde_param, 2) / np.linalg.norm(u_exact_pde_param, 2)
        error_u_abs = np.mean(np.abs(u_exact_pde_param - u_pred_pde_param))
        error_u_linf = np.linalg.norm(u_exact_pde_param - u_pred_pde_param, np.inf) / np.linalg.norm(u_exact_pde_param, np.inf)

        print(f'pde_param: {pde_param}')
        print('Test metrics:')
        print('Relative error: %e' % (error_u_relative))
        print('Absolute error: %e' % (error_u_abs))
        print('linf error: %e' % (error_u_linf))
        print('')
        
        with open(f'{path_save}/args.txt', 'a') as file:
            file.write(f'pde_param: {pde_param}\n')
            file.write('Test metrics:\n')
            file.write(f'Relative error: {error_u_relative}\n')
            file.write(f'Absolute error: {error_u_abs}\n')
            file.write(f'linf error: {error_u_linf}\n')

    # save state dict
    with open(f'{path_save}/state_dict.pkl', 'wb') as f:
        pickle.dump(solver.state_dict, f)

    # saved samples
    samples_pde = constrained_pinn.samples_pde
    samples_pde = torch.stack(samples_pde, dim=0)
    samples_pde = samples_pde.detach().cpu()
    torch.save(samples_pde, f'{path_save}/samples_pde.pt')
    assert samples_pde.shape == (args.epochs, args.n_samples_pde, 4)

    # only non-empty when using IC and BC loss together as the objective
    samples_data = constrained_pinn.samples_data
    samples_data = torch.stack(samples_data, dim=0)
    samples_data = samples_data.detach().cpu()
    torch.save(samples_data, f'{path_save}/samples_data.pt')
    assert samples_data.shape == (args.epochs, args.n_samples_data, 2)


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')

# SCL'(M) for training a parametric solution for the Helmholtz equation (parametric dependence thorugh a1 and a2)
# Want to find NN f(x,t,a1,a2) that approximates solution everywhere for all (a1,a2) in the domain
# NOTE: pde_param refers to the tuple of PDE parameters (a1,a2) that defines the Helmholtz PDE.

# NOTE that the PDE loss is over (x,t,a1,a2) while the data loss (IC + BC) is only over (a1,a2) with fixed (x,t), i.e., the 
# sampling for the PDE samples (x,t,a1,a2) and the sampling for teh data loss only samples a1,a2. IT is very easy to do 
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
import math

sys.path.append('.') 

import numpy as np
import torch

from scl.unsupervised.csl_problem import ConstrainedStatisticalLearningProblem
from scl.unsupervised.solver import SimultaneousPrimalDual
from scl.unsupervised.model import MLP

from scl.unsupervised.langevin_monte_carlo import *
from scl.unsupervised.metropolis_hastings import *
from scl.unsupervised.generate_data.generate_data_helmholtz import *
from scl.unsupervised.utils import set_seed
from scl.unsupervised.visualize_solution import *

parser = argparse.ArgumentParser(description='SCL(M) for parametric solution')

parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--num_collocation_pts', type=int, default=1000, help='Number of collocation points.')
parser.add_argument('--epochs', type=int, default=200000, help='Number of epochs to train for.')   
parser.add_argument('--lr_primal', type=float, default=1e-3, help='Learning rate for primal variables (NN parameters).')
parser.add_argument('--lr_dual', type=float, default=1e-4, help='Learning rate for dual variables (lambdas).')
parser.add_argument('--eps', nargs='+', default=[5e-1], help='Tolerances for the constraints.')
parser.add_argument('--use_primal_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the primal variables.')
parser.add_argument('--use_dual_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the dual variables.')
parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions of layers of the NN (except the first layer).')

parser.add_argument('--use_mh_sampling', action=argparse.BooleanOptionalAction, default=True, help='Whether to use Metropolis-Hastings sampling (or Langevin Monte Carlo) to sample from psi_alpha for computing worst-case losses.')

# arguments for MH sampling of IC, BC and data loss
parser.add_argument('--sampler_batch_size_data', type=int, default=1, help='Batch size for speeding up MH/LMC sampling.')
parser.add_argument('--steps_data', type=int, default=50, help='Number of steps for sampling.')
parser.add_argument('--n_samples_data', type=int, default=25, help='Number of samples to use.')

# arguments for MH sampling of PDE loss
parser.add_argument('--sampler_batch_size_pde', type=int, default=20, help='Batch size for speeding up MH/LMC sampling.')
parser.add_argument('--steps_pde', type=int, default=250, help='Number of steps for sampling.')
parser.add_argument('--n_samples_pde', type=int, default=2500, help='Number of samples to use.')

parser.add_argument('--sigma_x', type=float, default=0.2, help='Standard deviation for the Gaussian proposal in the 3D MH sampling.')
parser.add_argument('--sigma_y', type=float, default=0.2, help='Standard deviation for the Gaussian proposal in the 3D MH sampling.')
parser.add_argument('--sigma_a1', type=float, default=0.2, help='Standard deviation for the Gaussian proposal in the MH sampling.')
parser.add_argument('--sigma_a2', type=float, default=0.2, help='Standard deviation for the Gaussian proposal in the MH sampling.')

parser.add_argument('--nx', type=int, default=256, help='Number of points in the x grid.')
parser.add_argument('--ny', type=int, default=256, help='Number of points in the y grid.')
parser.add_argument('--domain_a1', nargs='+', default=[1.7, 2.0], help='Domain for a1')
parser.add_argument('--domain_a2', nargs='+', default=[1.7, 2.0], help='Domain for a2')
parser.add_argument('--a1_test', nargs='+', default=[1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0], help='a1 for testing after training.')
parser.add_argument('--a2_test', nargs='+', default=[1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0], help='a2 for testing after training.')

parser.add_argument('--k', type=float, default=1, help='k value for the Helmholtz PDE.')

parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True, help='Visualize the solution and prediction of the model.')
parser.add_argument('--plot_diagnostics', action=argparse.BooleanOptionalAction, default=True, help='Plot diagnostics of the model.')
parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval_every', type=int, default=100, help='Evaluate the model every n epochs.')


class SCL_M_ParametricSolution(ConstrainedStatisticalLearningProblem):
    def __init__(
            self,
            args, 
            model, 
            bc,
            device):

        self.model = model
        self.device = device
        self.k = args.k
        self.nx = args.nx
        self.ny = args.ny
        eps = [float(eps) for eps in args.eps]

        # data
        self.x_bc = torch.tensor(bc[:, 0:1]).float().to(device)
        self.y_bc = torch.tensor(bc[:, 1:2]).float().to(device)

        # adversarial/Linf losses
        self.samples_pde = []
        self.samples_BC = []

        domain_x = [-1, 1]
        domain_y = [-1, 1]
        domain_a1 = [float(args.domain_a1[0]), float(args.domain_a1[1])]
        domain_a2 = [float(args.domain_a2[0]), float(args.domain_a2[1])]
        covariance_matrix_pde = torch.tensor([[args.sigma_x**2, 0, 0, 0], [0, args.sigma_y**2, 0, 0], [0, 0, args.sigma_a1**2, 0], [0, 0, 0, args.sigma_a2**2]], device=device, dtype=torch.float32)
        covariance_matrix_data = torch.tensor([[args.sigma_a1**2, 0], [0, args.sigma_a2**2]], device=device, dtype=torch.float32)

        # sampling/worst-case loss
        if args.use_mh_sampling:
            self.sampler_pde = MH_Gaussian_4D(self, domain_x, domain_y, domain_a1, domain_a2, args.steps_pde, args.n_samples_pde, self.pde_loss_per_sample, covariance_matrix_pde, device, args.sampler_batch_size_pde)
            self.sampler_bc = MH_Gaussian_2D(self, domain_a1, domain_a2, args.steps_data, args.n_samples_data, self.bc_loss_per_pde_param, covariance_matrix_data, device, args.sampler_batch_size_data)
        else:
            raise NotImplementedError

        self.objective_function = self.worst_case_bc_loss
        self.constraints = [self.worst_case_pde_loss]
        self.rhs = eps  

        super().__init__()

    def forward(self, x, y, pde_param):
        """Make a forward pass throught the model. The model takes (x,y, pde_param) as input and predicts the value of the funcion, u."""
        N = x.shape[0]                                                         # number of points
        pde_param = pde_param.reshape(1, -1).repeat(N, 1)                      # shape of pde_param: (2,) --> (N, 2)
        pde_param = pde_param.float().to(self.device)
        u = self.model(torch.cat([x, y, pde_param], dim=1))
        return u

    def forward_data(self, x, y, pde_param):
        """Make a forward pass through the model. The model takes (x,y, pde_param) as input and predicts the value of the funcion, u."""
        N = x.shape[1]                                                                 # number of points
        pde_param = pde_param.unsqueeze(1).repeat(1, N, 1).float().to(self.device)     # shape of pde_param: (batch size, 2) --> (batch size, N, 2)
        u = self.model(torch.cat([x, y, pde_param], dim=-1))
        return u
    
    def forward_pde(self, x, y, pde_param):
        u = self.model(torch.cat([x, y, pde_param], dim=1))
        return u

    def pde_residual(self, x, y, pde_param):
        """ Autograd for calculating the PDE residual, i.e., if the model satisfies the PDE"""
        u = self.forward_pde(x, y, pde_param)

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y,
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
        
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
            )[0]
        
        a1 = pde_param[:, 0:1]
        a2 = pde_param[:, 1:2]
        q = self.get_forcing(x, y, a1, a2)
        residual = u_xx + u_yy + (self.k ** 2) * u - q
        
        return residual
    
    def get_forcing(self, x, y, a1, a2):
        """Get the forcing term for the 2D Helmholtz PDE."""
        q = - ((a1 * math.pi) ** 2) * torch.sin(a1 * math.pi * x)*torch.sin(a2 * math.pi * y) \
                - ((a2 * math.pi) ** 2) * torch.sin(a1 * math.pi * x) * torch.sin(a2 * math.pi * y) \
                + (self.k ** 2) * torch.sin(a1 * math.pi * x) * torch.sin(a2 * math.pi * y)
        return q
    
    def bc_loss_per_pde_param(self, a1, a2):
        pde_param = torch.cat([a1, a2], dim=1)
        bc_u = generate_bc(self.nx, self.ny, a1, a2)

        # expand first dim of boundary tensors based on number of pde params
        n = pde_param.shape[0]          # number of different pde params
        x_bc_exp = self.x_bc.unsqueeze(0).repeat(n, 1, 1)
        y_bc_exp = self.y_bc.unsqueeze(0).repeat(n, 1, 1)

        u_pred_bc = self.forward_data(x_bc_exp, y_bc_exp, pde_param)
        loss_bc = torch.mean((u_pred_bc - bc_u) ** 2, dim=-2)
        return loss_bc

    def worst_case_bc_loss(self):
        pde_params_adv = self.sampler_bc()
        self.samples_BC.append(pde_params_adv)
        worst_case_loss = self.bc_loss_per_pde_param(pde_params_adv[:,0:1], pde_params_adv[:,1:2])
        worst_case_loss = torch.mean(worst_case_loss)
        return worst_case_loss
    
    def pde_loss_per_sample(self, x, y, pde_param):
        x.requires_grad_(True)
        y.requires_grad_(True)
        residual_pde = self.pde_residual(x, y, pde_param)
        loss_pde = residual_pde ** 2
        return loss_pde 
    
    def worst_case_pde_loss(self):
        samples = self.sampler_pde()
        self.samples_pde.append(samples)    # store samples

        x = samples[:,0:1]
        y = samples[:,1:2]
        pde_param = samples[:,2:4]

        worst_case_loss = self.pde_loss_per_sample(x, y, pde_param)
        worst_case_loss = torch.mean(worst_case_loss)
        return worst_case_loss
    
    def predict(self, X, pde_param):
        """Make predictions. Used during evaluation. """
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        pde_param = torch.tensor(pde_param).float()

        self.model.eval()
        u = self.forward(x, y, pde_param)
        u = u.detach().cpu().numpy()

        return u


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    a1_test = [float(a1) for a1 in args.a1_test]
    a2_test = [float(a2) for a2 in args.a2_test]

    # layers of NN
    layers = [int(item) for item in args.layers.split(',')]

    # get data by solving the PDE
    params_test = list(itertools.product(a1_test, a2_test))    # all combinations of a1 and a2
    data_test = {}
    for a1, a2 in params_test:
        X_train_pde, bc, bc_u, X_test, u_exact, x, y, u_exact_1d = generate_helmholtz_data(args, a1, a2)
        data_pde_param = {'X_train_pde': X_train_pde, 'bc': bc, 'bc_u': bc_u, 'X_test': X_test, 'u_exact': u_exact, 'x': x, 'y': y, 'u_exact_1d': u_exact_1d}
        data_test[(a1, a2)] = data_pde_param
    
    # add first layer with correct number of input nodes
    layers.insert(0, X_test.shape[-1] + 2)       # + 2 for a1 and a2 (Helmholtz PDE params)

    # set set for reproducibility
    set_seed(args.seed)

    # define constrained learning problem
    model = MLP(layers, 'tanh').to(device)
    constrained_pinn = SCL_M_ParametricSolution(
        args, 
        model,  
        bc,
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
    suffix = f'scl_m_parametric_solution_helmholtz'
    dir_name = date_time_string  + '_' + suffix

    path_save = f'saved/helmholtz/{dir_name}'

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
            u_pred_pde_param = u_pred_pde_param.reshape(len(y), len(x))
            u_exact_pde_param = data_test[pde_param]['u_exact']
            u_exact_1d_pde_param = data_test[pde_param]['u_exact_1d']

            plot_exact_u(u_exact_pde_param, x, y, path_save, label_x='x', label_y='y', suffix=f'_{pde_param}', flip_axis_plotting=False)
            plot_u_diff(u_exact_pde_param, u_pred_pde_param, x, y, path_save, label_x='x', label_y='y', suffix=f'_{pde_param}', flip_axis_plotting=False)
            plot_u_pred(u_exact_1d_pde_param, u_pred_pde_param, x, y, path_save, label_x='x', label_y='y', suffix=f'_{pde_param}', flip_axis_plotting=False)

    # save arguments
    with open(f'{path_save}/args.txt', 'a') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')

    # test model and print metrics
    for pde_param in data_test.keys():
        u_exact_1d_pde_param = data_test[pde_param]['u_exact_1d']
        u_pred_pde_param = preds[pde_param]
        u_pred_pde_param = u_pred_pde_param.reshape(-1,1)

        error_u_relative = np.linalg.norm(u_exact_1d_pde_param - u_pred_pde_param, 2) / np.linalg.norm(u_exact_1d_pde_param, 2)
        error_u_abs = np.mean(np.abs(u_exact_1d_pde_param - u_pred_pde_param))
        error_u_linf = np.linalg.norm(u_exact_1d_pde_param - u_pred_pde_param, np.inf) / np.linalg.norm(u_exact_1d_pde_param, np.inf)

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

    samples_BC = constrained_pinn.samples_BC
    samples_BC = torch.stack(samples_BC, dim=0)
    samples_BC = samples_BC.detach().cpu()
    torch.save(samples_BC, f'{path_save}/samples_BC.pt')
    assert samples_BC.shape == (args.epochs, args.n_samples_data, 2)


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')

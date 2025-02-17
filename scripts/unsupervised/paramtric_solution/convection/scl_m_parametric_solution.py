# SCL'(M) for training a parametric solution for the convection equation (parametric dependence thorugh convection parameter beta)
# Want to find NN f(x,t,beta) that approximates solution everywhere for all betas in teh domain

# NOTE that the PDE loss is over (x,t,beta) while the data loss (IC + BC) is only over beta with fixed (x,t), i.e., the 
# sampling for the PDE samples (x,t,beta) and the sampling for teh data loss only samples beta. IT is very easy to do 
# the data loss for the 3D case as well.

# NOTE: In association with the worst case loss we sometimes use adversarial (or adv) as part of variable names 
# (e.g., refering to the samples as samples_adv) du to the connection between worst-case and adversarial optimization.


import sys
import os
import argparse
from datetime import datetime
import pickle

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


parser = argparse.ArgumentParser(description='SCL(M) parametric solution')

parser.add_argument('--system', type=str, default='convection')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--num_collocation_pts', type=int, default=100, help='Number of collocation points.')
parser.add_argument('--epochs', type=int, default=200000, help='Number of epochs to train for.')   
parser.add_argument('--lr_primal', type=float, default=1e-3, help='Learning rate for primal variables (NN parameters).')
parser.add_argument('--lr_dual', type=float, default=1e-4, help='Learning rate for dual variables (lambdas).')
parser.add_argument('--eps', nargs='+', default=[1e-3], help='Tolerances for the constraints.')
parser.add_argument('--use_primal_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the primal variables.')
parser.add_argument('--use_dual_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the dual variables.')

parser.add_argument('--use_mh_sampling', action=argparse.BooleanOptionalAction, default=True, help='Whether to use Metropolis-Hastings sampling (or Langevin Monte Carlo) to sample from psi_alpha in order to compute worst-case loss')

# arguments for MH sampling for data loss
parser.add_argument('--sampler_batch_size_data', type=int, default=1, help='Batch size for speeding up MH/LMC sampling.')
parser.add_argument('--steps_data', type=int, default=50, help='Number of steps for sampling.')
parser.add_argument('--n_samples_data', type=int, default=25, help='Number of samples to use.')
parser.add_argument('--sigma', type=float, default=3, help='Standard deviation for the Gaussian proposal in the 1D MH sampling.')

# arguments for MH sampling for PDE loss
parser.add_argument('--sampler_batch_size_pde', type=int, default=20, help='Batch size for speeding up MH/LMC sampling.')
parser.add_argument('--steps_pde', type=int, default=250, help='Number of steps for sampling.')
parser.add_argument('--n_samples_pde', type=int, default=2500, help='Number of samples to use.')
parser.add_argument('--sigma_x', type=float, default=0.5, help='Standard deviation for the Gaussian proposal in the 3D MH sampling.')
parser.add_argument('--sigma_t', type=float, default=0.1, help='Standard deviation for the Gaussian proposal in the 3D MH sampling.')
parser.add_argument('--sigma_beta', type=float, default=3, help='Standard deviation for the Gaussian proposal in the 3D MH sampling.')

parser.add_argument('--num_x_pts', type=int, default=256, help='Number of points in the x grid.')
parser.add_argument('--num_t_pts', type=int, default=100, help='Number of points in the time grid.')
parser.add_argument('--domain_beta', nargs='+', default=[1, 30], help='Domain of beta.')
parser.add_argument('--betas_test', nargs='+', default=[1.0, 5.0, 10.0, 20.0, 30.0], help='betas for testing.')

parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition (sin(x) or similiar for convection else gauss)')
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

        domain_x = [0, 2*np.pi]
        domain_t = [0, 1]
        domain_beta = [float(args.domain_beta[0]), float(args.domain_beta[1])]

        # sampling/worst-case loss
        self.samples_pde = []
        self.samples_data = []
        covariance_matrix = torch.tensor([[args.sigma_x**2, 0, 0], [0, args.sigma_t**2, 0], [0, 0, args.sigma_beta**2]], device=device, dtype=torch.float32)
        if args.use_mh_sampling:
            self.sampler_pde = MH_Gaussian_3D(self, domain_x, domain_t, domain_beta, args.steps_pde, args.n_samples_pde, self.pde_loss_per_sample, covariance_matrix, device, args.sampler_batch_size_pde)
            # self.sampler_BC = MH_Gaussian_1D(self, domain_beta, args.steps_data, args.n_samples_data, self.BC_loss_per_beta, args.sigma, device, args.sampler_batch_size_data)
            # self.sampler_IC = MH_Gaussian_1D(self, domain_beta, args.steps_data, args.n_samples_data, self.IC_loss_per_beta, args.sigma, device, args.sampler_batch_size_data)
            self.sampler_data = MH_Gaussian_1D(self, domain_beta, args.steps_data, args.n_samples_data, self.data_loss_per_beta, args.sigma, device, args.sampler_batch_size_data)
        else:
            raise NotImplementedError
        
        self.objective_function = self.worst_case_data_loss
        self.constraints = [self.worst_case_pde_loss]
        self.rhs = eps  

        super().__init__()

    def forward(self, x, t, beta):
        """Make a forward pass throught the model. The model takes (x,t, beta) as input and predicts the value of the funcion, u."""
        beta = torch.full_like(x, beta).float().to(self.device)
        u = self.model(torch.cat([x, t, beta], dim=1))
        return u

    def forward_data(self, x, t, beta):
        """Make a forward pass through the model. The model takes (x,t, beta) as input and predicts the value of the funcion, u."""
        N = x.shape[0]                                                      # number of points
        batch_size = beta.shape[0]                                          # batch size for sampling betas in MH sampling             
        x = x.unsqueeze(0).repeat(batch_size, 1, 1)                         # shape of x: (N, 1) --> (batch size, N, 1)
        t = t.unsqueeze(0).repeat(batch_size, 1, 1)                         # shape of t: (N, 1) --> (batch size, N, 1)
        beta = beta.unsqueeze(1).repeat(1, N, 1).float().to(self.device)    # shape of beta: (batch size, 1) --> (batch size, N, 1)
        u = self.model(torch.cat([x, t, beta], dim=-1))
        return u
    
    def forward_pde(self, x, t, beta):
        u = self.model(torch.cat([x, t, beta], dim=1))
        return u

    def pde_residual(self, x, t, beta):
        """ Autograd for calculating the PDE residual, i.e., if the model satisfies the PDE"""
        u = self.forward_pde(x, t, beta)

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
        
        residual = u_t + beta*u_x
        return residual

    # def BC_loss_per_beta(self, beta,):
    #     """Computes the boundary condition loss. per point refers to that we don't 
    #     average over beta."""
    #     u_pred_lb = self.forward_data(self.x_bc_lb, self.t_bc_lb, beta)
    #     u_pred_ub = self.forward_data(self.x_bc_ub, self.t_bc_ub, beta)
        
    #     loss_boundary = torch.mean((u_pred_lb - u_pred_ub) ** 2, dim=-2)
    #     return loss_boundary
    
    # def worst_case_BC_loss(self):
    #     betas_adv = self.sampler_BC()
    #     self.samples_BC.append(betas_adv)   
        
    #     worst_case_loss = self.BC_loss_per_beta(betas_adv)
    #     worst_case_loss = torch.mean(worst_case_loss)
    #     return worst_case_loss
    
    # def IC_loss_per_beta(self, beta):
    #     """Computes the initial condition loss. per point refers to that we don't 
    #     average over beta."""
    #     u_pred_initial = self.forward_data(self.x_initial, self.t_initial, beta)
    #     loss_initial = torch.mean((self.u_initial - u_pred_initial) ** 2, dim=-2)   # mean over the IC points
    #     return loss_initial
    
    # def worst_case_IC_loss(self):
    #     betas_adv = self.sampler_IC()
    #     self.samples_IC.append(betas_adv)   
        
    #     worst_case_loss = self.IC_loss_per_beta(betas_adv)
    #     worst_case_loss = torch.mean(worst_case_loss)
    #     return worst_case_loss
    
    def data_loss_per_beta(self, beta, dummy_1=None, dummy_2=None):
        """Computes the data loss. The data loss is computed for points on the 
        boundary and for the initial condition. per point refers to that we don't 
        average over beta."""

        # prediction for initial condition, lower boundary, and upper boundary
        u_pred_initial = self.forward_data(self.x_initial, self.t_initial, beta)
        u_pred_lb = self.forward_data(self.x_bc_lb, self.t_bc_lb, beta)
        u_pred_ub = self.forward_data(self.x_bc_ub, self.t_bc_ub, beta)

        loss_initial = torch.mean((self.u_initial - u_pred_initial) ** 2, dim=-2)
        loss_boundary = torch.mean((u_pred_lb - u_pred_ub) ** 2, dim=-2)

        loss = loss_initial + loss_boundary
        return loss
    
    def worst_case_data_loss(self):
        betas_adv = self.sampler_data()
        self.samples_data.append(betas_adv)    # store sampled betas
        worst_case_loss = self.data_loss_per_beta(betas_adv)
        worst_case_loss = torch.mean(worst_case_loss)
        return worst_case_loss
    
    def pde_loss_per_sample(self, x, t, beta):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE.
        Per beta refers to that we don't average over beta but only over the collocation points."""
        x.requires_grad_(True)
        t.requires_grad_(True)

        residual_pde = self.pde_residual(x, t, beta)
        loss_pde = residual_pde ** 2
        return loss_pde 
    
    def worst_case_pde_loss(self):
        samples = self.sampler_pde()
        self.samples_pde.append(samples)    # store samples

        x = samples[:,0:1]
        t = samples[:,1:2]
        beta = samples[:,2:3]

        worst_case_loss = self.pde_loss_per_sample(x, t, beta)
        worst_case_loss = torch.mean(worst_case_loss)
        return worst_case_loss
    
    def predict(self, X, beta):
        """Make predictions. Used during evaluation. """
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        self.model.eval()
        u = self.forward(x, t, beta)
        u = u.detach().cpu().numpy()

        return u


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert args.system == 'convection' 

    nu = 0
    rho = 0
    betas_test = [float(beta) for beta in args.betas_test]   

    # layers of NN
    layers = [int(item) for item in args.layers.split(',')]

    # data
    data_test = {}
    for beta in betas_test:
        X_train_initial, u_train_initial, X_train_pde, bc_lb, bc_ub, X_test, u_exact, x, t, exact_solution, u_exact_1d, G = generate_data(args, nu, beta, rho)
        dict_beta = {'X_train_initial': X_train_initial, 'u_train_initial': u_train_initial, 'X_train_pde': X_train_pde, 'bc_lb': bc_lb, 'bc_ub': bc_ub, 'X_test': X_test, 'u_exact': u_exact, 'x': x, 't': t, 'exact_solution': exact_solution, 'u_exact_1d': u_exact_1d, 'G': G}
        data_test[beta] = dict_beta
    
    # add first layer with correct number of input nodes
    layers.insert(0, X_train_initial.shape[-1] + 1)      # + 1 for convection parameter beta

    # set set for reproducibility
    set_seed(args.seed)

    # define constrained learning problem
    model = MLP(layers, 'tanh').to(device)

    # NOTE: X_train_initial, u_train_initial, bc_lb, bc_ub are same for all betas so we don't need them for any particular beta
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
    suffix = f'scl_m_parametric_solution_convection'
    dir_name = date_time_string  + '_' + suffix

    path_save = f'saved/convection/{dir_name}'

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
    for beta in betas_test:
        u_pred = constrained_pinn.predict(X_test, beta)
        preds[beta] = u_pred

    if args.visualize:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        
        for beta in betas_test:
            u_pred_beta = preds[beta]
            u_pred_beta = u_pred_beta.reshape(len(t), len(x))
            exact_solution_beta = data_test[beta]['exact_solution']
            u_exact_1d_beta = data_test[beta]['u_exact_1d']

            plot_exact_u(exact_solution_beta, x, t, path=path_save, label_x='x', label_y='t', suffix=f'_{beta}', flip_axis_plotting=True)
            plot_u_diff(exact_solution_beta, u_pred_beta, x, t, path=path_save, label_x='x', label_y='t', suffix=f'_{beta}', flip_axis_plotting=True)
            plot_u_pred(u_exact_1d_beta, u_pred_beta, x, t, path=path_save, label_x='x', label_y='t',suffix=f'_{beta}', flip_axis_plotting=True)
    

    # save arguments
    with open(f'{path_save}/args.txt', 'a') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')

    # test model and print metrics
    for beta in betas_test:
        u_exact_beta = data_test[beta]['u_exact']
        u_pred_beta = preds[beta]
        u_pred_beta = u_pred_beta.reshape(-1,1)
        error_u_relative = np.linalg.norm(u_exact_beta-u_pred_beta, 2)/np.linalg.norm(u_exact_beta, 2)
        error_u_abs = np.mean(np.abs(u_exact_beta - u_pred_beta))
        error_u_linf = np.linalg.norm(u_exact_beta - u_pred_beta, np.inf)/np.linalg.norm(u_exact_beta, np.inf)

        print(f'Beta: {beta}')
        print('Test metrics:')
        print('Relative error: %e' % (error_u_relative))
        print('Absolute error: %e' % (error_u_abs))
        print('L_inf error: %e' % (error_u_linf))
        print('')
        
        with open(f'{path_save}/args.txt', 'a') as file:
            file.write(f'Beta: {beta}\n')
            file.write('Test metrics:\n')
            file.write(f'Relative error: {error_u_relative}\n')
            file.write(f'Absolute error: {error_u_abs}\n')
            file.write(f'L_inf error: {error_u_linf}\n')

    # save state dict
    with open(f'{path_save}/state_dict.pkl', 'wb') as f:
        pickle.dump(solver.state_dict, f)

    # saved samples
    samples_pde = constrained_pinn.samples_pde
    samples_pde = torch.stack(samples_pde, dim=0)
    samples_pde = samples_pde.detach().cpu()
    torch.save(samples_pde, f'{path_save}/samples_pde.pt')
    assert samples_pde.shape == (args.epochs, args.n_samples_pde, 3)

    samples_data = constrained_pinn.samples_data
    samples_data = torch.cat(samples_data, dim=1)
    samples_data = samples_data.T.detach().cpu()
    torch.save(samples_data, f'{path_save}/samples_data.pt')
    assert samples_data.shape == (args.epochs, args.n_samples_data)


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')

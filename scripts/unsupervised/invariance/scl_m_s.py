# SCL(M+S) used in invariance experiments
# Fixed collocation points are used (i.e., no worst-case PDE loss/stochastic collocation points)
#  and the structural (invariance) constraint is incorporated as a worst-case loss.

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


parser = argparse.ArgumentParser(description='SCL(M+S)')

parser.add_argument('--system', type=str, default='convection', help='PDE system. Should be fixed but needed for data generation.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--num_collocation_pts', type=int, default=100, help='Number of collocation points.')
parser.add_argument('--fixed_collocation_pts', action=argparse.BooleanOptionalAction, default=True, help='Whether to use fixed collocation points or sample new ones at each epoch.')
parser.add_argument('--epochs', type=int, default=200000, help='Number of epochs to train for.')
parser.add_argument('--lr_primal', type=float, default=1e-3, help='Learning rate for primal variables (NN parameters).')
parser.add_argument('--lr_dual', type=float, default=1e-4, help='Learning rate for dual variables (lambdas).')
parser.add_argument('--eps', nargs='+', default=[1e-3, 1e-3], help='Tolerances for the constraints.')
parser.add_argument('--use_primal_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the primal variables.')
parser.add_argument('--use_dual_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the dual variables.')

parser.add_argument('--use_mh_sampling', action=argparse.BooleanOptionalAction, default=True, help='Whether to use Metropolis-Hastings sampling (or Langevin Monte Carlo) to sample from psi_alpha to compute worst-case loss.')
parser.add_argument('--use_mh_projection', action=argparse.BooleanOptionalAction, default=False, help='Whether to use projection in MH sampling (alternative is to reject proposed samples outside of the domain).')
parser.add_argument('--causal', action=argparse.BooleanOptionalAction, default=True, help='Whether to start sampling from t=0 or from a random t. Starting from t=0 is similair to causal training of PINNs')
parser.add_argument('--sampler_batch_size', type=int, default=20, help='Batch size for speeding up MH/LMC sampling.')
parser.add_argument('--eta', type=float, default=1e-3, help='Step size for LMC sampling.')
parser.add_argument('--T', type=float, default=1.0, help='Temperature for LMC sampling.')
parser.add_argument('--steps', type=int, default=250, help='Number of steps for sampling.')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to use.')
parser.add_argument('--sigma_x', type=float, default=0.5, help='Standard deviation for the Gaussian proposal in the MH sampling.')
parser.add_argument('--sigma_t', type=float, default=0.1, help='Standard deviation for the Gaussian proposal in the MH sampling.')
parser.add_argument('--sigma_xt', type=float, default=0.0, help='Standard deviation for the Gaussian proposal in the MH sampling.')

parser.add_argument('--num_x_pts', type=int, default=256, help='Number of points in the x grid.')
parser.add_argument('--num_t_pts', type=int, default=100, help='Number of points in the time grid.')
parser.add_argument('--beta', type=float, default=30.0, help='Convection parameter beta')

parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition (sin(x) or similiar for convection else gauss)')
parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions of layers of the NN (except the first layer).')
parser.add_argument('--source', default=0, type=float, help="Source term. Not used.")

parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True, help='Visualize the solution and prediction of the model.')
parser.add_argument('--plot_diagnostics', action=argparse.BooleanOptionalAction, default=True, help='Plot diagnostics of the model.')
parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval_every', type=int, default=100, help='Evaluate the model every n epochs.')


class SCL_M_S(ConstrainedStatisticalLearningProblem):
    def __init__(
            self,
            args,  
            model, 
            X_train_initial, 
            u_train_initial,
            X_train_pde,
            bc_lb, 
            bc_ub,
            nu, 
            beta, 
            rho,
            device):
        self.system = args.system
        self.num_collocation_pts = args.num_collocation_pts
        self.model = model
        self.device = device
        eps = [float(eps) for eps in args.eps]

        # PDE parameters
        self.nu = nu
        self.beta = beta
        self.rho = rho

        # data
        self.x_initial = torch.tensor(X_train_initial[:, 0:1], requires_grad=True).float().to(device)
        self.t_initial = torch.tensor(X_train_initial[:, 1:2], requires_grad=True).float().to(device)
        self.x_pde = torch.tensor(X_train_pde[:, 0:1], requires_grad=True).float().to(device)
        self.t_pde = torch.tensor(X_train_pde[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)
        self.u_initial = torch.tensor(u_train_initial, requires_grad=True).float().to(device)

        # invariance constraint
        self.period = ((2 * np.pi) / beta)      # period of the convection system
        domain_x = [0, 2*np.pi]
        domain_t = [0, 1 - self.period]     # t + period must be in [0,1]
        
        # there must be at least one period in the domain
        assert self.period < 1

        # sample for psi_alpha (to evaluate worst-case loss)
        if args.use_mh_sampling:
            # covariance matrix for the Gaussian proposal used in Metropolis-Hastings
            covariance_matrix = torch.tensor([[args.sigma_x**2, args.sigma_xt], [args.sigma_xt, args.sigma_t**2]], device=device, dtype=torch.float32)

            if args.use_mh_projection:
                self.sampler = MH_Projection_Gaussian_2D(self, domain_x, domain_t, args.steps, args.n_samples, self.invariance_loss_per_point, covariance_matrix, device, args.sampler_batch_size, args.causal)
            else:
                self.sampler = MH_Gaussian_2D(self, domain_x, domain_t, args.steps, args.n_samples, self.invariance_loss_per_point, covariance_matrix, device, args.sampler_batch_size, args.causal)
        else:
            self.sampler = LMC_Gaussian(self, args.eta, args.T, domain_x, domain_t, args.steps, args.n_samples, self.invariance_loss_per_point, device, args.sampler_batch_size)


        # # define objective, constraints, and the right hand side of the constraints (i.e. the c in f(x) < c)
        # self.objective_function = self.data_loss
        # if args.fixed_collocation_pts:
        #     self.constraints = [self.fixed_pde_loss]
        # else:
        #     self.constraints = [self.stochastic_pde_loss]
        # self.rhs = [eps[0]]        

        # TODO: remeber to change eps when going from this do data loss as objective
        self.objective_function = self.BC_loss
        self.constraints = [self.IC_loss]
        if args.fixed_collocation_pts:
            self.constraints.append(self.fixed_pde_loss)
        else:
            self.constraints.append(self.stochastic_pde_loss)
        self.rhs = [1e-5, eps[0]]  

        # include invariance constraint
        self.constraints.append(self.worst_case_invariance_loss)
        self.rhs.append(eps[1])
        
        super().__init__()

    def forward(self, x, t):
        """Make a forward pass throught the model. The model takes (x,t) as input and predicts the value of the funcion, u."""
        u = self.model(torch.cat([x, t], dim=1))
        return u

    def pde_residual(self, x, t):
        """ Autograd for calculating the residual for different PDE systems of the model. I.e. if the model satisfies the PDE"""
        u = self.forward(x, t)

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

        if 'convection' in self.system or 'diffusion' in self.system:
            residual = u_t - self.nu*u_xx + self.beta*u_x
        elif 'rd' in self.system:
            residual = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
        elif 'reaction' in self.system:
            residual = u_t - self.rho*u + self.rho*u**2
        
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
    
    def IC_loss(self):
        """Computes loss for the initial condition."""
        u_pred_initial = self.forward(self.x_initial, self.t_initial)
        loss_initial = torch.mean((self.u_initial - u_pred_initial) ** 2)
        return loss_initial
    
    def BC_loss(self):
        """Computes loss for the boundary condition."""
        u_pred_lb = self.forward(self.x_bc_lb, self.t_bc_lb)
        u_pred_ub = self.forward(self.x_bc_ub, self.t_bc_ub)
        loss_boundary = torch.mean((u_pred_lb - u_pred_ub) ** 2)

        if self.nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.boundary_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
            loss_boundary += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
        
        return loss_boundary
    
    def data_loss(self):
        """Computes the data loss. The data loss is computed for points on the 
        boundary and for the initial condition."""
        # prediction for initial condition, lower boundary, and upper boundary
        u_pred_initial = self.forward(self.x_initial, self.t_initial)
        u_pred_lb = self.forward(self.x_bc_lb, self.t_bc_lb)
        u_pred_ub = self.forward(self.x_bc_ub, self.t_bc_ub)

        loss_initial = torch.mean((self.u_initial - u_pred_initial) ** 2)
        loss_boundary = torch.mean((u_pred_lb - u_pred_ub) ** 2)

        if self.nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.boundary_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
            loss_boundary += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)

        loss = loss_initial + loss_boundary
        return loss
    
    def fixed_pde_loss(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE
        fo fixed collocation points."""
        residual_pde = self.pde_residual(self.x_pde, self.t_pde)
        loss_pde = torch.mean(residual_pde ** 2)
        return loss_pde
    
    def stochastic_pde_loss(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE
        for stochastic collocation points."""

        # sample collocation points for spatial dimension (i.e. x coordinate) in the range (0, 2pi)
        x_pde = torch.rand(self.num_collocation_pts) * (2 * np.pi)

        # sample collocation points for time dimension in the range (0, 1]
        t_pde = torch.rand(self.num_collocation_pts)

        x_pde = x_pde.reshape(-1, 1).float().to(self.device)
        t_pde = t_pde.reshape(-1, 1).float().to(self.device)

        x_pde.requires_grad_(True)
        t_pde.requires_grad_(True)

        residual_pde = self.pde_residual(x_pde, t_pde)
        loss_pde = torch.mean(residual_pde ** 2)
        return loss_pde
    
    def invariance_loss_per_point(self, x, t):
        """Computes the invariance loss per point, i.e. seperatly for each (x,t).
        This is important and required when sampling points (using LMC and MH) for the worst_case loss using batches.
        """
        u = self.forward(x, t)
        u_period = self.forward(x, t + self.period)

        invariance_loss = (u - u_period) ** 2
        return invariance_loss
    
    def invariance_loss(self, x, t):
        """Computes the invariance loss (i.e. consiscency of points separated by a period in time)."""
        #  t + period must be in the domain
        assert (t <= (1 - self.period)).all()

        u = self.forward(x, t)
        u_period = self.forward(x, t + self.period)

        invariance_loss = torch.mean((u - u_period) ** 2)
        return invariance_loss

    def worst_case_invariance_loss(self):
        coords_adv = self.sampler()

        x_adv = coords_adv[:, 0:1]
        t_adv = coords_adv[:, 1:2]
        
        worst_case_loss = self.invariance_loss(x_adv, t_adv)
        return worst_case_loss
    
    def predict(self, X, dummy=None):
        """Make predictions. Used during evaluation.
        dummy=None is required to be able to use the same solver 
        for solving specific BVPs and parametric solutions."""
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        self.model.eval()
        u = self.forward(x, t)
        u = u.detach().cpu().numpy()

        return u


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert args.system == 'convection'

    beta = args.beta
    # nu and rho are zero since we are considering the convection equation (but needed for data generation).
    # they are used for reaction-diffusion
    nu = 0.0
    rho = 0.0

    # layers of NN
    layers = [int(item) for item in args.layers.split(',')]

    # get data by solving the PDE
    X_train_initial, u_train_initial, X_train_pde, bc_lb, bc_ub, X_test, u_exact, x, t, exact_solution, u_exact_1d, _ = generate_data(args, nu, beta, rho)
    
    # add first layer with correct number of input nodes
    layers.insert(0, X_train_initial.shape[-1])

    # set set for reproducibility
    set_seed(args.seed)

    # define constrained learning problem
    model = MLP(layers, 'tanh').to(device)
    constrained_pinn = SCL_M_S(
        args, 
        model, 
        X_train_initial, 
        u_train_initial,
        X_train_pde, 
        bc_lb, 
        bc_ub, 
        nu,
        beta, 
        rho,
        device)

    optimizers = {'primal_optimizer': 'Adam',
                  'use_primal_lr_scheduler': args.use_primal_lr_scheduler,
                'dual_optimizer': 'Adam',
                'use_dual_lr_scheduler': args.use_dual_lr_scheduler}
    
    data_dict = {beta: {'X_test': X_test, 'u_exact': u_exact}}      # data used for eval/test
    solver = SimultaneousPrimalDual(constrained_pinn, optimizers, args.lr_primal, args.lr_dual, args.epochs, args.eval_every, data_dict)

    # solve constrained learning problem
    solver.solve(constrained_pinn)

    # test model and print metrics
    u_pred = constrained_pinn.predict(X_test)

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
    suffix = f'scl_m_s_convection'
    dir_name = date_time_string  + '_' + suffix

    path_save = f'saved/{args.system}/{dir_name}'

    if args.visualize:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        u_pred = u_pred.reshape(len(t), len(x))
        plot_exact_u(exact_solution, x, t, path=path_save, label_x='x', label_y='t', flip_axis_plotting=True)
        plot_u_diff(exact_solution, u_pred, x, t, path=path_save, label_x='x', label_y='t', flip_axis_plotting=True)
        plot_u_pred(u_exact_1d, u_pred, x, t, path=path_save, label_x='x', label_y='t', flip_axis_plotting=True)

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

    u_pred = u_pred.reshape(-1,1)
    error_u_relative = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    error_u_abs = np.mean(np.abs(u_exact - u_pred))
    error_u_linf = np.linalg.norm(u_exact - u_pred, np.inf) / np.linalg.norm(u_exact, np.inf)

    print('Test metrics:')
    print('Relative error: %e' % (error_u_relative))
    print('Absolute error: %e' % (error_u_abs))
    print('linf error: %e' % (error_u_linf))
    print('')

    # save arguments
    with open(f'{path_save}/args.txt', 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
        
        # save error metrics
        file.write(f'Relative error: {error_u_relative}\n')
        file.write(f'Absolute error: {error_u_abs}\n')
        file.write(f'linf error: {error_u_linf}\n')

    # save state dict
    with open(f'{path_save}/state_dict.pkl', 'wb') as f:
        pickle.dump(solver.state_dict, f)
    

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')

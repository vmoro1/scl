# SCL(M) for the Eikonal equaiton.
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
import torch.nn.functional as F

from scl.unsupervised.csl_problem import ConstrainedStatisticalLearningProblem
from scl.unsupervised.solver import SimultaneousPrimalDual
from scl.unsupervised.model import MLP
from scl.unsupervised.langevin_monte_carlo import *
from scl.unsupervised.metropolis_hastings import *

from scl.unsupervised.utils import *
from scl.unsupervised.visualize_solution import *
from scl.unsupervised.generate_data.load_data_eikonal import load_eikonal_data

parser = argparse.ArgumentParser(description='SCL(M)')

parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--num_collocation_pts', type=int, default=1000, help='Number of collocation points.')
parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train for.')
parser.add_argument('--lr_primal', type=float, default=1e-3, help='Learning rate for primal variables (NN parameters).')
parser.add_argument('--lr_dual', type=float, default=1e-5, help='Learning rate for dual variables (lambdas).')
# parser.add_argument('--eps', nargs='+', default=[5e-1], help='Tolerances for the constraints.')
parser.add_argument('--eps', nargs='+', default=[5e-1, 1e-5], help='Tolerances for the constraints.')
parser.add_argument('--use_primal_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the primal variables.')
parser.add_argument('--use_dual_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the dual variables.')
parser.add_argument('--layers', type=str, default='128,128,128,128,1', help='Dimensions of layers of the NN (except the first layer).')

parser.add_argument('--use_mh_sampling', action=argparse.BooleanOptionalAction, default=True, help='Whether to use Metropolis-Hastings sampling (or Langevin Monte Carlo) to to sample from psi_alpha in order to compute worst-case losses.')
parser.add_argument('--sampler_batch_size', type=int, default=20, help='Batch size for speeding up MH/LMC sampling.')
parser.add_argument('--eta', type=float, default=1e-3, help='Step size for LMC sampling.')
parser.add_argument('--T', type=float, default=1.0, help='Temperature for LMC sampling.')
parser.add_argument('--steps', type=int, default=250, help='Number of steps for sampling.')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to use.')
parser.add_argument('--sigma_x', type=float, default=0.2, help='Standard deviation for the Gaussian proposal in the MH sampling.')
parser.add_argument('--sigma_y', type=float, default=0.2, help='Standard deviation for the Gaussian proposal in the MH sampling.')
parser.add_argument('--sigma_xy', type=float, default=0.0, help='Standard deviation for the Gaussian proposal in the MH sampling.')

parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True, help='Visualize the solution and prediction of the model.')
parser.add_argument('--plot_diagnostics', action=argparse.BooleanOptionalAction, default=True, help='Plot diagnostics of the model.')
parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate the model every n epochs.')
parser.add_argument('--run_location', type=str, default='local', help='Where the script is executed', choices=['local', 'horeka', 'bro_cluster'])


class SCL_M(ConstrainedStatisticalLearningProblem):
    def __init__(
            self,
            args, 
            model,
            X_bc,
            X_ic,
            device):

        self.num_collocation_pts = args.num_collocation_pts
        self.model = model
        self.device = device
        eps = [float(eps) for eps in args.eps]

        # data
        self.x_ic = torch.tensor(X_ic[:, 0:1]).float().to(device)
        self.y_ic = torch.tensor(X_ic[:, 1:2]).float().to(device)
        self.x_bc = torch.tensor(X_bc[:, 0:1]).float().to(device)
        self.y_bc = torch.tensor(X_bc[:, 1:2]).float().to(device)

        domain_x = [-1, 1]
        domain_y = [-1, 1]

        # sample for psi_alpha (to evaluate worst-case loss)
        if args.use_mh_sampling:
            # covariance matrix for the Gaussian proposal used in Metropolis-Hastings
            covariance_matrix = torch.tensor([[args.sigma_x**2, args.sigma_xy], [args.sigma_xy, args.sigma_y**2]], device=device, dtype=torch.float32)

            self.sampler = MH_Gaussian_2D(self, domain_x, domain_y, args.steps, args.n_samples, self.pde_loss_per_point, covariance_matrix, device, args.sampler_batch_size)
        else:
            self.sampler = LMC_Gaussian(self, args.eta, args.T, domain_x, domain_y, args.steps, args.n_samples, self.pde_loss_per_point, device, args.sampler_batch_size)

        # define objective, constraints, and the right hand side of the constraints (i.e. the c in f(x) < c)
        # self.objective_function = self.data_loss
        # self.constraints = [self.worst_case_pde_loss]
        # self.rhs = eps  
        
        # with strucutral constraint
        self.objective_function = self.ic_loss
        self.constraints = [self.worst_case_pde_loss, self.bc_loss]
        self.rhs = eps

        super().__init__()

    def forward(self, x, y):
        """Make a forward pass through the model. The model takes (x,t) as input and predicts the value of the funcion, u."""
        u = self.model(torch.cat([x, y], dim=1))
        return u
    
    def pde_residual(self, x, y):
        """ Autograd for calculating the residual for different PDE systems of the model. I.e. if the model satisfies the PDE"""
        u = self.forward(x, y)

        u_x = torch.autograd.grad(
            u, 
            x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, 
            y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        residual = u_x ** 2 + u_y ** 2 - 1
        return residual
    
    def data_loss(self):
        """Computes the data loss. The data loss is computed for points on the 
        boundary."""
        loss = self.bc_loss() + self.ic_loss()
        return loss
    
    def ic_loss(self):
        u_pred = self.forward(self.x_ic, self.y_ic)
        loss_ic = (u_pred ** 2).mean()  # Signed Distance Field (SDF) = 0 on zero contour set
        return loss_ic

    def bc_loss(self):
        """Can also be viewed as structural constraint for the Eikonal equation to enforde the SDF to be non-negative on boundary."""
        u_pred = self.forward(self.x_bc, self.y_bc)
        loss_bc = (F.relu(-u_pred)).mean()  # Signed Distance Field (SDF) should be positive on boundary
        return loss_bc
    
    def pde_loss_per_point(self, x, y):
        """Computes the PDE loss per point, i.e. for each (x,y) seperatly. 
        This is important and required when sampling in batches using LMC and MH."""
        x.requires_grad_(True)
        y.requires_grad_(True)

        residual_pde = self.pde_residual(x, y)
        loss_pde = residual_pde ** 2
        return loss_pde
    
    def worst_case_pde_loss(self):
        coords_adv = self.sampler()
        x_adv = coords_adv[:, 0:1]
        y_adv = coords_adv[:, 1:2]

        x_adv.requires_grad_(True)
        y_adv.requires_grad_(True)
        
        worst_case_loss = self.pde_loss_per_point(x_adv, y_adv)
        worst_case_loss = torch.mean(worst_case_loss)
        return worst_case_loss
    
    def predict(self, X, dummy=None):
        """Make predictions. Used during evaluation.
        dummy=None is required to be able to use the same solver 
        for solving specific BVPs and parametric solutions."""
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        self.model.eval()
        u = self.forward(x, y)
        u = u.detach().cpu().numpy()

        return u


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get data
    X_train_pde, X_bc, X_ic, X_test, u_exact, x, y, u_exact_1d = load_eikonal_data(args)

    # layers of NN
    layers = [int(item) for item in args.layers.split(',')]

    # add first layer with correct number of input nodes
    layers.insert(0, X_test.shape[-1])

    # set set for reproducibility
    set_seed(args.seed)

    # for saving
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    suffix = f'scl_m_eikonal'
    dir_name = date_time_string  + '_' + suffix
    path_save = f'saved/eikonal/{dir_name}'

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # define constrained learning problem
    model = MLP(layers, 'tanh').to(device)
    constrained_pinn = SCL_M(
        args, 
        model,
        X_bc,
        X_ic,
        device)

    optimizers = {'primal_optimizer': 'Adam',
                  'use_primal_lr_scheduler': args.use_primal_lr_scheduler,
                'dual_optimizer': 'Adam',
                'use_dual_lr_scheduler': args.use_dual_lr_scheduler}
    
    data_dict = {'Eikonal': {'X_test': X_test, 'u_exact': u_exact_1d}}      # data used for eval/test
    solver = SimultaneousPrimalDual(constrained_pinn, 
                                    optimizers, args.lr_primal, 
                                    args.lr_dual, 
                                    args.epochs, 
                                    args.eval_every, 
                                    data_dict,
                                    solving_specific_BVP=True,
                                    path_save=path_save)

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
    if args.visualize:
        u_pred = u_pred.reshape(len(y), len(x))
        plot_exact_u(u_exact, x, y, path_save, label_x='x', label_y='y', flip_axis_plotting=False, cmap='coolwarm')
        plot_u_diff(u_exact, u_pred, x, y, path_save, label_x='x', label_y='y', flip_axis_plotting=False)
        plot_u_pred(u_exact, u_pred, x, y, path_save, label_x='x', label_y='y', flip_axis_plotting=False, cmap='coolwarm')

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
    error_u_relative = np.linalg.norm(u_exact_1d - u_pred, 2) / np.linalg.norm(u_exact, 2)
    error_u_abs = np.mean(np.abs(u_exact_1d - u_pred))
    error_u_linf = np.linalg.norm(u_exact_1d - u_pred, np.inf) / np.linalg.norm(u_exact, np.inf)

    print('Test metrics at final epoch:')
    print('Relative error: %e' % (error_u_relative))
    print('Absolute error: %e' % (error_u_abs))
    print('L_inf error: %e' % (error_u_linf))
    print('')

    if solver.best_epoch is not None:
        idx_best = int(solver.best_epoch / args.eval_every)
        pde_param = next(iter(data_dict))   # there's only one pde_param
        print('Test metrics for epoch with arly stopping:')
        print('Relative error: %e' % (solver.state_dict['Relative Error'][f'Relative Error_pde_param_{pde_param}'][idx_best]))
        print('Absolute error: %e' % (solver.state_dict['Absolute Error'][f'Absolute Error_pde_param_{pde_param}'][idx_best]))
        print('L_inf error: %e' % (solver.state_dict['linf Error'][f'linf Error_pde_param_{pde_param}'][idx_best]))
        print('')
    else:
        print('No early stopping')

    # save arguments
    with open(f'{path_save}/args.txt', 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
        
        # save error metrics
        file.write(f'Relative error: {error_u_relative}\n')
        file.write(f'Absolute error: {error_u_abs}\n')
        file.write(f'linf error: {error_u_linf}\n')

        if solver.best_epoch is not None:
            # save metrics for early stopping
            file.write(f'Best relative error: {solver.best_rel_l2_error}\n')
            file.write(f'Best epoch: {solver.best_epoch}\n')
            file.write(f'Relative error at early stopping: {solver.state_dict["Relative Error"][f"Relative Error_pde_param_{pde_param}"][idx_best]}\n')
            file.write(f'Absolute error at early stopping: {solver.state_dict["Absolute Error"][f"Absolute Error_pde_param_{pde_param}"][idx_best]}\n')
            file.write(f'linf error at early stopping: {solver.state_dict["linf Error"][f"linf Error_pde_param_{pde_param}"][idx_best]}\n')
        else:
            file.write('No early stopping\n')

    # save state dict
    with open(f'{path_save}/state_dict.pkl', 'wb') as f:
        pickle.dump(solver.state_dict, f)

    # best model
    if solver.best_epoch is not None:
        print('Best model at epoch:', solver.best_epoch)
        print('Best relative L2 error:', solver.best_rel_l2_error)
    

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')

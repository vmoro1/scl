# Baseline parametric solution (PI / Cho et al., 2024) for convection
# Loss terms (data + PDE) are aggregated for different betas in the loss function.
# Want to find NN f(x,t,beta) that approximates solution everywhere for all betas in teh domain

import sys
import os
import argparse
from datetime import datetime
import pickle

sys.path.append('.') 

import numpy as np
import torch

from scl.unsupervised.model import MLP

from scl.unsupervised.generate_data.generate_data_convection_rd import *
from scl.unsupervised.utils import *
from scl.unsupervised.visualize_solution import *

parser = argparse.ArgumentParser(description='Baseline parametric solution')

parser.add_argument('--system', type=str, default='convection', help='PDE system. Should be fixed but needed for data generation.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--num_collocation_pts', type=int, default=1000, help='Number of collocation points.')
parser.add_argument('--fixed_collocation_pts', action=argparse.BooleanOptionalAction, default=False, help='Whether to use fixed collocation points or sample new ones at each epoch.')
parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train for.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--use_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler.')
parser.add_argument('--data_loss_weight', type=float, default=1.0, help='Weight for the data loss.')

parser.add_argument('--num_x_pts', type=int, default=256, help='Number of points in the x grid.')
parser.add_argument('--num_t_pts', type=int, default=100, help='Number of points in the time grid.')
parser.add_argument('--betas_train', nargs='+', default=[1.0, 2.5, 5.0], help='betas for training.')
parser.add_argument('--betas_test', nargs='+', default=[1, 2.5, 5], help='betas for testing.')

parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition if no forcing term.')
parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")
parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus the first layer.')

parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True, help='Visualize the solution and prediction of the model.')
parser.add_argument('--plot_diagnostics', action=argparse.BooleanOptionalAction, default=True, help='Plot diagnostics of the model.')
parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval_every', type=int, default=100, help='Evaluate the model every n epochs.') 


class BaselineParametricSolution:
    def __init__(
            self,
            args, 
            model, 
            X_train_initial, 
            u_train_initial,
            X_train_pde,
            bc_lb, 
            bc_ub, 
            device):
        
        self.num_collocation_pts = args.num_collocation_pts
        self.fixed_collocation_pts = args.fixed_collocation_pts
        self.model = model
        self.device = device
        self.betas_train = torch.tensor([float(beta) for beta in args.betas_train]).reshape(-1, 1).float().to(device)
        self.n_betas = len(self.betas_train)

        # data
        self.x_initial = torch.tensor(X_train_initial[:, 0:1]).unsqueeze(0).repeat(self.n_betas, 1, 1).float().to(device)
        self.t_initial = torch.tensor(X_train_initial[:, 1:2]).unsqueeze(0).repeat(self.n_betas, 1, 1).float().to(device)
        self.x_pde = torch.tensor(X_train_pde[:, 0:1]).unsqueeze(0).repeat(self.n_betas, 1, 1).requires_grad_(True).float().to(device)
        self.t_pde = torch.tensor(X_train_pde[:, 1:2]).unsqueeze(0).repeat(self.n_betas, 1, 1).requires_grad_(True).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1]).unsqueeze(0).repeat(self.n_betas, 1, 1).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2]).unsqueeze(0).repeat(self.n_betas, 1, 1).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1]).unsqueeze(0).repeat(self.n_betas, 1, 1).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2]).unsqueeze(0).repeat(self.n_betas, 1, 1).float().to(device)
        self.u_initial = torch.tensor(u_train_initial).unsqueeze(0).repeat(self.n_betas, 1, 1).float().to(device)
    
    def forward(self, x, t, beta):
        """Make a forward pass throught the model. The model takes (x,t) as input and predicts the value of the funcion, u."""
        N = x.shape[0]                                                # number of points
        beta = beta.reshape(1, -1).repeat(N, 1)             
        u = self.model(torch.cat([x, t, beta], dim=1))
        return u
    
    def forward_batch(self, x, t):
        """Make a forward pass throught the model. The model takes (x,t) as input and predicts the value of the funcion, u."""
        N = x.shape[1]      # number of points
        betas = self.betas_train.unsqueeze(1).repeat(1, N, 1)
        u = self.model(torch.cat([x, t, betas], dim=-1))
        return u

    def pde_residual(self, x, t):
        """ Autograd for calculating the PDE residual, i.e., if the model satisfies the PDE"""
        u = self.forward_batch(x, t)

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

        beta = self.betas_train.reshape(-1,1,1)
        residual = u_t + beta * u_x
        return residual
    
    def data_loss(self):
        """Computes the data loss. The data loss is computed for points on the 
        boundary and for the initial condition."""

        # prediction for initial condition, lower boundary, and upper boundary
        u_pred_initial = self.forward_batch(self.x_initial, self.t_initial)
        u_pred_lb = self.forward_batch(self.x_bc_lb, self.t_bc_lb)
        u_pred_ub = self.forward_batch(self.x_bc_ub, self.t_bc_ub)

        # average over all points and sum over all pde parameters
        loss_initial = torch.mean((self.u_initial - u_pred_initial) ** 2, dim=-2)
        loss_initial = sum(loss_initial)

        loss_boundary = torch.mean((u_pred_lb - u_pred_ub) ** 2, dim=-2)
        loss_boundary = sum(loss_boundary)

        loss = loss_initial + loss_boundary
        return loss
    
    def pde_loss_stochastic(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE."""

        # sample collocation points for spatial dimension (i.e. x coordinate) in the range (0, 2pi)
        x_pde = torch.rand(self.num_collocation_pts) * (2 * np.pi)

        # sample collocation points for time dimension in the range (0, 1]
        t_pde = torch.rand(self.num_collocation_pts)

        x_pde = x_pde.reshape(1, -1, 1).repeat(self.n_betas, 1, 1).float().to(self.device)
        t_pde = t_pde.reshape(1, -1, 1).repeat(self.n_betas, 1, 1).float().to(self.device)

        x_pde.requires_grad_(True)
        t_pde.requires_grad_(True)

        residual_pde = self.pde_residual(x_pde, t_pde)

        # average over all collocation points and sum over 
        loss_pde = torch.mean(residual_pde ** 2, dim=-2)
        loss_pde = sum(loss_pde)
        return loss_pde
    
    def pde_loss_fixed(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE
        for fixed collocation points."""
        residual_pde = self.pde_residual(self.x_pde, self.t_pde)
        
        # average over all collocation points and sum over 
        loss_pde = torch.mean(residual_pde ** 2, dim=-2)
        loss_pde = sum(loss_pde)
        return loss_pde
        
    def pde_loss(self):
        if self.fixed_collocation_pts:
            return self.pde_loss_fixed()
        else:
            return self.pde_loss_stochastic()
    
    def predict(self, X, beta):
        """Make predictions. Used during evaluation."""
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        beta = torch.tensor(beta).float().to(self.device)

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
    layers.insert(0, X_train_initial.shape[-1] + 1)     # +1 for convection parameter beta

    # set set for reproducibility
    set_seed(args.seed)  

    model = MLP(layers, 'tanh').to(device)

    # X_train_initial, u_train_initial, bc_lb, bc_ub are independent of beta so we don't need them for different betas
    pinn = BaselineParametricSolution(
        args,
        model,
        X_train_initial,
        u_train_initial,
        X_train_pde,
        bc_lb,
        bc_ub,
        device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # lr schedulers
    if args.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    rel_error = {f'Relative Error PDE Parameter: {pde_param}': [] for pde_param in betas_test}
    abs_error = {f'Absolute Error PDE Parameter: {pde_param}': [] for pde_param in betas_test}
    linf_error = {f'L_inf Error PDE Parameter: {pde_param}': [] for pde_param in betas_test}

    model.train()
    for e in range(args.epochs):
        optimizer.zero_grad()
        loss = args.data_loss_weight * pinn.data_loss() + pinn.pde_loss()
        loss.backward()
        optimizer.step()

        if args.use_lr_scheduler:
            lr_scheduler.step()

        # eval
        if e % args.eval_every == 0:
            for beta in betas_test:
                X_test = data_test[beta]['X_test']
                u_exact = data_test[beta]['u_exact']

                u_pred = pinn.predict(X_test, beta)
                error_u_relative = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
                error_u_abs = np.mean(np.abs(u_exact - u_pred))
                error_u_linf = np.linalg.norm(u_exact - u_pred, np.inf) / np.linalg.norm(u_exact, np.inf)

                rel_error[f'Relative Error PDE Parameter: {beta}'].append(error_u_relative)
                abs_error[f'Absolute Error PDE Parameter: {beta}'].append(error_u_abs)
                linf_error[f'L_inf Error PDE Parameter: {beta}'].append(error_u_linf)

    # make predictions
    preds = {}
    for beta in betas_test:
        u_pred = pinn.predict(X_test, beta)
        preds[beta] = u_pred

    # visualize PINN solution and plot diagnostics
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    suffix = f'baseline_parametric_solution_convection'
    dir_name = date_time_string  + '_' + suffix

    path_save = f'saved/convection/{dir_name}'

    if args.save_model:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        torch.save(pinn.model.state_dict(), f"{path_save}/model.pt")
        torch.save(optimizer.state_dict(), f'{path_save}/optimizer.pt')

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
        error_u_relative = np.linalg.norm(u_exact_beta - u_pred_beta, 2) / np.linalg.norm(u_exact_beta, 2)
        error_u_abs = np.mean(np.abs(u_exact_beta - u_pred_beta))
        error_u_linf = np.linalg.norm(u_exact_beta - u_pred_beta, np.inf) / np.linalg.norm(u_exact_beta, np.inf)

        print(f'Beta: {beta}')
        print('Test metrics:')
        print('Relative error: %e' % (error_u_relative))
        print('Absolute error: %e' % (error_u_abs))
        print('linf error: %e' % (error_u_linf))
        print('')

        with open(f'{path_save}/args.txt', 'a') as file:
            file.write(f'Beta: {beta}\n')
            file.write('Test metrics:\n')
            file.write(f'Relative error: {error_u_relative}\n')
            file.write(f'Absolute error: {error_u_abs}\n')
            file.write(f'linf error: {error_u_linf}\n')

    # plot errors
    plot_errors_parametric_solution(rel_error, abs_error, linf_error, args.epochs, args.eval_every, betas_test, path_save)

    # saver errors
    with open(f'{path_save}/rel_error.pkl', 'wb') as f:
        pickle.dump(rel_error, f)
    with open(f'{path_save}/abs_error.pkl', 'wb') as f:
        pickle.dump(abs_error, f)
    with open(f'{path_save}/linf_error.pkl', 'wb') as f:
        pickle.dump(linf_error, f)


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')

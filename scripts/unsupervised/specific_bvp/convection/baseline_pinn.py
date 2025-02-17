# Baseline PINN for convection equation

import sys
import os
import argparse
from datetime import datetime

sys.path.append('.') 
sys.path.append('/slurm-storage/vigmor/scl')

import numpy as np
import torch

from scl.unsupervised.model import MLP

from scl.unsupervised.generate_data.generate_data_convection_rd import *
from scl.unsupervised.utils import set_seed
from scl.unsupervised.visualize_solution import *

parser = argparse.ArgumentParser(description='PINN baseline')

parser.add_argument('--system', type=str, default='convection', help='PDE system. Should be fixed but needed for data generation.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--num_collocation_pts', type=int, default=1000, help='Number of collocation points.')
parser.add_argument('--fixed_collocation_pts', action=argparse.BooleanOptionalAction, default=False, help='Whether to use fixed collocation points or sample new ones at each epoch.')
parser.add_argument('--epochs', type=int, default=200000, help='Number of epochs to train for.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--use_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler.')
parser.add_argument('--data_loss_weight', type=float, default=100.0, help='Weight for the data loss term.')

parser.add_argument('--num_x_pts', type=int, default=256, help='Number of points in the x grid.')
parser.add_argument('--num_t_pts', type=int, default=100, help='Number of points in the time grid.')
parser.add_argument('--beta', type=float, default=30.0, help='Convection parameter beta')

parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition (sin(x) or similiar for convection else gauss)')
parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions of layers of the NN (except the first layer).')
parser.add_argument('--source', default=0, type=float, help="Source term. Not used but needed for data generation.")

parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True, help='Visualize the solution and prediction of the model.')
parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate the model every n epochs.')


class PINN:
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

        self.system = args.system    # pde system
        self.model = model
        self.nu = nu
        self.beta = beta
        self.rho = rho
        self.fixed_collocation_pts = args.fixed_collocation_pts
        self.num_collocation_pts = args.num_collocation_pts
        self.device = device

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
    
    def pde_loss_fixed(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE
        for fixed collocation points."""
        residual_pde = self.pde_residual(self.x_pde, self.t_pde)
        loss_pde = torch.mean(residual_pde ** 2)
        return loss_pde
    
    def pde_loss_stochastic(self):
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
    
    def pde_loss(self):
        if self.fixed_collocation_pts:
            return self.pde_loss_fixed()
        else:
            return self.pde_loss_stochastic()
    
    def predict(self, X):
        """Make predictions. Used during evaluation."""
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
    X_train_initial, u_train_initial, X_train_pde, bc_lb, bc_ub, X_test, u_exact, x, t, exact_solution, u_exact_1d, G = generate_data(args, nu, beta, rho)
    
    # add first layer with correct number of input nodes
    layers.insert(0, X_train_initial.shape[-1])

    # set set for reproducibility
    set_seed(args.seed)  
    
    model = MLP(layers, 'tanh').to(device)
    pinn = PINN(
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    rel_error = []
    abs_error = []
    linf_error = []
    loss_list = []

    # for early stopping
    best_loss = np.inf
    best_model = None

    # for saving
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    suffix = f'baseline_pinn_convection'
    dir_name = date_time_string  + '_' + suffix

    path_save = f'saved/convection/{dir_name}'

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    model.train()
    for e in range(args.epochs):
        optimizer.zero_grad()
        loss = args.data_loss_weight * pinn.data_loss() + pinn.pde_loss()   
        loss.backward()
        optimizer.step()

        if args.use_lr_scheduler:
            lr_scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = pinn.model
            epoch = e
            torch.save(best_model.state_dict(), f'{path_save}/best_model.pt')

            u_pred = pinn.predict(X_test)
            best_rel_l2_error = np.linalg.norm(u_exact-u_pred, 2)/np.linalg.norm(u_exact, 2)

        loss_list.append(loss.item())

        # eval
        if e % args.eval_every == 0:
            u_pred = pinn.predict(X_test)
            error_u_relative = np.linalg.norm(u_exact-u_pred, 2)/np.linalg.norm(u_exact, 2)
            error_u_abs = np.mean(np.abs(u_exact - u_pred))
            error_u_linf = np.linalg.norm(u_exact - u_pred, np.inf)/np.linalg.norm(u_exact, np.inf)

            rel_error.append(error_u_relative)
            abs_error.append(error_u_abs)
            linf_error.append(error_u_linf)

    # test model and print metrics
    u_pred = pinn.predict(X_test)

    error_u_relative = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    error_u_abs = np.mean(np.abs(u_exact - u_pred))
    error_u_linf = np.linalg.norm(u_exact - u_pred, np.inf)  /np.linalg.norm(u_exact, np.inf)

    print('Test metrics at final epoch:')
    print('Relative error: %e' % (error_u_relative))
    print('Absolute error: %e' % (error_u_abs))
    print('L_inf error: %e' % (error_u_linf))
    print('')

    idx_best = int(epoch / args.eval_every)
    print('Test metrics for epoch with the lowest loss:')
    print('Relative error: %e' % (rel_error[idx_best]))
    print('Absolute error: %e' % (abs_error[idx_best]))
    print('L_inf error: %e' % (linf_error[idx_best]))
    print('')

    if args.save_model:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        torch.save(pinn.model.state_dict(), f"{path_save}/model.pt")
        torch.save(optimizer.state_dict(), f'{path_save}/optimizer.pt')

    # visualize PINN solution
    if args.visualize:
        u_pred = u_pred.reshape(len(t), len(x))
        plot_exact_u(exact_solution, x, t, path=path_save, label_x='x', label_y='t', flip_axis_plotting=True)
        plot_u_diff(exact_solution, u_pred, x, t, path=path_save, label_x='x', label_y='t', flip_axis_plotting=True)
        plot_u_pred(u_exact_1d, u_pred, x, t, path=path_save, label_x='x', label_y='t', flip_axis_plotting=True)

    # save arguments
    with open(f'{path_save}/args.txt', 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
        
        # save error metrics
        file.write(f'Relative error: {error_u_relative}\n')
        file.write(f'Absolute error: {error_u_abs}\n')
        file.write(f'L_inf error: {error_u_linf}\n')

        file.write(f'Best relative error: {best_rel_l2_error}\n')
        file.write(f'Best epoch: {epoch}\n')
        file.write(f'Best relative error: {rel_error[idx_best]}\n')
        file.write(f'Best absolute error: {abs_error[idx_best]}\n')
        file.write(f'Best L_inf error: {linf_error[idx_best]}\n')

    # plot errors
    plot_errors(rel_error, abs_error, linf_error, args.epochs, args.eval_every, path_save)

    # save errors
    np.save(f'{path_save}/relative_error.npy', rel_error)
    np.save(f'{path_save}/absolute_error.npy', abs_error)
    np.save(f'{path_save}/linf_error.npy', linf_error)

    # save loss
    np.save(f'{path_save}/loss.npy', loss_list)

    # save u_pred
    np.save(f'{path_save}/u_pred.npy', u_pred.T)

    # best model
    print('Best model at epoch:', epoch)
    print('Best relative L2 error:', best_rel_l2_error)
    

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')

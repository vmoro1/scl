# Constrained FNO for Burgers Equation with constrains per sample (e.g. PDE constraint is per sample and not an average over all samples)

# TODO: multiply per sample constraints by factor 1/n_samples? Otherwise, more samples will mean less emphasis on the objective

import sys
import os
import argparse
from datetime import datetime
import pickle

import torch.utils

sys.path.append('.')
sys.path.append('/home/gridsan/vmoro/CSL/csl_neuraloperator')  
sys.path.append('/home/hk-project-test-p0021798/st_ac144859/csl_neuraloperator')

import torch
import wandb

from csl_neuraloperator.per_sample_constraints.csl_problem import ConstrainedStatisticalLearningProblem
from csl_neuraloperator.per_sample_constraints.solver import SimultaneousPrimalDual
from csl_neuraloperator.neuraloperator.neuralop.models import FNO
from csl_neuraloperator.pde_losses import BurgersPDE_Loss
from csl_neuraloperator.per_sample_constraints.dataset import DatasetPerSampleConstraints
from csl_neuraloperator.utils import *

parser = argparse.ArgumentParser(description='FNO for Burgers Equation')

parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
parser.add_argument('--lr_primal', type=float, default=1e-3, help='Learning rate for primal variables (NN parameters).')
parser.add_argument('--lr_dual', type=float, default=1e-4, help='Learning rate for dual variables (lambdas).')
parser.add_argument('--eps', nargs='+', default=[1e-3], help='Tolerances for the constraints.')
parser.add_argument('--per_sample_eps', nargs='+', default=[1e-3], help='Tolerances for the per sample constraints. All samples for a per sample constraint share the same tolerance.')
parser.add_argument('--use_primal_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the primal variables.')
parser.add_argument('--use_dual_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the dual variables.')
parser.add_argument('--pde_loss_method', type=str, default='finite_difference', help='Method used to compute the PDE loss/physics-informed loss (if the pde loss is used).')
parser.add_argument('--constrained_problem_formulation', choices=['per_sample_data', 'interior_data_with_per_sample_pde', 'boundary_data_with_per_sample_pde'], default='per_sample_data', help='Formulation of the constrained learning problem.')

parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples.')
parser.add_argument('--n_test', type=int, default=200, help='Number of test samples.')

parser.add_argument('--n_modes', type=int, default=16, help='Number of Fourier modes to use.')
parser.add_argument('--hidden_channels', type=int, default=64, help='Width of the FNO(i.e. number of channels)')
parser.add_argument('--projection_channels', type=int, default=128, help='Number of hidden channels of the projection back to the output')
parser.add_argument('--n_layers', type=int, default=4, help='Number of Fourier layers to use.')

parser.add_argument('--save_model', default=True)
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate the model every n epochs.')
parser.add_argument('--visualize', default=False, help='Visualize the solution and prediction of the model.')
parser.add_argument('--plot_diagnostics', default=True, help='Plot diagnostics of the model.')
parser.add_argument('--wandb_project_name', type=str, default='csl_neuraloperator')   
parser.add_argument('--wandb_run_name', type=str, default='constrained_fno_burgers_eq_per_sample')
parser.add_argument('--run_location', choices=['locally', 'supercloud', 'horeka'], default='locally', help='Choose where the script is executed.')


class ConstrainedFNO_BurgersEq_PerSampleConstraints(ConstrainedStatisticalLearningProblem):
    """Constrained FNO for Burgers Equation. The optimization problem is 
    formulated as a statistical constrained learning probelm with average 
    and/or per sample constraints."""
    
    def __init__(self, model, viscosity, n_train_samples, args):
        self.model = model
        self.device = next(model.parameters()).device
        self.relative_L2_loss_fn = LpLoss()
        self.per_sample_relative_L2_loss_fn = LpLoss(size_average=False, reduction=False)
        self.pde_loss_fn = BurgersPDE_Loss(visc=viscosity, method=args.pde_loss_method)
        self.per_sample_pde_loss_fn = BurgersPDE_Loss(visc=viscosity, method=args.pde_loss_method, reduce=False)

        assert args.pde_loss_method == 'finite_difference'   # only finite difference method is supported for per sample loss

        eps = [float(eps) for eps in args.eps]

        # define objective, average constraints, per sample constraints and the the 
        # right hand side (tolerance) for all constraints (i.e. the c in f_i(x) < c)
        if args.constrained_problem_formulation == 'interior_data_with_per_sample_pde':
            # data loss from all of the domain is the objective and the PDE loss for each sample are (seperate) constraints (i.e. per sample PDE loss constraints)
            self.objective_function = self.data_loss
            self.per_sample_constraints = [self.per_sample_pde_loss]
            self.per_sample_rhs = [torch.full((n_train_samples,), float(per_sample_tol), device=self.device) for per_sample_tol in args.per_sample_eps]
        elif args.constrained_problem_formulation == 'boundary_data_with_per_sample_pde':
            # data loss from the boundary (IC and BC) is the objective and the PDE loss for each sample are (seperate) constraints (i.e. per sample PDE loss constraints)
            self.objective_function = self.ic_and_bc_loss
            self.per_sample_constraints = [self.per_sample_pde_loss]
            self.per_sample_rhs = [torch.full((n_train_samples,), float(per_sample_tol), device=self.device) for per_sample_tol in args.per_sample_eps]
        elif args.constrained_problem_formulation == 'per_sample_data':
            self.objective_function = lambda: torch.tensor(1.0, dtype=float).to(self.device)
            self.per_sample_constraints = [self.per_sample_data_loss]
            self.per_sample_rhs = [torch.full((n_train_samples,), float(per_sample_tol), device=self.device) for per_sample_tol in args.per_sample_eps]
        else:
            raise ValueError('Invalid constrained problem formulation. Choose from: interior_data_with_per_sample_pde, boundary_data_with_per_sample_pde, per_sample_data.')

        super().__init__()

    def forward(self, x):
        self.y_pred = self.model(x).squeeze(1)
        self.x = x
        self.prediction_stored = True
        self.input_stored = True

    def store_targets_and_samples_idx(self, sample_idx, y):
        self.y = y
        self.target_stored = True
        self.samples_indices = sample_idx
        self.samples_indices_stored = True

    def reset(self):
        self.prediction_computed = False
        self.y_pred = None
        self.input_stored = False
        self.x = None
        self.target_stored = False
        self.y = None
        self.samples_indices_stored = False
        self.samples_indices = None


    def data_loss(self):
        """Computes the data loss (relative L2 loss)"""
        data_loss = self.relative_L2_loss_fn(self.y_pred, self.y)
        return data_loss
    
    def ic_loss(self):
        """Computes the loss for the initial condition"""
        y_ic_gt = self.y[:,0,:]         # initial condition for ground truth
        y_ic_pred = self.y_pred[:,0,:]
        ic_loss = self.relative_L2_loss_fn(y_ic_pred, y_ic_gt)
        return ic_loss
    
    def bc_loss(self):
        """Computes the loss for the boundary condition (loss for points on the spatial boundary)"""
        y_bc_pred = get_bc_1d(self.y_pred)
        y_bc_gt = get_bc_1d(self.y)
        bc_loss = self.relative_L2_loss_fn(y_bc_pred, y_bc_gt)
        return bc_loss
    
    def ic_and_bc_loss(self):
        """Computes the loss for the initial and boundary conditions"""
        ic_loss = self.ic_loss()
        bc_loss = self.bc_loss()
        return ic_loss + bc_loss
    
    def pde_loss(self):
        """Computes the loss for the PDE, i.e. the physics-informed loss."""
        pde_loss = self.pde_loss_fn(self.y_pred)
        return pde_loss
    
    def per_sample_pde_loss(self):
        """Computes the loss for the PDE, i.e. the physics-informed loss."""
        per_samples_pde_loss = self.per_sample_pde_loss_fn(self.y_pred)
        return per_samples_pde_loss
    
    def per_sample_data_loss(self):
        """Computes the per sample loss for the data, i.e. the relative L2 loss."""
        per_sample_data_loss = self.per_sample_relative_L2_loss_fn(self.y_pred, self.y)
        return per_sample_data_loss


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    if args.run_location == 'locally':
        path_base = ""
    elif args.run_location == 'supercloud':
        path_base = "/home/gridsan/vmoro/CSL/csl_neuraloperator/"
    elif args.run_location == 'horeka':
        path_base = "/home/hk-project-test-p0021798/st_ac144859/csl_neuraloperator/"
    else:
        raise ValueError('Invalid run location')

    date_time = datetime.now()
    date_time_string = date_time.strftime("%Y-%m-%d %H:%M:%S")
    name_date = f"{args.wandb_run_name}_{date_time_string}"

    # wandb logger
    path_wandb = path_base + "saved/wandb_logs"
    if not os.path.exists(path_wandb):
        os.makedirs(path_wandb)
    config = vars(args)
    os.environ["WANDB_API_KEY"] = '2ba73edbdcfc28a2f25e4fb15eb9f9d6b5af890f'
    os.environ["WANDB_MODE"] = 'offline'
    os.environ["WANDB_DIR"] = path_wandb
    
    wandb.init(
        project=args.wandb_project_name, 
        name=f"{args.wandb_run_name}_{name_date}",
        entity="viggomoro",
        config=config
        )

    # load 2D Burgers' equation data - learn solution operator from initial condition to solution at all times
    # TODO: remember to change this
    # n_train = 1000
    # n_test = 200

    # n_train = 4
    # n_test = 2
    # args.batch_size = 2
    # args.epochs = 2

    nx = 128    # spatial grid size
    nt = 101    # temporal grid size

    data_path = path_base + 'data/burgers_eq/burgers_pino.mat'
    data_reader = MatReader(data_path)
    viscosity_coeff = data_reader.read_field('visc').item()    # viscosity coefficient of PDE
    x_data = data_reader.read_field('input')                   # shape: (number of samples, nx). Each sample is u(0, x) values for discretized grid
    y_data = data_reader.read_field('output')                  # shape: (number of samples, nt, nx). Each sample is u(t, x) values for discretized grid

    # add temporal dim to input by repeating the input data. This is 
    # done since input and output grids need to be of the same size
    n_samples = x_data.shape[0]
    x_data = x_data.reshape(n_samples, 1, nx).repeat([1, nt, 1])

    # add positional encoding (i.e. x and t coordinates) to the input data
    domain_x = [0, 1]
    domain_t = [0, 1]
    grid_x = torch.tensor(np.linspace(domain_x[0], domain_x[1], nx + 1)[:-1], dtype=torch.float)
    grid_t = torch.tensor(np.linspace(domain_t[0], domain_t[1], nt), dtype=torch.float)
    grid_x = grid_x.reshape(1, 1, 1, nx)
    grid_t = grid_t.reshape(1, 1, nt, 1)

    x_data = x_data.unsqueeze(1)    # add channel dim, shape: (number of samples, 1, nt, nx)
    x_data = torch.cat([x_data, 
                        grid_x.repeat([n_samples, 1, nt, 1]), 
                        grid_t.repeat([n_samples, 1, 1, nx])
                        ], dim=1)

    x_train = x_data[:args.n_train]        
    y_train = y_data[:args.n_train]     
    x_test = x_data[-args.n_test:]
    y_test = y_data[-args.n_test:]

    train_loader = torch.utils.data.DataLoader(DatasetPerSampleConstraints(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(DatasetPerSampleConstraints(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = FNO(
        n_modes=(args.n_modes, args.n_modes), 
        hidden_channels=args.hidden_channels, 
        in_channels=3, 
        out_channels=1, 
        lifting_channels=False,
        projection_channels=args.projection_channels, 
        n_layers=args.n_layers)
    
    model = model.to(device)
    
    constrained_fno = ConstrainedFNO_BurgersEq_PerSampleConstraints(model, viscosity_coeff, args.n_train, args)

    optimizers = {'primal_optimizer': 'Adam',
                  'use_primal_lr_scheduler': args.use_primal_lr_scheduler,
                'dual_optimizer': 'Adam',
                'use_dual_lr_scheduler': args.use_dual_lr_scheduler}
    
    solver = SimultaneousPrimalDual(
        csl_problem=constrained_fno,
        optimizers=optimizers,
        primal_lr=args.lr_primal,
        dual_lr=args.lr_dual,
        epochs=args.epochs,
        eval_every=args.eval_every,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
    )
    
    solver.solve(constrained_fno)

    # relative L2 error on test
    final_test_error = eval_model_per_sample_constraints(model, test_loader, device)
    test_errors = solver.state_dict['Relative_L2_test_error']

    print()
    print('-------Training complete-------')
    print('Final relative L2 error on test set: ', final_test_error)
    print('Best relative L2 error on test set: ', min(test_errors))
    print('Epoch with lowest relative L2 error on test set: ', test_errors.index(min(test_errors)) * args.eval_every)
    print()

    # print final diagnostics
    print('Final diagnostics:')
    num_dual_variables = len(constrained_fno.lambdas)
    for i in range(num_dual_variables):
        print(f'Dual variable {i}: ', solver.state_dict['dual_variables'][f'dual_variable_{i}'][-1])
        print(f'Slack_{i}: ', solver.state_dict['slacks'][f'slack_{i}'][-1])
    print('Approximate duality gap: ', solver.state_dict['approximate_duality_gap'][-1])
    print('Approximate relative duality gap: ', solver.state_dict['aproximate_relative_duality_gap'][-1])
    print('Lagrangian: ', solver.state_dict['Lagrangian'][-1])
    print('Primal value: ', solver.state_dict['primal_value'][-1])
    print()

    # save 
    if args.constrained_problem_formulation == 'interior_data_with_per_sample_pde':
        name = 'per_sample_pde_constrained_fno_interior_data'
    elif args.constrained_problem_formulation == 'boundary_data_with_per_sample_pde': 
        name = 'per_sample_pde_constrained_fno_boundary_data'
    elif args.constrained_problem_formulation == 'per_sample_data':
        name = 'per_sample_data_constrained_fno'
    else:
        name = 'fno'

    random_number = torch.randint(1000, (1,)).item()
    path_save = f'saved/burgers_eq/{name}/{date_time_string}_{random_number}'

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    if args.plot_diagnostics:
        solver.plot(path_save)
    
    # TODO: visualize solution

    if args.save_model:
        torch.save(model.state_dict(), path_save + '/model.pt')

    # save arguments
    with open(f'{path_save}/args.txt', 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
        
        # save error metrics
        file.write(f'Best relative L2 error: {min(test_errors)}\n')
        file.write(f'Final relative L2 error: {final_test_error}\n')

    # save state dict
    with open(f'{path_save}/state_dict.pkl', 'wb') as f:
        pickle.dump(solver.state_dict, f)

    # close logger
    wandb.finish()

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')

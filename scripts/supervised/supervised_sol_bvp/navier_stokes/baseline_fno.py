# Baseline FNO (PII in paper) for Navier Stokes equaation. 
# We consider three different viscosities: 1e-3, 1e-4 and 1e-5.

import sys
import os
import argparse
from datetime import datetime

sys.path.append('.')

import torch
import wandb

from scl.supervised.utils import *
from scl.supervised.neuraloperator.neuralop.models import FNO

parser = argparse.ArgumentParser(description='Baseline FNO')

parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
parser.add_argument('--use_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler.')
parser.add_argument('--viscosity', type=float, choices=[1e-3, 1e-4, 1e-5], default=1e-3, help='Viscosity of the fluid.')
parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples.')
parser.add_argument('--n_test', type=int, default=200, help='Number of test samples.')

parser.add_argument('--n_modes', type=int, default=8, help='Number of Fourier modes to use.')
parser.add_argument('--hidden_channels', type=int, default=64, help='Width of the FNO(i.e. number of channels)')  
parser.add_argument('--projection_channels', type=int, default=128, help='Number of hidden channels of the projection back to the output')
parser.add_argument('--n_layers', type=int, default=8, help='Number of Fourier layers to use.')

parser.add_argument('--save_model', default=True)
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate the model every n epochs.')
parser.add_argument('--visualize', default=False, help='Visualize the solution and prediction of the model.')
parser.add_argument('--wandb_project_name', type=str, default='scl_supervised')   
parser.add_argument('--wandb_run_name', type=str, default='baseline_fno_navier_stokes')
parser.add_argument('--run_location', choices=['locally', 'supercloud', 'horeka'], default='locally', help='Choose where the script is executed.')


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    if args.run_location == 'locally':
        path_base = ""
    elif args.run_location == 'supercloud':
        path_base = "/home/gridsan/vmoro/scl/"
    elif args.run_location == 'horeka':
        path_base = "/home/hk-project-test-p0021798/st_ac144859/scl/"
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
    
    # # TODO: remember to change this
    args.n_train = 4
    args.n_test = 2
    args.batch_size = 2
    args.epochs = 2

    # load data
    if args.viscosity == 1e-3:
        n_spatial = 64      # spatial grid size
        n_temporal = 50     # temporal grid size
        
        data_path = path_base + 'data/supervised/navier_stokes/ns_V1e-3_N5000_T50.mat'
        data_reader = MatReader(data_path)
        data = data_reader.read_field('u')
        data = data.permute(0, 3, 1, 2)          # new shape: (number of samples, n_temporal, n_spatial, n_spatial)

        # data_path = path_base + 'data/navier_stokes/ns_solution_subset_v1e-3_T50.pt'
        # data = torch.load(data_path)           # shape: (number of samples, n_temporal, n_spatial, n_spatial). Each sample is w(t, x, y) values for discretized grid
        # data = data.permute(0, 3, 1, 2)        # new shape: (number of samples, n_temporal, n_spatial, n_spatial)
    elif args.viscosity == 1e-4:
        n_spatial = 64      # spatial grid size
        n_temporal = 30     # temporal grid size    # NOTE: cane be increased to 50 (or reduced further)

        data_path = path_base + 'data/supervised/navier_stokes/ns_V1e-4_N10000_T30.mat'
        data_reader = MatReader(data_path)
        data = data_reader.read_field('u')
        data = data.permute(0, 3, 1, 2)
        data = data[:, :n_temporal, :, :]       # new shape: (number of samples, n_temporal, n_spatial, n_spatial)
    elif args.viscosity == 1e-5:
        n_spatial = 64      # spatial grid size
        n_temporal = 20     # temporal grid size

        data_path = path_base + 'data/supervised/navier_stokes/NavierStokes_V1e-5_N1200_T20.mat'
        data_reader = MatReader(data_path)
        data = data_reader.read_field('u')
        data = data.permute(0, 3, 1, 2)         # new shape: (number of samples, n_temporal, n_spatial, n_spatial)
    else:
        raise ValueError('Invalid viscosity')

    # use the first time step to predict all other time steps
    x_data = data[:, 0:1, :, :]               # shape: (number of samples, 1, n_spatial, n_spatial). Each sample is w(0, x, y) values for discretized grid
    y_data = data[:, :, :, :]                 # shape: (number of samples, n_temporal, n_spatial, n_spatial). Each sample is w(t, x, y) values for discretized grid

    x_data = x_data.repeat([1, n_temporal, 1, 1])     # shape: (number of samples, n_temporal, n_spatial, n_spatial)

    # add positional encoding (i.e. x, y and t coordinates) to the input data
    grid_x_1d = torch.tensor(np.linspace(0, 1, n_spatial + 1)[:-1], dtype=torch.float)
    grid_y_1d = torch.tensor(np.linspace(0, 1, n_spatial + 1)[:-1], dtype=torch.float)
    gridt_1d = torch.tensor(np.linspace(1, n_temporal, n_temporal), dtype=torch.float)

    n_samples = x_data.shape[0]
    grid_x = grid_x_1d.reshape(1, 1, 1, n_spatial, 1).repeat([n_samples, 1, n_temporal, 1, n_spatial])
    grid_y = grid_y_1d.reshape(1, 1, 1, 1, n_spatial).repeat([n_samples, 1, n_temporal, n_spatial, 1])
    gridt = gridt_1d.reshape(1, 1, n_temporal, 1, 1).repeat([n_samples, 1, 1, n_spatial, n_spatial])

    x_data = x_data.unsqueeze(1)                                    # add channel dim. shape: (number of samples, 1, n_temporal, n_spatial, n_spatial)
    x_data = torch.cat([x_data, grid_x, grid_y, gridt], dim=1)      # shape: (number of samples, 4, n_temporal, n_spatial, n_spatial)

    x_train = x_data[:args.n_train]        
    y_train = y_data[:args.n_train]
    x_test = x_data[-args.n_test:]
    y_test = y_data[-args.n_test:]        

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = FNO(
        n_modes=(args.n_modes, args.n_modes, args.n_modes), 
        hidden_channels=args.hidden_channels, 
        in_channels=4, 
        out_channels=1, 
        lifting_channels=False,
        projection_channels=args.projection_channels, 
        n_layers=args.n_layers)

    
    model = model.to(device)
    
    relative_L2_loss_fn = LpLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # train
    test_errors = []
    for e in range(args.epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # forward pass
            y_pred = model(x).squeeze(1)

            # loss
            data_loss = relative_L2_loss_fn(y_pred, y)

            # backward pass
            optimizer.zero_grad()
            data_loss.backward()
            optimizer.step()

            # log
            mini_batch_idx = (e * len(train_loader) + i)
            to_log = {'Loss/training loss': data_loss.item(), 'Loss/Mini batch index': mini_batch_idx, 'Loss/Epoch': e}
            wandb.log(to_log)

        if args.use_lr_scheduler:
            scheduler.step()

        # eval
        if e % args.eval_every == 0:
            test_error = eval_model(model, test_loader, device)
            test_errors.append(test_error)

            print(f'Epoch {e}, Relative L2 test error: {test_error}')

            # log
            to_log = {'Relative L2 test error/Relative L2 test error': test_error, 'Relative L2 test error/Epoch': e}
            wandb.log(to_log)

    # relative L2 error on test sef after training is done
    final_test_error = eval_model(model, test_loader, device)

    print()
    print('-------Training complete-------')
    print('Final relative L2 error on test set: ', final_test_error)
    print('Best relative L2 error on test set: ', min(test_errors))
    print('Epoch with lowest relative L2 error on test set: ', test_errors.index(min(test_errors)) * args.eval_every)
    print()

    # save
    name = 'baseline_fno'
    path_save = f'saved/navier_stokes/{name}/{date_time_string}/' 

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    if args.save_model:
        torch.save(model.state_dict(), path_save + 'model.pt')

    # save arguments
    with open(f'{path_save}/args.txt', 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
        
        # save error metrics
        file.write(f'Best relative L2 error: {min(test_errors)}\n')

    # close logger
    wandb.finish()

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')

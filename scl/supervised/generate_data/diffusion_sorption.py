# Generate dataset for diffusion sorption PDE from h5 file from PDEBench
# The h5 file from PDEBench can be found at https://darus.uni-stuttgart.de/dataverse/sciml_benchmark

import torch
import numpy as np
import h5py


def main(file_path, path_base):
    solution_list = []
    t_grid_list = []
    x_grid_list = []

    with h5py.File(file_path, 'r') as h5file:
        # iterate through each group in the HDF5 file (each group corresponds to a different initial condition)
        for group_name in h5file.keys():
            group = h5file[group_name]
            
            solution = group['data'][:]
            t_grid = group['grid/t'][:]
            x_grid = group['grid/x'][:]
            
            solution_list.append(solution)
            t_grid_list.append(t_grid)
            x_grid_list.append(x_grid)

        solution = np.stack(solution_list, axis=0)
        t_grid = np.stack(t_grid_list, axis=0)
        x_grid = np.stack(x_grid_list, axis=0)

        # t_grid and x_grid should be same for all groups
        assert np.all(t_grid == t_grid[0])
        assert np.all(x_grid == x_grid[0])

        assert solution.shape == (10000, 101, 1024, 1)
        assert t_grid.shape == (10000, 101)
        assert x_grid.shape == (10000, 1024)

        x_grid = torch.tensor(x_grid[0], dtype=torch.float32)
        t_grid = torch.tensor(t_grid[0], dtype=torch.float32)

        solution = solution.squeeze(-1)
        solution = torch.tensor(solution, dtype=torch.float32)

        torch.save(solution, f'{path_base}/data/diffusion_sorption/solution.pt')
        torch.save(t_grid, f'{path_base}/data/diffusion_sorption/t_grid.pt')
        torch.save(x_grid, f'{path_base}/data/diffusion_sorption/x_grid.pt')


if __name__ == '__main__':
    path_base = '/Users/viggomoro/ScienceConstrainedLearning/NeuralOperators/csl_neuraloperator'
    file_path = path_base + '/data/PDEBench/1D_diff-sorp_NA_NA.h5'

    main(file_path, path_base)

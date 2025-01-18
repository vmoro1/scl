import random

import numpy as np
import scipy.io
import torch
import h5py


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, mode='r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class LpLoss:
    """loss function with rel Lp loss"""
    def __init__(self, p=2, size_average=True, reduction=True):

        # dimension and Lp-norm type are postive
        assert p > 0

        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        batch_size = x.shape[0]

        diff_norms = torch.linalg.vector_norm(x.reshape(batch_size, -1) - y.reshape(batch_size, -1), self.p, 1)
        y_norms = torch.linalg.vector_norm(y.reshape(batch_size,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def eval_model(model, data_loader, device):
    """Evaluate model. Metric is relative L2 error."""
    loss_fn = LpLoss()
    y_pred_store = []
    y_target_store = []
    model.eval()
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).squeeze(1)

            # store predictions and targets
            y_pred_store.append(y_pred.cpu())
            y_target_store.append(y.cpu())

    y_pred_store = torch.cat(y_pred_store, dim=0)
    y_target_store = torch.cat(y_target_store, dim=0)

    test_error = loss_fn(y_pred_store, y_target_store).item()
    return test_error


def eval_model_per_sample_constraints(model, data_loader, device):
    """Evaluate constrained model with per sample constraints. The dataloader for 
    per sample constrained problems also returns the sample index which is the 
    only difference between this and the other eval function. Metric is relative 
    L2 error."""
    loss_fn = LpLoss()
    y_pred_store = []
    y_target_store = []
    model.eval()

    with torch.no_grad():
        for _, x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).squeeze(1)

            # store predictions and targets
            y_pred_store.append(y_pred.cpu())
            y_target_store.append(y.cpu())

    y_pred_store = torch.cat(y_pred_store, dim=0)
    y_target_store = torch.cat(y_target_store, dim=0)

    test_error = loss_fn(y_pred_store, y_target_store).item()
    return test_error


# NOTE: The below functions for getting the points on the spatial boundary assumes the temporal domain (if it exists)
# comes before the spatial domain, i.e. that the shape of the tensor is (n_samples, n_temporal, n_spatial).
def get_bc_1d(y):
    """Get points on the spatial boundary of the domain for 1 spatial dimension."""
    bc_lower = y[..., 0]
    bc_upper = y[..., -1]
    bc_pounts = torch.cat([bc_lower, bc_upper], dim=-1)
    return bc_pounts

def get_bc_2d(y):
    """Get points on the spatial boundary of the domain for 2 spatial dimensions."""
    bc_left = y[..., :, 0]
    bc_right = y[..., :, -1]
    bc_top = y[..., 0, :]
    bc_bottom = y[..., -1, :]

    bc_points = torch.cat([bc_left, bc_right, bc_top, bc_bottom], dim=-1)
    return bc_points



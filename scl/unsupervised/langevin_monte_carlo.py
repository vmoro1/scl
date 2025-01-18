# Sample from probability distributions (psi_alpha) using Langevin Monte Carlo (LMC) in order to evaluate worst-case losses.
# Here, it's done for the 2D case when psi_alpha depende on two variables (e.g., x and t)

import math

import torch
from torch.distributions.laplace import Laplace


class WorstCaseLoss:
    """Base class to compute worst-case losses."""

    def __init__(self, problem, eta, T, domain_x, domain_t, steps, n_samples, loss_fn, device, batch_size=1):
        self.problem = problem          # constrained learning problem
        self.eta = eta                  # step size for LMC
        self.T = T                      # temperature for LMC
        self.domain_x = domain_x    
        self.domain_t = domain_t    
        self.steps = steps         
        self.n_samples = n_samples
        self.loss_fn = loss_fn          # function/distribution (without normalizing constant) to sample from   
        self.device = device
        self.batch_size = batch_size    # batch size for LMC (used to speed up sampling by sampling by constructing multiple markov chains)

    def project_onto_domain(self, coord):
        x = torch.clamp(coord[:,0:1], self.domain_x[0], self.domain_x[1])
        t = torch.clamp(coord[:,1:2], self.domain_t[0], self.domain_t[1])
        return torch.cat((x, t), dim=1)

    def forward(self):
        raise NotImplementedError
    
    def __call__(self):
        return self.forward()


class LMC_Gaussian(WorstCaseLoss):
    """Langevin Monte Carlo sampling with Gaussian noise for computing worst-case losses."""
    def __init__(self, problem, eta, T, domain_x, domain_t, steps, n_samples, loss_fn, device, batch_size=1):
        super(LMC_Gaussian, self).__init__(problem, eta, T, domain_x, domain_t, steps, n_samples, loss_fn, device, batch_size)

    def forward(self):
        self.problem.model.eval()
        
        # initial coordinates
        x_adv = torch.rand((self.batch_size,1), dtype=torch.float32, device=self.device) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        t_adv = torch.rand((self.batch_size,1), dtype=torch.float32, device=self.device) * (self.domain_t[1] - self.domain_t[0]) + self.domain_t[0]

        coord_adv = torch.cat((x_adv, t_adv), dim=1)    # adversarial coordinate
        coords_adv_store = []

        for _ in range(self.steps):
            coord_adv.requires_grad_(True)

            adv_loss = self.loss_fn(coord_adv[:,0:1], coord_adv[:,1:2])
            adv_loss = torch.clamp(adv_loss, min=1e-9)      # avoid log(0)  TODO: better way to do this?
            U = torch.log(adv_loss)

            grad = torch.autograd.grad(U, [coord_adv], grad_outputs=torch.ones_like(U))[0].detach()   # NOTE: it's important that the loss is't averaged over the batch as that gives the wrong gradients
            noise = torch.randn_like(coord_adv).to(self.device).detach()

            coord_adv = coord_adv + self.eta * grad + math.sqrt(2 * self.eta * self.T) * noise
            coord_adv = self.project_onto_domain(coord_adv)

            coords_adv_store.append(coord_adv.clone().detach())

        coords_adv_store = torch.cat(coords_adv_store, dim=0)
        coords_adv_store = coords_adv_store[-self.n_samples:]       # return only the last n_samples (since markov chain needs to mix before it samples from the target distribution)

        self.problem.model.train()
        return coords_adv_store


class LMC_Laplacian(WorstCaseLoss):
    """Langevin Monte Carlo sampling with Laplassian noise for computing worst-case losses."""
    def __init__(self, problem, eta, T, domain_x, domain_t, steps, n_samples, loss_fn, device, batch_size=1):
        super(LMC_Laplacian, self).__init__(problem, eta, T, domain_x, domain_t, steps, n_samples, loss_fn, device, batch_size)

    def forward(self):
        self.problem.model.eval()
        noise_dist = Laplace(torch.tensor(0.), torch.tensor(1.))
        
        x_adv = torch.rand((self.batch_size,1), dtype=torch.float32, device=self.device) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        t_adv = torch.rand((self.batch_size,1), dtype=torch.float32, device=self.device) * (self.domain_t[1] - self.domain_t[0]) + self.domain_t[0]

        coord_adv = torch.cat((x_adv, t_adv), dim=1)
        coords_adv_store = []

        for _ in range(self.steps):
            coord_adv.requires_grad_(True)

            adv_loss = self.loss_fn(coord_adv[:,0:1], coord_adv[:,1:2])
            adv_loss = torch.clamp(adv_loss, min=1e-9)      # avoid log(0)
            U = torch.log(adv_loss)

            grad = torch.autograd.grad(U, [coord_adv], grad_outputs=torch.ones_like(U))[0].detach()
            noise = noise_dist.sample(grad.shape)

            coord_adv = coord_adv + self.eta * torch.sign(grad + math.sqrt(2 * self.eta * self.T) * noise)
            coord_adv = self.project_onto_domain(coord_adv)

            coords_adv_store.append(coord_adv.clone().detach())

        coords_adv_store = torch.cat(coords_adv_store, dim=0)
        coords_adv_store = coords_adv_store[-self.n_samples:]

        self.problem.model.train()
        return coords_adv_store
    
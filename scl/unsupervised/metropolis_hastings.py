# Sample from probability distributions (psi_alpha) using Metropolis Hastings (MH) in order to 
# evaluate worst-case losses. There are different classes for this depending on the dimensionality 
# of the input to the distribution being sampled from (psi_alpha). For example, if psi_alpha - psi_alpha(x, t), 
# then it's the 2D case since we want to sample 2D samples. This is done for 1D, 2D, 3D and 4D.

# NOTE: the naming of variables might not always correspond with what is being sampled. For example, (x,t) might 
# as well be (nu,rho). Oftentime, the naming follow the convection parameters/use-case


import torch
from torch.distributions import MultivariateNormal


# ======================= 1D case =======================


class WorstCaseLoss_1D:
    """Base class to compute worst-case losses for 1D case."""

    def __init__(self, problem, domain, steps, n_samples, loss_fn, sigma, device, batch_size=1):
        self.problem = problem          # csl problem
        self.domain = domain            # domain of beta  
        self.steps = steps
        self.n_samples = n_samples
        self.loss_fn = loss_fn          # function/distribution (without normalizing constant) to sample from
        self.sigma = sigma              # standard deviation for the Gaussian proposal in the MH sampling
        self.device = device
        self.batch_size = batch_size    # batch size for MH (used to speed up sampling by sampling by constructing multiple markov chains)

    def in_domain(self, beta):
        """Check if proposed beta is within the domain"""
        within_domain = (beta >= self.domain[0]) & (beta <= self.domain[1])
        return within_domain
    
    def project_onto_domain(self, beta):
        """Project beta onto the domain."""
        projected_beta = torch.clamp(beta, self.domain[0], self.domain[1])
        return projected_beta

    def forward(self):
        raise NotImplementedError
    
    def __call__(self):
        return self.forward()


class MH_Gaussian_1D(WorstCaseLoss_1D):
    """Metropolis-Hastings sampling with Gaussian proposal distribution for 1D case.
    Proposals outside of the domain are rejected."""
    def __init__(self, problem, domain, steps, n_samples, loss_fn, sigma, device, batch_size=1):
        super(MH_Gaussian_1D, self).__init__(problem, domain, steps, n_samples, loss_fn, sigma, device, batch_size)

    def forward(self):
        self.problem.model.eval()
        
        current_beta = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain[1] - self.domain[0]) + self.domain[0]
        current_loss = self.loss_fn(current_beta).detach()
        current_loss = torch.clamp(current_loss, min=1e-9)

        betas_store = []

        for _ in range(self.steps):
            # sample from gaussian proposal distribution (one sample per markov chain)
            proposed_beta = torch.normal(current_beta, self.sigma)
            # proposed_beta = self.project_onto_domain(proposed_beta)
            in_domain = self.in_domain(proposed_beta)

            # evaluate the target distribution at proposed betainates
            proposed_loss = self.loss_fn(proposed_beta).detach()
            proposed_loss = torch.clamp(proposed_loss, min=1e-9)    # done to avoid division by zero TODO: better way to handle this?

            # acceptance probability for each proposal
            acceptance_probability = torch.min(torch.ones_like(current_loss, device=self.device), proposed_loss / current_loss)

            # decide to accept or reject the proposal (for each sample in the batch)
            accept = (torch.rand((self.batch_size, 1), device=self.device) < acceptance_probability) & in_domain

            # update where proposals are accepted
            current_beta[accept] = proposed_beta[accept]
            current_loss[accept] = proposed_loss[accept]

            betas_store.append(current_beta.clone())

        betas_store = torch.cat(betas_store, dim=0)
        betas_store = betas_store[-self.n_samples:]       # return only the last n_samples (since markov chain needs some warmup to get sample from the target distribution)

        self.problem.model.train()
        return betas_store


# ======================= 2D case =======================


class WorstCaseLoss_2D:
    """Base class to compute worst-case losses for 2D case."""

    def __init__(self, problem, domain_x, domain_t, steps, n_samples, loss_fn, covariance_matrix, device, batch_size=1, causal=False):
        self.problem = problem          # csl problem
        self.domain_x = domain_x    
        self.domain_t = domain_t   
        self.steps = steps
        self.n_samples = n_samples
        self.loss_fn = loss_fn          # function/distribution (without normalizing constant) to sample from
        self.device = device
        self.batch_size = batch_size    # batch size for MH (used to speed up sampling by sampling by constructing multiple markov chains)
        self.causal = causal            # whether to start from t-0 or from a random t (starting from t=0 is similair to causal training of PINNs)

        # covariance matrix for the Gaussian proposal used in Metropolis-Hastings
        self.covariance_matrix = covariance_matrix

    def project_onto_domain(self, coord):
        x = torch.clamp(coord[:,0:1], self.domain_x[0], self.domain_x[1])
        t = torch.clamp(coord[:,1:2], self.domain_t[0], self.domain_t[1])
        return torch.cat((x, t), dim=1)

    def in_domain(self, coord):
        """Check if proposed coordinates are within the domain"""
        within_domain = (coord[:, 0:1] >= self.domain_x[0]) & (coord[:, 0:1] <= self.domain_x[1]) & \
                        (coord[:, 1:2] >= self.domain_t[0]) & (coord[:, 1:2] <= self.domain_t[1])
        
        return within_domain

    def forward(self):
        raise NotImplementedError
    
    def __call__(self):
        return self.forward()


class MH_Gaussian_2D(WorstCaseLoss_2D):
    """Metropolis-Hastings sampling with Gaussian proposal distribution for 2D case.
    Proposals outside of the domain are rejected."""
    def __init__(self, problem, domain_x, domain_t, steps, n_samples, loss_fn, covariance_matrix, device, batch_size=1, causal=False):
        super(MH_Gaussian_2D, self).__init__(problem, domain_x, domain_t, steps, n_samples, loss_fn, covariance_matrix, device, batch_size, causal)

    def forward(self):
        self.problem.model.eval()
        
        x_current = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        if self.causal:
            t_current = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        else:
            t_current = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_t[1] - self.domain_t[0]) + self.domain_t[0]

        current_coord = torch.cat((x_current, t_current), dim=1)
        current_loss = self.loss_fn(current_coord[:, 0:1], current_coord[:, 1:2]).detach()
        current_loss = torch.clamp(current_loss, min=1e-9)

        coords_store = []

        for _ in range(self.steps):
            # gaussian proposal
            proposal_distribution = MultivariateNormal(current_coord, self.covariance_matrix.expand(self.batch_size, 2, 2))      

            # sample from the proposal distribution (one sample per markov chain)
            proposed_coord = proposal_distribution.sample()
            in_domain = self.in_domain(proposed_coord)

            # evaluate the target distribution at proposed coordinates
            proposed_loss = self.loss_fn(proposed_coord[:, 0:1], proposed_coord[:, 1:2]).detach()
            proposed_loss = torch.clamp(proposed_loss, min=1e-9)    # done to avoid division by zero TODO: better way to handle this?

            # acceptance probability for each proposal
            acceptance_probability = torch.min(torch.ones_like(current_loss, device=self.device), proposed_loss / current_loss)

            # decide to accept or reject the proposal (for each sample in the batch)
            accept = (torch.rand((self.batch_size, 1), device=self.device) < acceptance_probability) & in_domain
            accept = accept.squeeze()

            # update where proposals are accepted
            current_coord[accept] = proposed_coord[accept]
            current_loss[accept] = proposed_loss[accept]

            coords_store.append(current_coord.clone())

        coords_store = torch.cat(coords_store, dim=0)
        coords_store = coords_store[-self.n_samples:]       # return only the last n_samples (since markov chain needs some warmup to get sample from the target distribution)

        self.problem.model.train()
        return coords_store

class MH_Projection_Gaussian_2D(WorstCaseLoss_2D):
    """Metropolis-Hastings sampling with Gaussian proposal distributio for 2D case.
    Proposals outside of the domain are projected onto the domain."""
    def __init__(self, problem, domain_x, domain_t, steps, n_samples, loss_fn, covariance_matrix, device, batch_size=1, causal=False):
        super(MH_Projection_Gaussian_2D, self).__init__(problem, domain_x, domain_t, steps, n_samples, loss_fn, covariance_matrix, device, batch_size, causal)

    def forward(self):
        self.problem.model.eval()
        
        x_current = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        if self.causal:
            t_current = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        else:
            t_current = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_t[1] - self.domain_t[0]) + self.domain_t[0]

        current_coord = torch.cat((x_current, t_current), dim=1)
        current_loss = self.loss_fn(current_coord[:, 0:1], current_coord[:, 1:2]).detach()
        current_loss = torch.clamp(current_loss, min=1e-9)

        coords_store = []

        for _ in range(self.steps):
            # gaussian proposal
            proposal_distribution = MultivariateNormal(current_coord, self.covariance_matrix.expand(self.batch_size, 2, 2))      

            # sample from the proposal distribution (one sample per markov chain)
            proposed_coord = proposal_distribution.sample()
            proposed_coord = self.project_onto_domain(proposed_coord)

            # evaluate the target distribution at proposed coordinates
            proposed_loss = self.loss_fn(proposed_coord[:, 0:1], proposed_coord[:, 1:2]).detach()
            proposed_loss = torch.clamp(proposed_loss, min=1e-9)    # done to avoid division by zero TODO: better way to handle this?

            # acceptance probability for each proposal
            acceptance_probability = torch.min(torch.ones_like(current_loss, device=self.device), proposed_loss / current_loss)

            # decide to accept or reject the proposal (for each sample in the batch)
            accept = (torch.rand((self.batch_size, 1), device=self.device) < acceptance_probability)
            accept = accept.squeeze()

            # update where proposals are accepted
            current_coord[accept] = proposed_coord[accept]
            current_loss[accept] = proposed_loss[accept]

            coords_store.append(current_coord.clone())

        coords_store = torch.cat(coords_store, dim=0)
        coords_store = coords_store[-self.n_samples:]       # return only the last n_samples (since markov chain needs some warmup to get sample from the target distribution)

        self.problem.model.train()
        return coords_store
    

# ======================= 3D case =======================


class WorstCaseLoss_3D:
    """Base class to compute worst-case losses for 3D case, e.g., sampling (x,t,beta)"""

    def __init__(self, problem, domain_x, domain_t, domain_beta, steps, n_samples, loss_fn, covariance_matrix, device, batch_size=1):
        self.problem = problem          # csl problem
        self.domain_x = domain_x        
        self.domain_t = domain_t
        self.domain_beta = domain_beta
        self.steps = steps
        self.n_samples = n_samples
        self.loss_fn = loss_fn          # function/distribution (without normalizing constant) to sample from
        self.covariance_matrix = covariance_matrix
        self.device = device
        self.batch_size = batch_size    # batch size for MH (used to speed up sampling by sampling by constructing multiple markov chains)

    def in_domain(self, sample):
        """Check if proposed sample is within the domain"""
        in_domain = (self.domain_x[0] <= sample[:,0:1]) & (sample[:,0:1] <= self.domain_x[1]) & \
                    (self.domain_t[0] <= sample[:,1:2]) & (sample[:,1:2] <= self.domain_t[1]) & \
                    (self.domain_beta[0] <= sample[:,2:3]) & (sample[:,2:3] <= self.domain_beta[1])
        return in_domain
    
    def project_onto_domain(self, sample):
        """Project sample onto the domain."""
        x = torch.clamp(sample[:,0:1], self.domain_x[0], self.domain_x[1])
        t = torch.clamp(sample[:,1:2], self.domain_t[0], self.domain_t[1])
        beta = torch.clamp(sample[:,2:3], self.domain_beta[0], self.domain_beta[1])
        projected_sample = torch.cat([x, t, beta], dim=1)
        return projected_sample

    def forward(self):
        raise NotImplementedError
    
    def __call__(self):
        return self.forward()


class MH_Gaussian_3D(WorstCaseLoss_3D):
    """Metropolis-Hastings sampling with Gaussian proposal distribution for 3D case.
    Proposals outside of the domain are rejected."""
    def __init__(self, problem, domain_x, domain_t, domain_beta, steps, n_samples, loss_fn, covariance_matrix, device, batch_size):
        super(MH_Gaussian_3D, self).__init__(problem, domain_x, domain_t, domain_beta, steps, n_samples, loss_fn, covariance_matrix, device, batch_size)

    def forward(self):
        self.problem.model.eval()
        
        current_x = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        current_t = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_t[1] - self.domain_t[0]) + self.domain_t[0]
        current_beta = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_beta[1] - self.domain_beta[0]) + self.domain_beta[0]
        current_sample = torch.cat([current_x, current_t, current_beta], dim=1)
        current_loss = self.loss_fn(current_sample[:,0:1], current_sample[:,1:2], current_sample[:,2:3]).detach()  # input to loss_fn is e.g., (x,t,beta)
        current_loss = torch.clamp(current_loss, min=1e-9)

        samples_store = []

        for _ in range(self.steps):
            # gaussian proposal
            proposal_distribution = MultivariateNormal(current_sample, self.covariance_matrix.expand(self.batch_size, 3, 3))      

            # sample from the proposal distribution (one sample per markov chain)
            proposed_sample = proposal_distribution.sample()
            # proposed_sample = self.project_onto_domain(proposed_sample)
            in_domain = self.in_domain(proposed_sample)

            # evaluate the target distribution (loss) at proposed samples
            proposed_loss = self.loss_fn(proposed_sample[:,0:1], proposed_sample[:,1:2], proposed_sample[:,2:3]).detach()
            proposed_loss = torch.clamp(proposed_loss, min=1e-9) 

            # acceptance probability for each proposal
            acceptance_probability = torch.min(torch.ones_like(current_loss, device=self.device), proposed_loss / current_loss)

            # decide to accept or reject the proposal (for each sample in the batch)
            accept = (torch.rand((self.batch_size, 1), device=self.device) < acceptance_probability) & in_domain
            accept = accept.squeeze() 

            # update where proposals are accepted
            current_sample[accept] = proposed_sample[accept]
            current_loss[accept] = proposed_loss[accept]

            samples_store.append(current_sample.clone())

        samples_store = torch.cat(samples_store, dim=0)
        samples_store = samples_store[-self.n_samples:]       # return only the last n_samples (since markov chain needs some warmup to get sample from the target distribution)

        self.problem.model.train()
        return samples_store


# ======================= 4D case =======================


class WorstCaseLoss_4D:
    """Base class to compute worst-case losses for 4D case, e.g., sampling (x,t,nu,rho)"""

    def __init__(self, problem, domain_x, domain_t, domain_nu, domain_rho, steps, n_samples, loss_fn, covariance_matrix, device, batch_size=1):
        self.problem = problem          # csl problem
        self.domain_x = domain_x        
        self.domain_t = domain_t
        self.domain_nu = domain_nu
        self.domain_rho = domain_rho
        self.steps = steps
        self.n_samples = n_samples
        self.loss_fn = loss_fn          # function/distribution (without normalizing constant) to sample from
        self.covariance_matrix = covariance_matrix
        self.device = device
        self.batch_size = batch_size    # batch size for MH (used to speed up sampling by sampling by constructing multiple markov chains)

    def in_domain(self, sample):
        """Check if proposed sample is within the domain"""
        in_domain = (self.domain_x[0] <= sample[:,0:1]) & (sample[:,0:1] <= self.domain_x[1]) & \
                    (self.domain_t[0] <= sample[:,1:2]) & (sample[:,1:2] <= self.domain_t[1]) & \
                    (self.domain_nu[0] <= sample[:,2:3]) & (sample[:,2:3] <= self.domain_nu[1]) & \
                    (self.domain_rho[0] <= sample[:,3:4]) & (sample[:,3:4] <= self.domain_rho[1])
        return in_domain
    
    def project_onto_domain(self, sample):
        """Project sample onto the domain."""
        x = torch.clamp(sample[:,0:1], self.domain_x[0], self.domain_x[1])
        t = torch.clamp(sample[:,1:2], self.domain_t[0], self.domain_t[1])
        nu = torch.clamp(sample[:,2:3], self.domain_nu[0], self.domain_nu[1])
        rho = torch.clamp(sample[:,3:4], self.domain_rho[0], self.domain_rho[1])
        projected_sample = torch.cat([x, t, nu, rho], dim=1)
        return projected_sample

    def forward(self):
        raise NotImplementedError
    
    def __call__(self):
        return self.forward()


class MH_Gaussian_4D(WorstCaseLoss_4D):
    """Metropolis-Hastings sampling with Gaussian proposal distribution for 4D case.
    Proposals outside of the domain are rejected."""
    def __init__(self, problem, domain_x, domain_t, domain_nu, domain_rho, steps, n_samples, loss_fn, covariance_matrix, device, batch_size):
        super(MH_Gaussian_4D, self).__init__(problem, domain_x, domain_t, domain_nu, domain_rho, steps, n_samples, loss_fn, covariance_matrix, device, batch_size)

    def forward(self):
        self.problem.model.eval()
        
        current_x = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        current_t = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_t[1] - self.domain_t[0]) + self.domain_t[0]
        current_nu = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_nu[1] - self.domain_nu[0]) + self.domain_nu[0]
        current_rho = torch.rand((self.batch_size, 1), dtype=torch.float32, device=self.device) * (self.domain_rho[1] - self.domain_rho[0]) + self.domain_rho[0]
        current_sample = torch.cat([current_x, current_t, current_nu, current_rho], dim=1)
        current_loss = self.loss_fn(current_sample[:,0:1], current_sample[:,1:2], current_sample[:,2:4]).detach()  # input to loss_fn is (x,t,pde_param)
        current_loss = torch.clamp(current_loss, min=1e-9)

        samples_store = []
        for _ in range(self.steps):
            # gaussian proposal
            proposal_distribution = MultivariateNormal(current_sample, self.covariance_matrix.expand(self.batch_size, 4, 4))      

            # sample from the proposal distribution (one sample per markov chain)
            proposed_sample = proposal_distribution.sample()

            # proposed_sample = self.project_onto_domain(proposed_sample)
            in_domain = self.in_domain(proposed_sample)

            # evaluate the target distribution (loss) at proposed samples
            proposed_loss = self.loss_fn(proposed_sample[:,0:1], proposed_sample[:,1:2], proposed_sample[:,2:4]).detach()   # input is (x,t, pde_param)
            proposed_loss = torch.clamp(proposed_loss, min=1e-9) 

            # acceptance probability for each proposal
            acceptance_probability = torch.min(torch.ones_like(current_loss, device=self.device), proposed_loss / current_loss)

            # decide to accept or reject the proposal (for each sample in the batch)
            accept = (torch.rand((self.batch_size, 1), device=self.device) < acceptance_probability) & in_domain
            accept = accept.squeeze() 

            # update where proposals are accepted
            current_sample[accept] = proposed_sample[accept]
            current_loss[accept] = proposed_loss[accept]

            samples_store.append(current_sample.clone())

        samples_store = torch.cat(samples_store, dim=0)
        samples_store = samples_store[-self.n_samples:]       # return only the last n_samples (since markov chain needs some warmup to get sample from the target distribution)

        self.problem.model.train()
        return samples_store
    
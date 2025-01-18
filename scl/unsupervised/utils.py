import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_random(X_all, N):
    """Given an array of (x,t) points, sample N points from this at radom."""
    set_seed(0)

    idx = np.random.choice(X_all.shape[0], N, replace=False)
    X_sampled = X_all[idx, :]

    return X_sampled


def plot_errors(rel_error, abs_error, linf_error, epochs, eval_every, path_save):
    epochs_eval = np.arange(0, epochs, eval_every) + 1

    # plot all errors in one plot
    plt.figure()
    plt.plot(epochs_eval, rel_error, color='blue')
    plt.title('Relative Error')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    plt.grid(True)
    plt.savefig(f'{path_save}/relative_error.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.plot(epochs_eval, abs_error, color='blue')
    plt.title('Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    plt.savefig(f'{path_save}/absolute_error.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(epochs_eval, linf_error, color='blue')
    plt.title('linf Error')
    plt.xlabel('Epochs')
    plt.ylabel('linf Error')
    plt.grid(True)
    plt.savefig(f'{path_save}/linf_error.pdf', format='pdf', bbox_inches='tight')
    plt.close()   


def plot_errors_parametric_solution(rel_error, abs_error, linf_error, epochs, eval_every, betas, path_save):
    epochs_eval = np.arange(0, epochs, eval_every) + 1
    for beta in betas:
        # plot Relative Error
        plt.figure()
        plt.plot(epochs_eval, rel_error[f'Relative Error Beta={beta}'], color='blue')
        plt.title(f'Relative Error for beta={beta}')
        plt.xlabel('Epochs')
        plt.ylabel('Relative Error')
        plt.grid(True)
        plt.savefig(f'{path_save}/relative_error_beta_{beta}.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        # plot Absolute Error
        plt.figure()
        plt.plot(epochs_eval, abs_error[f'Absolute Error Beta={beta}'], color='blue')
        plt.title(f'Absolute Error for beta={beta}')
        plt.xlabel('Epochs')
        plt.ylabel('Absolute Error')
        plt.grid(True)
        plt.savefig(f'{path_save}/absolute_error_beta_{beta}.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        # plot Linf Error
        plt.figure()
        plt.plot(epochs_eval, linf_error[f'linf Error Beta={beta}'], color='blue')
        plt.title(f'linf Error for beta={beta}')
        plt.xlabel('Epochs')
        plt.ylabel('linf Error')
        plt.grid(True)
        plt.savefig(f'{path_save}/linf_error_beta_{beta}.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # plot all errors in one plot
    plt.figure()
    for beta in betas:
        plt.plot(epochs_eval, rel_error[f'Relative Error Beta={beta}'], label=f'Relative Error Beta={beta}')
    plt.title('Relative Error')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    plt.legend(frameon=True, loc='best', fontsize=12)
    plt.grid(True)
    plt.savefig(f'{path_save}/relative_error.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    plt.figure()
    for beta in betas:
        plt.plot(epochs_eval, abs_error[f'Absolute Error Beta={beta}'], label=f'Absolute Error Beta={beta}')
    plt.title('Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('Absolute Error')
    plt.legend(frameon=True, loc='best', fontsize=12)
    plt.grid(True)
    plt.savefig(f'{path_save}/absolute_error.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    for beta in betas:
        plt.plot(epochs_eval, linf_error[f'linf Error Beta={beta}'], label=f'linf Error Beta={beta}')
    plt.title('linf Error')
    plt.xlabel('Epochs')
    plt.ylabel('linf Error')
    plt.legend(frameon=True, loc='best', fontsize=12)
    plt.grid(True)
    plt.savefig(f'{path_save}/linf_error.pdf', format='pdf', bbox_inches='tight')
    plt.close()   
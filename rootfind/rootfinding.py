import torch
from .broyden import broyden


def compute_ic(guess_in, func, batch_size, state_dim, device, eps=1e-9, threshold=100, ls=True):
    # guess = guess_mag*(2.*torch.rand(batch_size, 1, state_dim) - 1.)
    # return broyden(func, guess, threshold, eps, ls=ls, name="unknown")['result'].view(-1, state_dim)
    guess = guess_in#torch.zeros(batch_size, 1, state_dim).to(device)
    return broyden(func, guess, threshold, eps, ls=ls, device=device).view(-1, state_dim).detach()

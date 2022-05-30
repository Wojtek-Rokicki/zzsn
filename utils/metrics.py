import torch
from torch import Tensor
import torch.nn as nn
import torch.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import numpy.random as npr
import yaml
import os.path as osp
from easydict import EasyDict
from torch import zeros
from scipy.stats import wasserstein_distance
from ot.lp import emd, emd2                 # Optimal transport solvers


def check_size(set1: Tensor, set2: Tensor):
    """ Args:
        set1: Tensor of a set [batch size, point per set, dimension]
        set2: Tensor of a set [batch size, point per set, dimension]
        both dimension must be equal
    Returns:
        The Chamfer distance between both sets
    """
    bs1, n, d1 = set1.size()
    bs2, m, d2 = set2.size()
    assert (d1 == d2 and bs1 == bs2)  # Both sets must live in the same space to be compared


def chamfer_loss(set1: Tensor, set2: Tensor) -> torch.Tensor:
        check_size(set1, set2)
        dist = torch.cdist(set1, set2, 2)
        out_dist, _ = torch.min(dist, dim=2)
        out_dist2, _ = torch.min(dist, dim=1)
        total_dist = (torch.mean(out_dist) + torch.mean(out_dist2)) / 2
        return total_dist, None             # TODO: compute the assignment


def hungarian_loss(set1, set2) -> torch.Tensor:
    """ set1, set2: (bs, N, 3)"""
    check_size(set1, set2)
    set1=set1.type(torch.float64)
    set2=set2.type(torch.float64)
    batch_dist = torch.cdist(set1, set2, 2)
    numpy_batch_dist = batch_dist.detach().cpu().numpy()            # bs x n x n
    numpy_batch_dist[np.isnan(numpy_batch_dist)] = 1e6
    indices = map(linear_sum_assignment, numpy_batch_dist)
    indices = list(indices)
    loss = [dist[row_idx, col_idx].mean() for dist, (row_idx, col_idx) in zip(batch_dist, indices)]
    # Sum over the batch (not mean, which would reduce the importance of sets in big batches)
    total_loss = torch.sum(torch.stack(loss))
    return total_loss, indices


class VAELoss(nn.Module):
    def __init__(self, loss, lmbdas: list, predict_n: bool):
        """  lmbda: penalization of the Gaussian prior on the latent space
             loss: any loss function between sets. """
        super().__init__()
        self.lmbdas = np.array(lmbdas)
        self.loss = loss
        self.predict_n = predict_n
        if self.predict_n:
            self.n_loss = torch.nn.L1Loss(reduction='sum')

    def forward(self, output: Tensor, mu: Tensor, log_var: Tensor, n: Tensor, real: Tensor):
        """
        Args:
            output: output of the network: [[bs, n, 3], [bs, n, valency_max], None]
            mu: mean computed in the network: [bs, latent_dim]
            log_var: log variance computed in the network: [bs, latent_dim]
            real: expected value to compare to the output: [[bs, n, 3], [bs, valency_max], None]
        Returns:
            The variational loss computed as the sum of the hungarian loss and the Kullback-Leiber divergence.
        """
        output_set = output[0]
        
        bs, n, _ = output_set.shape

        device = output_set.device
        real_set = real


        real_n = float(real_set.shape[1]) * torch.ones(real_set.shape[0], dtype=torch.float32).to(device)
        # print(output_set.min().item(), output_set.max().item())
        reconstruction_loss, assignment = self.loss(output_set, real_set)

        dkl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.pow(2)) if log_var is not None else 0.0
        n = torch.tensor([n])
        n_loss = self.n_loss(n, real_n) if self.predict_n else zeros(1).to(device)
        total_loss = reconstruction_loss + \
                     self.lmbdas[0] * dkl + \
                     self.lmbdas[1] * n_loss

        all_metrics = [total_loss, reconstruction_loss, self.lmbdas[0] * dkl, self.lmbdas[1] * n_loss]
        return all_metrics


class HungarianVAELoss(VAELoss):
    def __init__(self, *args):
        """  lmbda: penalization of the Gaussian prior on the latent space. """
        super().__init__(hungarian_loss, *args)


class ChamferVAELoss(VAELoss):
    def __init__(self, *args):
        super().__init__(chamfer_loss, *args)


def constrained_loss(generated, config): 
    """ TODO  MSE PRECISION, RECALL"""
    loss_total = 0

    return loss_total


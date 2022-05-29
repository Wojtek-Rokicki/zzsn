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
    set1=set1.type(torch.float32)
    set2=set2.type(torch.float32)
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

        dkl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) if log_var is not None else 0.0

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


def constrained_loss(generated, config, pairs_diversity_evaluation=1000):
    """ Given access to the true generation process, compute a loss on a list of generated sets."""
    loss_total = [0.0, 0.0, 0.0, 0.0, 0.0]
    val_counts = np.zeros(config.n_max + config.extrapolation_n)
    n_dist = np.zeros(config.n_max + config.extrapolation_n + 1)
    points_total = 0

    all_valencies_0 = 0
    all_large_valencies = 0

    for i, out in enumerate(generated):
        x = out[0].squeeze(0)      # Keep only the positions, remove batch dimension

        n_dist[x.shape[0]] += 1

        # 0. Compute the loss on the bounding boss
        min_bound, max_bound = config.min_bound, config.max_bound
        r = torch.relu
        proj = (r(min_bound[0] - x[:, 0]) + r(x[:, 0] - max_bound[0]) +
                r(min_bound[1] - x[:, 1]) + r(x[:, 1] - max_bound[1]) +
                r(min_bound[2] - x[:, 2]) + r(x[:, 2] - max_bound[2]))
        dist0 = torch.sum(proj)

        # 1. Compute the loss on the nearest neighbors
        cdist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
        diag = torch.eye(cdist.shape[1], cdist.shape[1], dtype=torch.bool)
        dist1 = torch.sum(torch.relu(config.min_dist - cdist[~diag]))

        valencies = torch.sum(cdist < config.neighbor_threshold, dim=0) - 1       # Each atom is at distance 0 of itself
        # 2. There should be no isolated node or nodes with a too high valency
        # dist2 = torch.sum(valencies == 0) + torch.sum(valencies > max(config.atoms_val))
        # all_valencies_0 += torch.sum(valencies == 0)
        # all_large_valencies += torch.sum(valencies > max(config.atoms_val))

        # 3. The right proportion of points should have a given valency
        # valencies = valencies[valencies != 0]
        # valencies = valencies[valencies <= max(config.atoms_val)]
        if len(valencies > 0):
            for i, val in enumerate(np.arange(config.n_max + config.extrapolation_n)):
                val_counts[i] += torch.sum(valencies == val)

        loss_total[0] += dist0.item()
        loss_total[1] += dist1.item()
        points_total += x.shape[0]

    for i in range(len(loss_total)):
        loss_total[i] /= points_total

    val_counts = val_counts / (np.sum(val_counts) if np.sum(val_counts) > 0 else 1)

    # 4. Compute the Wasserstein distance between the histograms of valencies
    dataset_counts = config.val_statistics
    if np.sum(val_counts) == 0:   # Can happen if all valencies are wrong
        loss_total[2] = 0
    else:
        print("Distribution of the valencies:", val_counts)
        dataset_counts_long = np.zeros(val_counts.shape)
        for i, x in enumerate(dataset_counts):
            dataset_counts_long[i + 1] = x          # Skip value 0
        loss_total[2] = wasserstein_distance(np.arange(config.n_max + config.extrapolation_n),
                                             np.arange(config.n_max + config.extrapolation_n),
                                             val_counts, dataset_counts_long)

    # 5. The distribution of number of atoms should be correct
    n_dist = n_dist / np.sum(n_dist)
    dataset_n_dist = np.zeros(config.n_max + config.extrapolation_n + 1)
    dataset_n_dist[:len(config.n_dist)] = config.n_dist
    loss_total[3] = wasserstein_distance(np.arange(config.n_max + config.extrapolation_n + 1),
                                         np.arange(config.n_max + config.extrapolation_n + 1), n_dist, dataset_n_dist)

    # 6. Compute the Wasserstein distance between pairs of points to measure diversity
    num_sets = len(generated)
    rand_ints = npr.randint(0, num_sets, 2 * pairs_diversity_evaluation).reshape(-1, 2)
    points_total_w = 0
    for pair in rand_ints:
        set1 = generated[pair[0]][0]
        set2 = generated[pair[1]][0]
        points_total_w += 1
        cost_mat = torch.cdist(set1, set2).squeeze(0).cpu().numpy()
        wasserstein_d = emd2([], [], cost_mat, log=False)
        loss_total[4] += wasserstein_d / pairs_diversity_evaluation

    return loss_total


import numpy as np
import torch
from scipy.ndimage.filters import maximum_filter1d, minimum_filter
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


def euclidean_dist(x, y):
    """Compute euclidean distance between two tensors."""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)

    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
    # return torch.sqrt(torch.pow(x - y, 2).sum(2))


def cosine_dist(x, y):
    """Compute euclidean distance between two tensors."""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)

    y = y.unsqueeze(0).expand(n, m, d)

    return -torch.nn.CosineSimilarity(dim=2, eps=1e-6)(x, y)


def prototypical_loss(_input, target, n_support):
    def supp_idxs(c):
        return target.eq(c).nonzero()[:n_support].squeeze(1)

    def query_idxs(c):
        return target.eq(c).nonzero()[n_support:]

    # Preparation
    classes = torch.unique(target)
    n_classes = len(classes)
    n_query = target.eq(classes[0].item()).sum().item() - n_support

    # Compute prototypes
    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([_input[idx_list].mean(0) for idx_list in support_idxs])

    # Compute queries
    query_idxs = torch.stack(list(map(query_idxs, classes))).view(-1)
    query_samples = _input[query_idxs]
    # Compute distance between query and prototypes
    dists = euclidean_dist(
        query_samples, prototypes
    )  # torch.Size([50, 512]), torch.Size([10, 512])
    # dists = cosine_dist(query_samples, prototypes) # torch.Size([50, 512]), torch.Size([10, 512])
    log_p_y = F.log_softmax(-dists, dim=1).view(
        n_classes, n_query, -1
    )  # torch.Size([10, 5, 10]) (how many class, how many queries, how many target classes)
    dist_loss = torch.tensor([0.0]).cuda()
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    target_inds = target_inds.to(log_p_y.device)
    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    # import ipdb; ipdb.set_trace()
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss, acc_val, dist_loss


def prototypical_loss_filter_negative(_input, target, n_support):
    def all_idxs(c):
        return target.eq(c).nonzero().squeeze(1)

    def supp_idxs(c):
        return target.eq(c).nonzero()[:n_support].squeeze(1)

    def query_idxs(c):
        return target.eq(c).nonzero()[n_support:]

    # def get_positive_class_only(classes):
    # Preparation
    classes = torch.unique(target)
    # Do not calculate the distance of negtive class
    pos_classes = classes[~(classes % 2 == 1)]
    # pos_classes = classes

    n_classes = len(classes)
    n_query = target.eq(classes[0].item()).sum().item() - n_support

    # Compute prototypes
    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([_input[idx_list].mean(0) for idx_list in support_idxs])

    # Compute intra class distance
    # all_id = list(map(all_idxs, pos_classes))
    # all_features = torch.stack([_input[idx_list] for idx_list in all_id])
    # dist_loss = prototype_dist_loss_SupCon(all_features)
    # dist_loss = prototype_dist_loss(all_features)
    dist_loss = torch.tensor([0.0]).cuda()

    # Compute queries
    query_idxs = torch.stack(list(map(query_idxs, pos_classes))).view(-1)
    query_samples = _input[query_idxs]

    # Compute distance between query and prototypes
    dists = euclidean_dist(
        query_samples, prototypes
    )  # torch.Size([50, 512]), torch.Size([10, 512])
    # dists = cosine_dist(query_samples, prototypes) # torch.Size([50, 512]), torch.Size([10, 512])
    log_p_y = F.log_softmax(-dists, dim=1).view(
        n_classes // 2, n_query, -1
    )  # torch.Size([10, 5, 10]) (how many positive class, how many queries, how many target classes)

    target_inds = torch.arange(0, n_classes // 2) * 2
    target_inds = target_inds.view(n_classes // 2, 1, 1)
    target_inds = target_inds.expand(n_classes // 2, n_query, 1).long()
    target_inds = target_inds.to(log_p_y.device)
    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss, acc_val, dist_loss


def prototype_dist_loss(features):
    prototypes = torch.mean(features, dim=1, keepdim=True)
    dist = torch.mean(torch.abs(features - prototypes), dim=-1)
    return dist.mean()


def prototype_dist_loss_SupCon(features):
    from src.utils.supconloss import SupConLoss

    supcon = SupConLoss()
    labels = torch.tensor(list(range(0, features.size(0)))).cuda()
    return supcon(torch.nn.functional.normalize(features, p=2.0, dim=2), labels)


if __name__ == "__main__":
    import ipdb

    ipdb.set_trace()

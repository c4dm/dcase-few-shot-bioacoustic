
# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from Model import Protonet
from Datagenerator import Datagen_test
import numpy as np
from batch_sampler import EpisodicBatchSampler
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter1d, minimum_filter
from torch.autograd import Variable




def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)

    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)



def prototypical_loss(input, target,n_support):
    '''
    Adopted from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
    Compute the prototypes by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      bprototypes, for each one of the current classes
    '''

    def supp_idxs(c):

        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    p = n_classes * n_support
    n_query = target.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs,classes))
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    query_idxs = torch.stack(list(map(lambda c:target.eq(c).nonzero()[n_support:],classes))).view(-1)
    query_samples = input.cpu()[query_idxs]


    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)

    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val


def get_probability(x_pos,neg_proto,query_set_out):


    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set

     Out:
     - Probabiility array for the positive class
     """

    pos_prototype = x_pos.mean(0)
    prototypes = torch.stack([pos_prototype,neg_proto])
    dists = euclidean_dist(query_set_out,prototypes)
    '''  Taking inverse distance for converting distance to probabilities'''
    inverse_dist = torch.div(1.0, dists)
    prob = torch.softmax(inverse_dist,dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,0]

    return prob_pos.detach().cpu().tolist()



def evaluate_prototypes(conf=None,hdf_eval=None,device= None,strt_index_query=None):

    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Features from the audio file
     - device:  cuda/cpu
     - str_index_query : start frame of the query set w.r.t to the original file

     Out:
     - onset: Onset array predicted by the model
     - offset: Offset array predicted by the model
      """
    hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel)

    gen_eval = Datagen_test(hdf_eval,conf)
    X_pos, X_neg,X_query = gen_eval.generate_eval()

    X_pos = torch.tensor(X_pos)
    Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
    X_neg = torch.tensor(X_neg)
    Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
    X_query = torch.tensor(X_query)
    Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

    num_batch_query = len(Y_query) // conf.eval.query_batch_size

    query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=conf.eval.query_batch_size,shuffle=False)
    query_set_feat = torch.zeros(0,1024).cpu()


    Model = Protonet()

    if device == 'cpu':
        Model.load_state_dict(torch.load(conf.path.best_model, map_location=torch.device('cpu')))
    else:
        Model.load_state_dict(torch.load(conf.path.best_model))

    Model.to(device)
    Model.eval()

    'List for storing the combined probability across all iterations'
    prob_comb = []

    iterations = conf.eval.iterations
    for i in range(iterations):
        prob_pos_iter = []
        neg_indices = torch.randperm(len(X_neg))[:conf.eval.samples_neg]
        X_neg = X_neg[neg_indices]
        Y_neg = Y_neg[neg_indices]
        batch_size_neg = conf.eval.negative_set_batch_size
        neg_dataset = torch.utils.data.TensorDataset(X_neg, Y_neg)
        negative_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None, batch_size=batch_size_neg)

        batch_samplr_pos = EpisodicBatchSampler(Y_pos, num_batch_query + 1, 1, conf.train.n_shot)
        pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
        pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=batch_samplr_pos)

        neg_iterator = iter(negative_loader)
        pos_iterator = iter(pos_loader)
        q_iterator = iter(q_loader)

        print("Iteration number {}".format(i))

        for batch in tqdm(neg_iterator):
            x_neg, y_neg = batch
            x_neg = x_neg.to(device)
            feat_neg = Model(x_neg)
            feat_neg = feat_neg.detach().cpu()
            query_set_feat = torch.cat((query_set_feat, feat_neg), dim=0)
        neg_proto = query_set_feat.mean(dim=0)
        neg_proto =neg_proto.to(device)

        for batch in tqdm(q_iterator):
            x_q, y_q = batch
            x_q = x_q.to(device)
            x_pos, y_pos = next(pos_iterator)
            x_pos = x_pos.to(device)
            x_pos = Model(x_pos)
            x_query = Model(x_q)
            probability_pos = get_probability(x_pos, neg_proto, x_query)
            prob_pos_iter.extend(probability_pos)

        prob_comb.append(prob_pos_iter)
    prob_final = np.mean(np.array(prob_comb),axis=0)

    krn = np.array([1, -1])
    prob_thresh = np.where(prob_final > 0.5, 1, 0)

    prob_pos_final = prob_final * prob_thresh
    changes = np.convolve(krn, prob_thresh)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr

    onset = (onset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    onset = onset + str_time_query

    offset = (offset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    return onset, offset























# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from Model import Protonet
from Datagenerator import Datagen_test
import numpy as np
from batch_sampler1 import EpisodicBatchSampler
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
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    n_query=n_support

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    p = n_classes * n_support

    prototypes = input_cpu[:p].view(n_support, n_classes, -1).mean(0)
    query_set = input_cpu[p:]

    dists = euclidean_dist(query_set, prototypes)


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

    pos_prototypes = x_pos.mean(0)
    prototypes = torch.stack([pos_prototypes,neg_proto])
    dists = euclidean_dist(query_set_out,prototypes)
    '''  Taking inverse distance for converting distance to probabilities'''
    inverse_dist = torch.div(1.0, dists)
    prob = torch.softmax(inverse_dist,dim=1)

    '''  Probability array for positive class'''
    prob_pos = prob[:,0]

    return prob_pos.detach().cpu().tolist()


def evaluate_prototypes(conf=None,hdf_eval=None,device= None,str_time_query=None):

    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Log mel features from the audio file
     - device:  cuda/cpu
     - str_time_query (secs) : timestamp of the query set w.r.t to the original file

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

    num_batch_query = len(Y_query) // 8

    num_batch_neg = num_batch_query + 1


    batch_samplr_neg = EpisodicBatchSampler(Y_neg,num_batch_neg,1,conf.eval.samples_neg)
    batch_samplr_pos = EpisodicBatchSampler(Y_pos, num_batch_query + 1, 1, conf.train.n_shot)

    neg_dataset = torch.utils.data.TensorDataset(X_neg, Y_neg)
    pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
    query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)


    pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=batch_samplr_pos)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=8,shuffle=False)
    negative_loader = torch.utils.data.DataLoader(dataset=neg_dataset,batch_sampler=batch_samplr_neg)
    prob_final = []

    Model = Protonet()

    if device == 'cpu':
        Model.load_state_dict(torch.load(conf.path.best_model, map_location=torch.device('cpu')))
    else:
        Model.load_state_dict(torch.load(conf.path.best_model))

    Model.to(device)
    Model.eval()

    prob_pos_iter = []
    neg_iterator = iter(negative_loader)
    pos_iterator = iter(pos_loader)
    q_iterator = iter(q_loader)
    for batch in tqdm(q_iterator):
        x_q, y_q = batch
        x_q = x_q.to(device)
        x_pos, y_pos = next(pos_iterator)
        x_neg, y_neg = next(neg_iterator)

        x_pos = x_pos.to(device)
        x_neg = x_neg.to(device)
        x_pos = Model(x_pos)
        x_neg = Model(x_neg)
        x_query = Model(x_q)

        neg_proto = x_neg.mean(0)
        probability_pos = get_probability(x_pos, neg_proto, x_query)
        prob_pos_iter.append(probability_pos)
    prob_iter_flat = [item for sublist in prob_pos_iter for item in sublist]
        #prob_final.append(prob_iter_flat)
    #prob_pos_final = np.mean(np.array(prob_final),axis=0)
    prob_pos_final = np.array(prob_iter_flat)


    krn = np.array([1, -1])
    prob_thresh = np.where(prob_pos_final> 0.5,1,0)

    changes = np.convolve(krn,prob_thresh)

    onset_frames = np.where(changes==1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_frame_query = str_time_query * conf.features.hop_mel / conf.features.sr

    onset = (onset_frames+1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    onset = onset + str_frame_query

    offset = (offset_frames+1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    offset = offset+  str_frame_query


    assert len(onset) == len(offset)
    return onset,offset
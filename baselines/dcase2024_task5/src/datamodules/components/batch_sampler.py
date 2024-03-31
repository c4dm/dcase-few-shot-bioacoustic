import torch
import numpy as np
import torch.utils.data as data


""" Episodic batch sampler adoted from https://github.com/jakesnell/prototypical-networks/"""


class EpisodicBatchSampler(data.Sampler):
    def __init__(self, labels, n_episodes, n_way, n_samples):
        """
        Sampler that yields batches per n_episodes without replacement.
        Batch format: (c_i_1, c_j_1, ..., c_n_way_1, c_i_2, c_j_2, ... , c_n_way_2, ..., c_n_way_n_samples)
        Args:
            label: List of sample labels (in dataloader loading order)
            n_episodes: Number of episodes or equivalently batch size
            n_way: Number of classes to sample
            n_samples: Number of samples per episode (Usually n_query + n_support)
        """

        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_samples = n_samples

        labels = np.array(labels)
        self.samples_indices = []
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.samples_indices.append(ind)

        if self.n_way > len(self.samples_indices):
            raise ValueError(
                'Error: "n_way" parameter is higher than the unique number of classes'
            )

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.samples_indices))[: self.n_way]
            for c in classes:
                l = self.samples_indices[c]
                pos = torch.randperm(len(l))[: self.n_samples]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

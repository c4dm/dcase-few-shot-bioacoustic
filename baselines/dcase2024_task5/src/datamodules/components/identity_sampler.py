# import numpy as np
# import torch
# import torch.utils.data as data


# class IdentityBatchSampler(data.Sampler):
#     def __init__(self, train_param, num_classes, batch_size, n_episode):
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.train_param = train_param
#         self.index = list(range(self.num_classes))
#         self.n_episode = n_episode

#     def __len__(self):
#         return self.n_episode

#     def __iter__(self):
#         for _ in range(self.n_episode):
#             index = np.random.permutation(self.index)[: self.train_param.k_way]
#             index = np.tile(index, self.batch_size)  # repeat for self.batch_size times
#             yield index

import numpy as np
import torch
import torch.utils.data as data


class IdentityBatchSampler(data.Sampler):
    def __init__(
        self,
        train_param,
        train_eval_class_idxs,
        extra_train_class_idxs,
        batch_size,
        n_episode,
    ):
        self.num_classes = len(train_eval_class_idxs + extra_train_class_idxs)
        self.batch_size = batch_size
        self.train_param = train_param
        self.train_eval_class_idxs = train_eval_class_idxs
        self.extra_train_class_idxs = extra_train_class_idxs
        self.n_episode = n_episode

    def __len__(self):
        return self.n_episode

    def __iter__(self):
        for _ in range(self.n_episode):
            if len(self.extra_train_class_idxs) > 0:
                index_1 = np.random.permutation(self.train_eval_class_idxs)[
                    : self.train_param.k_way // 2
                ]
                index_2 = np.random.permutation(self.extra_train_class_idxs)[
                    : self.train_param.k_way // 2
                ]
                index = np.concatenate([index_1, index_2])
            else:
                index = np.random.permutation(self.train_eval_class_idxs)[
                    : self.train_param.k_way
                ]
            index = np.tile(index, self.batch_size)  # repeat for self.batch_size times
            yield index

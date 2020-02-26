from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


# class RandomIdentitySampler(Sampler):
#     def __init__(self, data_source):
#         self.data_source = data_source
#         self.num_instances = 0
#         self.index_dic = defaultdict(list)
#         unique_ids = np.unique(data_source.ids)
#         self.num_samples = len(unique_ids)
#         for (index, value) in enumerate(data_source.ids):
#             self.index_dic[value].append(index)
#         for mini_list in self.index_dic.values():
#             if self.num_instances == 0:
#                 self.num_instances = len(mini_list)
#             else:
#                 if self.num_instances > len(mini_list):
#                     self.num_instances = len(mini_list)
#         self.pids = list(self.index_dic.keys())

#     def __len__(self):
#         return self.num_samples * self.num_instances

#     def __iter__(self):
#         indices = torch.randperm(self.num_samples)
#         ret = []
#         for i in indices:
#             pid = self.pids[i]
#             t = self.index_dic[pid]
#             if len(t) >= self.num_instances:
#                 t = np.random.choice(t, size=self.num_instances, replace=False)
#             else:
#                 t = np.random.choice(t, size=self.num_instances, replace=True)
#             ret.extend(t)
#         return iter(ret)

class RandomIdentitySampler(Sampler):
    """
    Sampler for triplet semihard sample mining.

    Attributes:
        _id2index (dict of list): mapping from person id to its image indexes in `data_source`
    """

    @staticmethod
    def _sample(population, k):
        if len(population) >= k:
            pop = np.random.choice(population, size=k, replace=False)
        else:
            pop = np.random.choice(population, size=k, replace=True)
        return pop

    def __init__(self, data_source, id_minibatch):
        """
        :param data_source: Market1501 dataset
        :param batch_image: batch image size for one person id
        """
        super(RandomIdentitySampler, self).__init__(data_source)
        self.data_source = data_source
        self.id_minibatch = id_minibatch

        self._id2index = defaultdict(list)
        for idx, id in enumerate(data_source.ids):
            self._id2index[id].append(idx)
        
        # for key in self._id2index.keys():
        #     idxes = self._id2index[key]
        #     if self.id_minibatch == 0:
        #         self.id_minibatch = len(idxes)
        #     else:
        #         if self.id_minibatch > len(idxes):
        #             self.id_minibatch = len(idxes)

    def __iter__(self):
        unique_ids = np.unique(self.data_source.ids)
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.id_minibatch))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.id_minibatch
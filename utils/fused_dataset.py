import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from PIL import Image
import torchvision
from . import market1501, veri776

class fuse(data.Dataset):
    def __init__(self, set1_path, set2_path, transform=None, dataset_name=None, has_gt=True):
        self.transform = transform
        self.set1 = market1501.Market1501(set1_path, transform=self.transform, dataset_name="Set1")
        self.set2 = veri776.VeRi776(set2_path, transform=self.transform, dataset_name="Set2")
        
        self.imgs = self.set1.imgs + self.set2.imgs
        set2_ids = [i + 1502 for i in self.set2.ids]
        self.ids = self.set1.ids + set2_ids
        set2_cams = [i + 10 for i in self.set2.cameras]
        self.cameras = self.set1.cameras + set2_cams

        self.name = dataset_name
        self.has_gt = has_gt
        self._id2label = {_id: idx for idx, _id in enumerate(np.unique(self.ids))}

    def __getitem__(self, index):
        path = self.imgs[index]
        label = self._id2label[self.ids[index]]
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)
        
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.ids)


class Fused_Dataset():

    def __init__(self, market_root, veri_root, train_transform, test_transform):
        self.market_train_path = os.path.join(market_root, "bounding_box_train")
        self.market_test_path = os.path.join(market_root, "bounding_box_test")
        self.market_query_path = os.path.join(market_root, "query")
        self.veri_train_path = os.path.join(veri_root, "image_train")
        self.veri_test_path = os.path.join(veri_root, "image_test")
        self.veri_query_path = os.path.join(veri_root, "image_query")

        self.train = fuse(self.market_train_path, self.veri_train_path, train_transform, dataset_name= "Fused Train")
        self.test = fuse(self.market_test_path, self.veri_test_path, test_transform, dataset_name= "Fused Test")
        self.query = fuse(self.market_query_path, self.veri_query_path, test_transform, dataset_name= "Fused Query")
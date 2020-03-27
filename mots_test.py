import sys
sys.path.append('/home/aistudio/external-libraries')

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
from termcolor import colored

from scipy.spatial.distance import cdist
# from original_model import Net
from model64_v1_2 import Net
from utils import util, eval_tools, triplet, sampler, mots_reid

parser = argparse.ArgumentParser(description="Train on market1501 and veri776")
parser.add_argument("--mots_dir",default='data',type=str)
parser.add_argument("--mask",default=False,type=bool)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument('--batch_size', default=512, type=int, help='Batch size for test')
parser.add_argument('--num_workers', default=0, type=int, help='threads to load data')
parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint state_dict file to evaluate.')
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

print('Loading......')
mots_root = args.mots_dir
train_dir = os.path.join(mots_root, "train")
test_dir = os.path.join(mots_root, "test")
query_dir = os.path.join(mots_root, "query")
train_set = mots_reid.MOTS_REID(train_dir, mask=args.mask, train=True, dataset_name="mots reid train")
test_set = mots_reid.MOTS_REID(test_dir, mask=args.mask, train=False, dataset_name="mots reid test")
query_set = mots_reid.MOTS_REID(query_dir, mask=args.mask, train=False, dataset_name="mots reid query")
# trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=sampler.RandomIdentitySampler(train_set, args.num_ids), num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
queryloader = torch.utils.data.DataLoader(query_set, batch_size=args.batch_size, num_workers=args.num_workers)


# net definition
num_classes = len(np.unique(train_set.ids))
net = Net(num_classes=num_classes)
assert os.path.isfile(args.checkpoint), "Error: no checkpoint file found!"

checkpoint = torch.load(args.checkpoint)
net_dict = checkpoint['net_dict']
# for key in list(net_dict.keys()):
#     if key.startswith('classifier'):
#         del net_dict[key]
net.load_state_dict(net_dict)
net.to(device)
print("Done")

def eval():
    net.is_train = False
    net.eval()
    print("Evaluating......")
    query = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in queryloader])
    test = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in testloader])
    dist = cdist(query, test)

    r = eval_tools.cmc(dist, query_set.ids, test_set.ids, query_set.cameras, test_set.cameras,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True,
            same_cam_valid=True)

    m_ap = eval_tools.mean_ap(dist, query_set.ids, test_set.ids, query_set.cameras, test_set.cameras, same_cam_valid=True)
    print(colored('model:%s  mAP=%f, r@1=%f, r@3=%f, r@5=%f, r@10=%f' % (os.path.basename(args.checkpoint), m_ap, r[0], r[2], r[4], r[9]), "red"))
            


if __name__ == '__main__':
    eval()
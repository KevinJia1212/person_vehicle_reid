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
from utils import market1501, veri776, util, eval_tools, fused_dataset, triplet, sampler

parser = argparse.ArgumentParser(description="Train on market1501 and veri776")
parser.add_argument("--market_dir",default='data',type=str)
parser.add_argument("--veri_dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument('--batch_size', default=512, type=int, help='Batch size for test')
parser.add_argument('--num_workers', default=4, type=int, help='threads to load data')
parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint state_dict file to evaluate.')
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# transform defination
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# data loading
market_root = args.market_dir
veri_root = args.veri_dir
data = fused_dataset.Fused_Dataset(market_root, veri_root, transform_train, transform_test)
testloader = torch.utils.data.DataLoader(data.test, batch_size=args.batch_size, num_workers=args.num_workers)
queryloader = torch.utils.data.DataLoader(data.query, batch_size=args.batch_size, ,num_workers=args.num_workers)

# net definition
num_classes = len(np.unique(data.train.ids))
net = Net(num_classes=num_classes)
assert os.path.isfile(args.checkpoint), "Error: no checkpoint file found!"
print('Loading......')
checkpoint = torch.load(args.checkpoint)
net_dict = checkpoint['net_dict']
# for key in list(net_dict.keys()):
#     if key.startswith('classifier'):
#         del net_dict[key]
net.load_state_dict(net_dict)
net.to(device)

def eval():
    net.is_train = False
    net.eval()
    print("Evaluating......")
    query = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in queryloader])
    test = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in testloader])
    dist = cdist(query, test)

    r = eval_tools.cmc(dist, data.query.ids, data.test.ids, data.query.cameras, data.test.cameras,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True)

    m_ap = eval_tools.mean_ap(dist, data.query.ids, data.test.ids, data.query.cameras, data.test.cameras)
    print(colored('model:%s  mAP=%f, r@1=%f, r@3=%f, r@5=%f, r@10=%f' % (os.path.basename(args.checkpoint), m_ap, r[0], r[2], r[4], r[9]), "red"))

    # test_loss = 0.
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for idx, (inputs, labels) in enumerate(queryloader):
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         features = net(inputs)
    #         loss = ce_loss(classes, labels)

    #         test_loss += loss.item()
    #         correct += outputs.max(dim=1)[1].eq(labels).sum().item()
    #         total += labels.size(0)
        
    #     print("Testing ...")
    #     end = time.time()
    #     print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
    #             100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
    #         ))

    # # saving checkpoint
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     best_acc = acc
    #     print("Saving parameters to checkpoint/ckpt.t7")
    #     checkpoint = {
    #         'net_dict':net.state_dict(),
    #         'acc':acc,
    #         'epoch':epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(checkpoint, './checkpoint/ckpt.t7')

    # return test_loss/len(testloader), 1.- correct/total
            


if __name__ == '__main__':
    eval()
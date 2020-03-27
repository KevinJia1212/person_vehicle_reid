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
from utils import mots_reid, util, eval_tools, triplet, sampler

parser = argparse.ArgumentParser(description="Fine-tuning on mots reid dataset")
parser.add_argument("--mots_dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.00001, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--batch_size', default=512, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from.')
parser.add_argument('--init_from', default="interrupt", type=str, help='init from pretrained model or interrupt model')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--margin', default=1.5, type=float)
parser.add_argument('--num_ids', default=8, type=int)
parser.add_argument('--epoches', default=200, type=int)
parser.add_argument('--start_epoch', default=None, type=int)
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# # transform defination
# transform_train = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((64,64)),
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize([0.176, 0.182, 0.200], [0.206, 0.210, 0.222])
# ])
# transform_test = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((64,64)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize([0.176, 0.182, 0.200], [0.206, 0.210, 0.222])
# ])

# data loading
mots_root = args.mots_dir
train_dir = os.path.join(mots_root, "train")
test_dir = os.path.join(mots_root, "test")
query_dir = os.path.join(mots_root, "query")
train_set = mots_reid.MOTS_REID(train_dir, mask=True, train=True, dataset_name="mots reid train")
test_set = mots_reid.MOTS_REID(test_dir, mask=True, train=False, dataset_name="mots reid test")
query_set = mots_reid.MOTS_REID(query_dir, mask=True, train=False, dataset_name="mots reid query")
trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=sampler.RandomIdentitySampler(train_set, args.num_ids), num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
queryloader = torch.utils.data.DataLoader(query_set, batch_size=args.batch_size, num_workers=args.num_workers)


# net definition
num_classes = len(np.unique(train_set.ids))
start_epoch = 0
start_lr = args.lr
# lr_adjust_list = [ 280, 320, 400, 460]
lr_adjust_list = [220, 360, 430]
# lr_adjust_list = [85, 120]
net = Net(num_classes=num_classes)
if args.resume is not None:
    assert os.path.isfile(args.resume), "Error: no checkpoint file found!"
    print('Loading from {}'.format(args.resume))
    if args.init_from == "interrupt":
        checkpoint = torch.load(args.resume)
        net_dict = checkpoint['net_dict']
        net.load_state_dict(net_dict)
    # best_acc = checkpoint['acc']
        if args.start_epoch != None:
            start_epoch = args.start_epoch
        else:
            start_epoch = checkpoint['epoch']
            new_lr_adjust_list = []
            for step in lr_adjust_list:
                if start_epoch >= step:
                    start_lr *= 0.1 
                else:
                    new_lr_adjust_list.append(step)
            for i in range(len(new_lr_adjust_list)):
                new_lr_adjust_list[i] -= start_epoch
            lr_adjust_list = new_lr_adjust_list

    elif args.init_from == "pretrain":
        pretrained_ckpt = torch.load(args.resume)['net_dict']
        state_dict = net.state_dict()
        for key in list(pretrained_ckpt.keys()):
            if key.startswith('classifier'):
                del pretrained_ckpt[key]
        state_dict.update(pretrained_ckpt)
        net.load_state_dict(state_dict)
        for name, value in net.named_parameters():
            if name.startswith('classifier'):
                value.requires_grad = True
            else:
                value.requires_grad = False

print("parameters to train:")
for name, value in net.named_parameters():    
    if value.requires_grad:
        print(name)

net.to(device)

# loss and optimizer
ce_loss = torch.nn.CrossEntropyLoss()
trp_loss = triplet.TripletSemihardLoss(args.margin)
# trp2_loss = triplet.TripletLoss(args.margin)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr, betas=(0.9, 0.99), weight_decay=0.0005)
# optimizer = torch.optim.SGD(net.parameters(), start_lr, momentum=0.9, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_adjust_list, gamma=0.1)
best_acc = 0.

# train function for each epoch
def train(epoch):
    print("Epoch : %d"%(epoch+1))
    net.is_train = True
    net.train()
    training_loss = 0.
    iding_loss = 0.
    triing_loss = 0.
    train_loss = 0.
    correct = 0
    precision = 0.
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device),labels.to(device)
        # print(np.unique(np.asarray(labels.cpu())))
        features, classes = net(inputs)
        id_loss = ce_loss(classes, labels)
        tri_loss, prec = trp_loss(features, labels)
        # loss = (id_loss + tri_loss) / 2.0
        loss = id_loss + tri_loss 

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        train_loss += loss.item()
        iding_loss += id_loss
        triing_loss += tri_loss
        correct += classes.max(dim=1)[1].eq(labels).sum().item()
        precision += prec
        total += labels.size(0)


        # print 
        if (idx+1)%interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s TotalLoss:{:.5f} id_loss:{:.5f} tri_loss:{:.5f} Correct:{}/{} Acc:[{:.3f}%] Prec:[{:.3f}%] lr:{:.2g}".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, iding_loss/interval, triing_loss/interval, correct, total, 100.*correct/total, 100.*precision/interval, optimizer.param_groups[0]['lr']
            ))
            training_loss = 0.
            iding_loss = 0.
            triing_loss = 0.
            precision = 0.
            start = time.time()
    
    return train_loss/len(trainloader)

# # lr decay
# def lr_decay():
#     global optimizer
#     for params in optimizer.param_groups:
#         params['lr'] *= 0.1
#         lr = params['lr']
#         print("Learning rate adjusted to {}".format(lr))

def eval(epoch):
    net.is_train = False
    net.eval()
    query = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in queryloader])
    test = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in testloader])
    dist = cdist(query, test)

    r = eval_tools.cmc(dist, query_set.ids, test_set.ids, query_set.cameras, test_set.cameras,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True,
            same_cam_valid=True)

    m_ap = eval_tools.mean_ap(dist, query_set.ids, test_set.ids, query_set.cameras, test_set.cameras,same_cam_valid=True)
    print(colored('epoch[%d]: mAP=%f, r@1=%f, r@3=%f, r@5=%f, r@10=%f' % (epoch + 1, m_ap, r[0], r[2], r[4], r[9]), "yellow"))


def main():
    try:
        for epoch in range(start_epoch, start_epoch + args.epoches):
            train_loss = train(epoch)
            scheduler.step()
            # test_loss, test_err = test(epoch)
            if (epoch+1) % 10 == 0:
                eval(epoch)
            if (epoch+1) % 100 == 0:
                print("Saving parameters to checkpoint/")
                checkpoint = {
                    'net_dict':net.state_dict(),
                    'epoch':epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                ckpt_path = "checkpoint/mots2_tune_" + str(epoch) + ".t7" 
                torch.save(checkpoint, ckpt_path)
            # draw_curve(epoch, train_loss, train_err, test_loss, test_err)
            # if (epoch+1)%10==0:
            #     lr_decay()
    except KeyboardInterrupt:
        print("Stop early. Saving checkpoint")
        checkpoint = {
                    'net_dict':net.state_dict(),
                    'epoch':epoch,
                }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        ckpt_path = "checkpoint/mots2_tune_" + str(epoch) + ".t7" 
        torch.save(checkpoint, ckpt_path)
        


if __name__ == '__main__':
    main()
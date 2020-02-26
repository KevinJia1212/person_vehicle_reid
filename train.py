import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision

from scipy.spatial.distance import cdist
# from original_model import Net
from original_model64 import Net
from utils import market1501, veri776, util, eval_tools, fused_dataset

parser = argparse.ArgumentParser(description="Train on market1501 and veri776")
parser.add_argument("--market_dir",default='data',type=str)
parser.add_argument("--veri_dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.1, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--resume', '-r',action='store_true')
parser.add_argument('--num_workers', default=4, type=int)
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
# train_path = os.path.join(dataset_root, "bounding_box_train")
# test_path = os.path.join(dataset_root, "bounding_box_test")
# query_path = os.path.join(dataset_root, "query")

train_path = os.path.join(veri_root, "image_train")
test_path = os.path.join(veri_root, "image_test")
query_path = os.path.join(veri_root, "image_query")
veri_train = veri776.VeRi776(train_path, transform=transform_train, dataset_name="Veri Train")
veri_test = veri776.VeRi776(test_path, transform=transform_test, dataset_name="Veri Test")
veri_query = veri776.VeRi776(query_path, transform=transform_test, dataset_name="Veri Query")
trainloader = torch.utils.data.DataLoader(veri_train, batch_size=128, shuffle=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(veri_test, batch_size=256)
queryloader = torch.utils.data.DataLoader(veri_query, batch_size=256)

# market_train = market1501.Market1501(train_path, transform=transform_train, dataset_name="Market Train")
# market_test = market1501.Market1501(test_path, transform=transform_test, dataset_name="Market Test")
# market_query = market1501.Market1501(query_path, transform=transform_test, dataset_name="Market Query")
# market_test = Market1501(test_filenames, test_ids, transform=transform_test, dataset_name="Market Test")
# data = fused_dataset.Fused_Dataset(market_root, veri_root, transform_train, transform_test)

# trainloader = torch.utils.data.DataLoader(data.train, batch_size=512, shuffle=True, num_workers=args.num_workers)
# testloader = torch.utils.data.DataLoader(data.test, batch_size=128)
# queryloader = torch.utils.data.DataLoader(data.query, batch_size=128)



# net definition
num_classes = len(np.unique(veri_train.ids))
start_epoch = 1
net = Net(num_classes=num_classes)
if args.resume:
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [ 15, 28, 40, 58, 72, 85, 91], gamma=0.1)
best_acc = 0.

# train function for each epoch
def train(epoch):
    print("Epoch : %d"%(epoch))
    net.reid = False
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print 
        if (idx+1)%interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:[{:.3f}%]  lr:{:.2g}".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total, scheduler.get_lr()[0]
            ))
            training_loss = 0.
            start = time.time()
    
    return train_loss/len(trainloader), 1.- correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))

    # saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss/len(testloader), 1.- correct/total

# plot figure
x_epoch = []
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")

# # lr decay
# def lr_decay():
#     global optimizer
#     for params in optimizer.param_groups:
#         params['lr'] *= 0.1
#         lr = params['lr']
#         print("Learning rate adjusted to {}".format(lr))

def eval(epoch):
    net.reid = True
    net.eval()
    query = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in queryloader])
    test = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in testloader])
    dist = cdist(query, test)

    r = eval_tools.cmc(dist, veri_query.ids, veri_test.ids, veri_query.cameras, veri_test.cameras,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True)

    m_ap = eval_tools.mean_ap(dist,veri_query.ids, veri_test.ids, veri_query.cameras, veri_test.cameras)
    print('epoch[%d]: mAP=%f, r@1=%f, r@3=%f, r@5=%f, r@10=%f' % (epoch + 1, m_ap, r[0], r[2], r[4], r[9]))


def main():
    for epoch in range(start_epoch, start_epoch+100):
        train_loss, train_err = train(epoch)
        scheduler.step()
        # test_loss, test_err = test(epoch)
        if epoch % 2 == 0:
            eval(epoch)
        if epoch % 50 == 0:
            print("Saving parameters to checkpoint/")
            checkpoint = {
                'net_dict':net.state_dict(),
                'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            ckpt_path = "./checkpoint/ckpt_" + str(epoch) + ".t7" 
            torch.save(checkpoint, ckpt_path)
        # draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        # if (epoch+1)%10==0:
        #     lr_decay()


if __name__ == '__main__':
    main()
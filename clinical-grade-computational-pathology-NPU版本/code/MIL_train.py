import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist

# parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
# parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
# parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
# parser.add_argument('--output', type=str, default='.', help='name of output file')
# parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
# parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
# parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
# parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
# parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
# parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default=r'../output/lib/cnn_train_data_lib.db', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default=r'../output/lib/cnn_val_data_lib.db', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default=r'../output/', help='name of output file')
parser.add_argument('--batch_size', type=int, default=2048, help='mini-batch size (default: 2048)')
parser.add_argument('--nepochs', type=int, default=40, help='number of epochs')
parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 16)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--world_size', default=None, type=int, help='world_size ')




best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()
    world_size = torch.npu.device_count()
    args.world_size = world_size
    #cnn
    model = models.resnet34(True)#加载预训练模型
    model.fc = nn.Linear(model.fc.in_features, 2)#全连接层的输出特征数为2
    #model.cuda()#转移到gpu
    #model = nn.DataParallel(model)#数据并行
    #多卡运行
    os.environ["MASTER_ADDR"] = "localhost" 
    os.environ["MASTER_PORT"] = "60000" 
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device('npu', local_rank)
    dist.init_process_group(backend="hccl", rank=local_rank, world_size=world_size)
    torch_npu.npu.set_device(device)
    args.batch_size = int(args.batch_size / world_size)    
    args.workers = int((4 + world_size - 1) / world_size)
    
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if args.weights==0.5:#平衡的损失函数
        criterion = nn.CrossEntropyLoss().cuda()#创建交叉熵损失函数
    else:
        w = torch.Tensor([1-args.weights, args.weights])#创建包含两个权重值的张量
        criterion = nn.CrossEntropyLoss(w).cuda()#创建交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)#Adam优化器，学习率，权重衰减

    cudnn.benchmark = True
                    
    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])#每个通道的均值和标准差
    trans = transforms.Compose([transforms.ToTensor(), normalize])#打包操作

    
    #load data
    train_dset = MILdataset(args.train_lib, trans)#接收两个参数：训练集的路径和预处理的转换    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset)
    
    train_loader = torch.utils.data.DataLoader(#数据加载器
        train_dset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False,sampler=train_sampler)
    if args.val_lib:#如果有验证集
        val_dset = MILdataset(args.val_lib, trans)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dset)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=False,sampler=val_sampler)

    #open output file
    fconv = open(os.path.join(args.output, 'CNN_convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')#写入列名
    fconv.close()
    
    #loop throuh epochs
    for epoch in range(args.nepochs):
        train_sampler.set_epoch(epoch)
        train_dset.setmode(1)
        probs = inference(epoch, train_loader, model)
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'CNN_convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        #Validation
        if args.val_lib and (epoch+1) % args.test_every == 0:
            val_sampler.set_epoch(epoch)
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err,fpr,fnr = calc_err(pred, val_dset.targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'CNN_convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #Save best model
            err = (fpr+fnr)/2.
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'CNN_checkpoint_best.pth'))



def inference(run, loader, model):#接收当前周期，加载器，模型
    model.eval()#评估模式
    probs = torch.FloatTensor(len(loader.dataset))#创建一个空的张量来存储概率
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):#周期，加载器，模型，损失函数，优化器
    model.train()#训练模式
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)#使用模型对输入数据进行前向传播，获取模型的输出
        loss = criterion(output, target)#计算预测和真实之间的损失
        optimizer.zero_grad()#清零优化器的梯度缓冲区，以便接下来的反向传播计算新的梯度
        loss.backward()#反向传播
        optimizer.step()#更新权重，减小损失
        running_loss += loss.item()*input.size(0)#累计损失
    return running_loss/len(loader.dataset)#平均损失

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]#错误率
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()#假阳性率
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()#假阴性率
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):#获取每个组内前 k 个最大值索引
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):#从每个组中获取最大值，最多获取nmax个
    out = np.empty(nmax)#创建数组out
    out[:] = np.nan#每个值设为nan
    order = np.lexsort((data, groups))#先按groups排序，再按data排序
    groups = groups[order]
    data = data[order]#使用order重新排序
    index = np.empty(len(groups), 'bool')#创建布尔类型数组
    index[-1] = True#最后一个设为True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out

class MILdataset(data.Dataset):#继承自torch.utils.data.Dataset
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)#字典
        slides = []
        for i,name in enumerate(lib['slides']):#lib['slide']:由切片路径组成的列表
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))#将每个切片打开后放入slides列表
        print('')
        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):#lib['grid']:一个列表，由许多小列表组成，每个小列表代表一张切片，其中元素是元组
            grid.extend(g)#将所有切片的所有tile坐标组成一个一维列表
            slideIDX.extend([i]*len(g))#一个一维列表，形如[0,0,0,1,1,1,1,2,2,2,3,3,3,3],每个数字代表一张切片，数字重复代表有多少个tile

        print('Number of tiles: {}'.format(len(grid)))
        self.datas = len(grid)
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = lib['mult']
        self.size = int(np.round(224*lib['mult']))
        self.level = lib['level']
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        # if self.mode == 1:
        #     return len(self.grid)
        # elif self.mode == 2:
        #     return len(self.t_data)
        return self.datas
        

if __name__ == '__main__':
    main()

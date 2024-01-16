import torch
import torch.nn as nn
import openslide
from collections import OrderedDict
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from time import time
import os
import torch_npu
from torch_npu.contrib import transfer_to_npu


# 一些全部变量
model_path = r'../output/CNN_checkpoint_best.pth'
batch_size = 256
top_k = 10    # 取 10 块 patch 作为 RNN 输入
patch_size = 224


# 读取数据
lib_test = torch.load(r'../output/lib/cnn_test_data_lib.db')

# transforms
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
trans = transforms.Compose([transforms.ToTensor(), normalize])

# 加载模型
model = models.resnet34(True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.cuda()
ch = torch.load(model_path)
model.load_state_dict(ch['state_dict'])
model.eval()
cudnn.benchmark = True


# 处理 test 部分的数据
for i, slide_name in enumerate(lib_test['slides']):
    t0 = time()
    sl = openslide.OpenSlide(slide_name)

    sl_prob = []      # 存储当前 slide 中每个 patch 对应的 prob
    imgs_batch = []   # 存储当前 batch

    # 处理每个 patch
    for j, grid in enumerate(lib_test['grid'][i]):
        imgs_batch.append(trans(sl.read_region(grid, lib_test['level'], (patch_size, patch_size)).convert('RGB')).unsqueeze(0))

        if ((j + 1) == len(lib_test['grid'][i])) or ((j + 1) % batch_size == 0):
            imgs_batch = torch.cat(imgs_batch, 0).cuda()
            output = F.softmax(model(imgs_batch), dim=1).detach()
            sl_prob.extend(output[:, 1].cpu().tolist())
            imgs_batch = []
            print('Processing {} / {} slide, patch {} / {}'.format(i + 1, len(lib_test['slides']), j + 1, len(lib_test['grid'][i])))

    # 提取前 top_k 个 patch
    sl_prob = np.array(sl_prob)
    sorted_index = np.argsort(-sl_prob)

    grid_array = np.array(lib_test['grid'][i])
    lib_test['grid'][i] = grid_array[sorted_index[:top_k]].tolist()

    print('time for slide {} is {}'.format(i, time() - t0))
torch.save(lib_test, r'../output/lib/rnn_test_data_lib.db')
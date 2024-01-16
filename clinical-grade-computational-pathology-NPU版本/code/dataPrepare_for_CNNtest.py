import torch
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os
import pandas as pd
import random

data_dir = r'../dataset_shiguanai/testing'
target_df = pd.read_csv(r'../dataset_shiguanai/targets_testing.csv')

test_data_lib = {}
test_slides_list = []   # 存储文件路径
test_targets_list = []  # 存储目标信息
test_grids_list = []    # 存储格点信息

mult = 1           # 缩放因子，1 表示不缩放
level = 0          # 使用 openslide 读取时的层级，默认表示以最高分辨率
patch_size = 224

for root, dirs, files in os.walk(data_dir):#以深度优先遍历目录，每次返回一个三元组，root当前地址，dirs当前地址下的文件夹，files当前地址下的文件
    for filename in files:#对每一个文件
        if filename[-4:] != '.svs':#判断是否是tif文件
            continue


        test_slides_list.append(os.path.join(root, filename))#将tif文件的路径放入slides列表
        test_targets_list.append(target_df[target_df['slide'] == filename].values[0][1])
            #target_df['slide']:读取slide列； target_df['slide'] == filename：找到与文件名相同的行为true，不同为false；
            #target_df[target_df['slide'] == filename]：打印true的行

            # 提取 patch 坐标
        slide = openslide.open_slide(os.path.join(root, filename))
        w, h = slide.dimensions

        cur_patch_cords = []

        for j in range(0, h, patch_size):
            for i in range(0, w, patch_size):
                cur_patch_cords.append((i,j))

        test_grids_list.append(cur_patch_cords)

test_data_lib['slides'] = test_slides_list
test_data_lib['grid'] = test_grids_list
test_data_lib['targets'] = test_targets_list
test_data_lib['mult'] = mult
test_data_lib['level'] = level
test_data_lib['patch_size'] = patch_size
torch.save(test_data_lib, r'../output/lib/cnn_test_data_lib.db')
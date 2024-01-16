import torch
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os
import pandas as pd
import random

# svs 文件所在路径
#data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(r'D:\clinical-grade-computational-pathology\dataset\training')) + os.path.sep + "."), 'dataset')
#data_dir = r'D:\clinical-grade-computational-pathology\dataset\training'
data_dir = r'../dataset_shiguanai/training'
#abspath获取绝对路径
#dirname去掉文件名，返回目录路径
#sep路径分分隔符

# target 列表
#target_df = pd.read_csv(r'D:\clinical-grade-computational-pathology\dataset\targets.csv')
target_df = pd.read_csv(r'../dataset_shiguanai/targets.csv')


# ---------------------- 相关变量的格式定义，参考 README.md ---------------------- #
# 最终保存全部数据的字典
train_data_lib = {}#保存数据的字典
train_slides_list = []   # 存储文件路径
train_targets_list = []  # 存储目标信息
train_grids_list = []    # 存储格点信息

val_data_lib = {}
val_slides_list = []   # 存储文件路径
val_targets_list = []  # 存储目标信息
val_grids_list = []    # 存储格点信息


mult = 1           # 缩放因子，1 表示不缩放
level = 0          # 使用 openslide 读取时的层级，默认表示以最高分辨率
patch_size = 224   # 切片的尺寸


# ---------------------- 开始处理数据，获取 lib ---------------------- #
for root, dirs, files in os.walk(data_dir):#以深度优先遍历目录，每次返回一个三元组，root当前地址，dirs当前地址下的文件夹，files当前地址下的文件
    for filename in files:#对每一个文件
        if filename[-4:] != '.svs':#判断是否是tif文件
            continue

        if random.randint(0, 21) < 14:#训练集
            train_slides_list.append(os.path.join(root, filename))#将tif文件的路径放入slides列表
            train_targets_list.append(target_df[target_df['slide'] == filename].values[0][1])
            #target_df['slide']:读取slide列； target_df['slide'] == filename：找到与文件名相同的行为true，不同为false；
            #target_df[target_df['slide'] == filename]：打印true的行

            # 提取 patch 坐标
            slide = openslide.open_slide(os.path.join(root, filename))
            w, h = slide.dimensions

            cur_patch_cords = []

            for j in range(0, h, patch_size):
                for i in range(0, w, patch_size):
                    cur_patch_cords.append((i,j))

            train_grids_list.append(cur_patch_cords)
        else:#验证集
            val_slides_list.append(os.path.join(root, filename))
            val_targets_list.append(target_df[target_df['slide'] == filename].values[0][1])

            # 提取 patch 坐标
            slide = openslide.open_slide(os.path.join(root, filename))
            w, h = slide.dimensions

            cur_patch_cords = []

            for j in range(0, h, patch_size):
                for i in range(0, w, patch_size):
                    cur_patch_cords.append((i,j))

            val_grids_list.append(cur_patch_cords)



train_data_lib['slides'] = train_slides_list
train_data_lib['grid'] = train_grids_list
train_data_lib['targets'] = train_targets_list
train_data_lib['mult'] = mult
train_data_lib['level'] = level
train_data_lib['patch_size'] = patch_size
#torch.save(train_data_lib, r'D:\clinical-grade-computational-pathology\output\lib\cnn_train_data_lib.db')
torch.save(train_data_lib, r'../output/lib/cnn_train_data_lib.db')

val_data_lib['slides'] = val_slides_list
val_data_lib['grid'] = val_grids_list
val_data_lib['targets'] = val_targets_list
val_data_lib['mult'] = mult
val_data_lib['level'] = level
val_data_lib['patch_size'] = patch_size
#torch.save(val_data_lib, r'D:\clinical-grade-computational-pathology\output\lib\cnn_val_data_lib.db')
torch.save(val_data_lib, r'../output/lib/cnn_val_data_lib.db')
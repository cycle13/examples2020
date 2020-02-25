# -*- coding: utf-8 -*-
"""
fun:
    img augments
ref:
    https://zhuanlan.zhihu.com/p/44673440
"""
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
###################数据预处理###########################
train_datagen = ImageDataGenerator( #实例化
    rescale = 1./255, #归一化
    rotation_range = 90,  #图片随机转动的角度
    width_shift_range = 0.2, #水平偏移的幅度0~1
    height_shift_range = 0.2, #竖直偏移的幅度0~1
    zoom_range = 0.3, #随机放大的程度0~1
    horizontal_flip = True,
    vertical_flip = True,
    shear_range= 0.2,# 剪切变换的程度0~1
    fill_mode = "nearest", #填充新出现像素的方法
)
train_generator = train_datagen.flow_from_directory(
        'data/train',#类别子文件夹的上一级文件夹
        target_size=(300, 300),
        batch_size=3,
        shuffle=False, # 按顺序读各文件夹及内容
        seed = 1, # 变形的随机种子，种子固定上面图像增强的变换方式固定
        save_to_dir="aug02",save_format="jpg",save_prefix="img",
        class_mode='categorical')

for i in range(9):
    train_generator.next()

print("train_generator", train_generator)
print("train_generator.filenames",train_generator.filenames)# 按顺序输出文件的名字
print("train_generator.class_indices", train_generator.class_indices)  #输出对应的标签文件夹

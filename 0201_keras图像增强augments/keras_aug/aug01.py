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
datagen = ImageDataGenerator(  # 实例化
    rotation_range=90,  # 图片随机转动的角度
    width_shift_range=0.2,  # 图片水平偏移的幅度
    height_shift_range=0.2,  # 图片竖直偏移的幅度
    zoom_range=0.3)  # 随机放大或缩小

img = cv2.imread('data/n02123045_275.JPEG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (300, 300))
plt.imshow(img)
plt.show()
x = np.expand_dims(img, axis=0)  # 扩展成4维
gen = datagen.flow(x, batch_size=1, save_to_dir="aug01", save_prefix="aug01", save_format="jpg")
# 显示生成的图片
plt.figure(figsize=(10, 10))
for i in range(3):
    for j in range(3):
        x_batch = next(gen)
        idx = (3 * i) + j
        plt.subplot(3, 3, idx + 1)
        plt.imshow(x_batch[0] / 255)
plt.show()

# -*- coding: utf-8 -*-
"""
nam:

fun:
    keras resnet 训练自己的数据集 图像分类
ref:
    https://blog.csdn.net/SugerOO/article/details/100031142
    040 keras图像多分类训练miniimagenet
    https://keras.io/applications/
"""
from __future__ import print_function
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,img_to_array
##################超参数###########################
print("channels location:",K.image_data_format())
model_name1 = 'model1.h5'
model_name2 = 'model2.h5'
num_classes = 3
train_data_dir = 'data/miniimagenet/train' #train文件夹下有对应为3分类名称文件夹
test_data_dir = 'data/miniimagenet/val'   #
img_rows = 224 # 227
img_cols = 224 # 227
epochs = 3
# 批量大小
batch_size = 4
# 训练样本总数
nb_train_samples = 3120 #3*1040
#all num of val samples
nb_validation_samples = 780 #3*260
##################数据导入及预处理###########################
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./ 255)
train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_rows, img_cols),
	batch_size=batch_size,
	class_mode='categorical')#多分类; 'binary')
print("train_generator.filenames",train_generator.filenames)# 按顺序输出文件的名字
print("train_generator.class_indices", train_generator.class_indices)  #输出对应的标签文件夹
validation_generator = test_datagen.flow_from_directory(
	test_data_dir,
	target_size=(img_rows, img_cols),
	batch_size=batch_size,
	class_mode='categorical')#多分类; 'binary')
print("validation_generator.filenames",validation_generator.filenames)# 按顺序输出文件的名字
print("validation_generator.class_indices", validation_generator.class_indices)  #输出对应的标签文件夹
##################训练网络模型###########################
# # model
model = ResNet50(weights=None, include_top=True,classes=num_classes)
# initiate optimizer
opt = keras.optimizers.Adam(lr=0.001) # keras.optimizers.Adam(lr=0.001) keras.optimizers.Adam(lr=0.001)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy']) #
# train
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,  # typically steps_per_epoch= dataset samples 3120 / batch_size 4
    epochs=epochs,  # finacal epoches
    validation_data=validation_generator,
    validation_steps=nb_validation_samples// batch_size, # typically validation_steps = validation dataset samples 780 / batch_size 4
    verbose = 2,
)
# save
model.save('model.h5')


# # test
model = keras.models.load_model('model.h5')
img_path = "data/miniimagenet/test/n03887697_5559.JPEG"
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)  # 为batch添加第四维
result = model.predict(img)
print("result.shape",result.shape) # (1,num_classes)
print("result[0]",result[0])
print("class_indices",np.argmax(result))
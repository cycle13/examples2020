# -*- coding: utf-8 -*-
"""
nam:
    finetune
fun:
    图片分类模型的示例
    利用ResNet50网络进行ImageNet分类
ref:
    https://blog.csdn.net/mago2015/article/details/84033104
    040 keras图像多分类训练miniimagenet
"""
from __future__ import print_function
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.layers import Input
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
##################网络搭建###########################
# create the base pre-trained model
#base_model = InceptionV3(weights='imagenet', include_top=False)
##在定制的输入tensor上构建InceptionV3
# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(img_rows, img_cols, 3))  # this assumes K.image_data_format() == 'channels_last'
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have num_classes classes
predictions = Dense(num_classes, activation='softmax')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
##################训练模型###########################
# train the model on the new data for a few epochs
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,  # typically steps_per_epoch= dataset samples 3120 / batch_size 4
    epochs=epochs,  # finacal epoches
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size, # typically validation_steps = validation dataset samples 780 / batch_size 4
    verbose = 2,
)
model.save(model_name1)
print('Model Saved.')

##################修改网络训练模型###########################
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print("layer", i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,  # typically steps_per_epoch= dataset samples 3120 / batch_size 4
    epochs=epochs,  # finacal epoches
    validation_data=validation_generator,
    validation_steps=nb_validation_samples// batch_size, # typically validation_steps = validation dataset samples 780 / batch_size 4
    verbose = 2,
)
# 保存整个模型
model.save(model_name2)
print('Model Saved.')
# 保存模型的权重
model.save_weights('model1_weights.h5')
# 保存模型的结构
json_string = model.to_json()
open('model1_to_json.json','w').write(json_string)
yaml_string = model.to_yaml()
open('model1_to_yaml.yaml','w').write(json_string)
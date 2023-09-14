#!/usr/bin/env python
# coding: utf-8
# import
import os
import numpy as np
import tensorflow as tf
from models import simple_CNN
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
import matplotlib.pyplot as plt

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
session.run(init)
model_save_path = 'trained_models/simpler_CNN.{epoch:02d}-{val_acc:.2f}.hdf5'
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')


image_size = train_x.shape[1:]
print(image_size)
num_classes = 2
batch_size = 64
num_epochs = 10

model = simple_CNN(image_size, num_classes)
#model.load_weights('trained_models/simpler_CNN.22-0.67.hdf5')



# 编译模型，categorical_crossentropy多分类选用
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 记录日志(Record Log)
csv_logger = CSVLogger('training.log')

# 保存检查点(Save checkpoint)
model_checkpoint = ModelCheckpoint(model_save_path,
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True)

model_callbacks = [model_checkpoint, csv_logger]


# 训练模型(Training Model)
history = model.fit(train_x, train_y, batch_size, num_epochs,
                    verbose=1,
                    callbacks=model_callbacks,
                    validation_split=.2,
                    shuffle=True)



# 绘制训练的准确率值(Plotting the accuracy of training)
plt.figure(1)
plt.plot(history.history['acc'])#acc最新版keras已经无法使用
plt.title('Model accuracy')#图名
plt.ylabel('Accuracy')#纵坐标名
plt.xlabel('Epoch')#横坐标名
plt.legend(['Train'], loc='upper left')#角标及其位置

# 绘制训练的损失值(Plotting the loss value of training)
plt.figure(2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


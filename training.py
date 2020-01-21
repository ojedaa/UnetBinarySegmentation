#encoding:utf-8
from unet import  *
from data import * #, solo para el último
import os
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#path to images which are prepared to train a model
train_path = "training"
image_folder = "img"
label_folder = "lbl"
valid_path =  "crossvalidation"
valid_image_folder ="img"
valid_label_folder = "lbl"
log_filepath = './log'
flag_multi_class = False
num_classes = 2
dp = data_preprocess(train_path=train_path,image_folder=image_folder,label_folder=label_folder,
                     valid_path=valid_path,valid_image_folder=valid_image_folder,valid_label_folder=valid_label_folder,
                     flag_multi_class=flag_multi_class,
                     num_classes=num_classes)

train_data = dp.trainGenerator(batch_size=2)
valid_data = dp.validLoad(batch_size=1)
model = unet(lrate=1e-4,ls=2)# WBCE

model_checkpoint1 = keras.callbacks.ModelCheckpoint('UnetWBCE_DL.hdf5', monitor='val_dice_loss',verbose=1,mode='min',save_best_only=True)#para guardar el entranamiento con menor dice loss en validation
#steps_per_epoch number= number of trainining samples / batch size of training
#validation steps number = number de validation samples  / batch size of validation
csv_logger = CSVLogger('trainingUnetWBCE.log', append=True, separator=';')#respaldo de datos de entranamiento 
history = model.fit_generator(train_data,
                              steps_per_epoch=1912,epochs=32,
                              validation_steps=207,
                              validation_data=valid_data,
                              callbacks=[model_checkpoint1,csv_logger])
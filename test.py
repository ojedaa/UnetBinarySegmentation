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


from metrics2 import dice_loss
from metrics2 import weighted_binary_crossentropy
 
model = load_model('entrenamientoguardado.hdf5',custom_objects={'weighted_binary_crossentropy':weighted_binary_crossentropy,'dice_loss': dice_loss})

#guardar Imageenes
def image_normalized(file_path):
    '''
    tif，size:512*512，gray
    :param dir_path: path to your images directory
    :return:
    '''
    img = cv2.imread(file_path)
    img_shape = img.shape
    image_size = (img_shape[1],img_shape[0])
    #img_standard = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_standard = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#verificar el formato de conversion de las imagenes de entranmiento
    img_standard = cv2.resize(img_standard, (256, 256), interpolation=cv2.INTER_CUBIC)
    #print(img_standard.shape)
    img_new = img_standard
    img_new = np.asarray([img_new])
    img_new = img_new.astype('float32')
    img_new /= 255.0
    #print(img_new.shape)
    return img_new,image_size

test_path = "testimg"#imagenes a realizar la predicción o segmentación
    # save the predict images
save_path = "predicted"
dp = data_preprocess(test_path=test_path,save_path=save_path,flag_multi_class=False,num_classes=2)

import skimage.io as io
from skimage import img_as_uint

for name in os.listdir(test_path):#
        image_path = os.path.join(test_path,name)
        x,img_size = image_normalized(image_path)
        results = model.predict(x)
        results[results<=0.93]=0.0
        results[results>0.93]=1.0
        io.imsave(os.path.join(save_path,name),img_as_uint(results[0].reshape((256,256))))
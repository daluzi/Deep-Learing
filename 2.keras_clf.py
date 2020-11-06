# coding:utf-8
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
# import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.Session(config=config)

np.set_printoptions(suppress=True)
model = ResNet50(weights='imagenet', include_top=True)

img_path = 'son.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
# print('Before:', x.shape)
x = np.expand_dims(x, axis=0)
# print('After:', x.shape)
x = preprocess_input(x)

preds = model.predict(x)
print('Pred = \n', preds)
result = decode_predictions(preds, top=5)
print(result)

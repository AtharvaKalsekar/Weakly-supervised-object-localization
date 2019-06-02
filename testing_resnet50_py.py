import keras
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16

'''img_ = cv2.imread('cifar/validation/dog/195_dog.png')
img=np.array(img_)
mean_center = img - np.mean(img, axis = None)
print(mean_center.shape)
print(mean_center)

std = np.std(img)
print(std)
standardized_img = mean_center/std
print(standardized_img.shape)
print(standardized_img)'''

#model = load_model('/home/atharvakalsekar/Downloads/cnn_11042019_colab.h5')
model = VGG16()
print(model.summary())
for layer in model.layers:
    print(layer.name)
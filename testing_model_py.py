from keras.models import load_model
import cv2
import numpy as np
from CAM import get_CAM
from B_BOX import get_bbox
'''
original_img = cv2.imread('cifar/validation/ship/62_ship.png', 1)
original_img = np.array(original_img)
original_img = np.expand_dims(original_img,axis=0)
result = model.predict(original_img)
print(result)
print(np.argmax(result))'''

model = load_model('/home/atharvakalsekar/Downloads/cnn_11042019_colab.h5')
#image_path  = 'cifar/validation/horse/318_horse.png'
image_path  = 'test_cat_2.jpeg'
image_class = 'cat'

#print(model.summary())

heatmap , original_img = get_CAM( model=model, image_path=image_path, image_class=image_class)
get_bbox(heatmap , original_img)
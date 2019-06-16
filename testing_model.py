from keras.models import load_model
import cv2
import numpy as np
from CAM import get_CAM
from B_BOX import get_bbox


model = load_model('custom_3_16062019.h5')


image_path  = 'test_airplane_1.jpeg'
image_class = 'airplane'

#print(model.summary())

heatmap , original_img = get_CAM( model=model, image_path=image_path, image_class=image_class)
get_bbox(heatmap , original_img)

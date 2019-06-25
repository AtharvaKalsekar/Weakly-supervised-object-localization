from keras.models import load_model
import cv2
import numpy as np
from bbox_generator.CAM import get_CAM
from bbox_generator.B_BOX import get_bbox


model = load_model('./models/cnn_10042019_colab.h5')


image_path  = 'test_ship_2.jpeg'
image_class = 'ship'

#print(model.summary())

heatmap , original_img = get_CAM( model=model, image_path=image_path, image_class=image_class)
get_bbox(heatmap , original_img)

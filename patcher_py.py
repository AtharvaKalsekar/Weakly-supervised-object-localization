import numpy as np
from keras.preprocessing import image
import random

def get_patched_image(one_image,image_size,patch_size):
    '''
    if image_dims=(image_hieght,image_width,n_channels)
    then image_size=image_height=image_width
    
    and
    patch_size=patch_hieght=patch_width
    '''
    mean=np.mean(one_image,axis=None)
    replacement_patch=np.full((patch_size,patch_size,3), fill_value=int(mean))
    no_of_strides=int(image_size/patch_size)
    no_of_patches=int(no_of_strides)**2
    patch_list=np.random.choice([0,1],size=(no_of_patches,),p=[0.5,0.5])
    for patch_number,patch in enumerate(patch_list):
        if patch==1:
            row=int(patch_number/no_of_strides)
            col=patch_number%no_of_strides    
            row_start=(patch_size)*row
            row_end=(row_start+patch_size)
            col_start=(patch_size)*col
            col_end=(col_start+patch_size)
            one_image[row_start:row_end,col_start:col_end,:]=replacement_patch
    return one_image

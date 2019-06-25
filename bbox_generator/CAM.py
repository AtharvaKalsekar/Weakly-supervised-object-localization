from keras.models import *#get_output_layer , load_model
import keras.backend as K
import cv2
import numpy as np


#model = load_model('cnn_08042019.h5')


def get_CAM( model, image_path, image_class):
    
    original_img = cv2.imread(image_path, 1)
    width, height, _ = original_img.shape
    original_img_copy = original_img.copy()
    mean_center = original_img - np.mean(original_img, axis = None)
    std = np.std(original_img)
    if(std == 0 or std == np.nan):
        std = 1
    original_img = mean_center/std
    classes = {
        'airplane'   : 0,
        'automobile' : 1,
        'bird'       : 2,
        'cat'        : 3,
        'deer'       : 4,
        'dog'        : 5,
        'frog'       : 6,
        'horse'      : 7,
        'ship'       : 8,
        'truck'      : 9
    }
    #Reshape to the network input shape (3, w, h).
    img = np.array([np.transpose(np.float32(original_img), (0, 1, 2))])

    #Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    print(class_weights.shape)
    #print(type(class_weights))
    #print(class_weights)

    #final_conv_layer = get_output_layer(model, "conv2d_4")
    #final_conv_layer = model.get_layer("conv2d_4")
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    #print(layer_dict)

    #final_conv_layer = layer_dict["activation_6"]
    final_conv_layer = layer_dict["activation_36"]
    #print(final_conv_layer.shape)

    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    for key,value in classes.items() :
        if value == np.argmax(predictions):
            print(key)
            #image_class = key
    '''
    print(conv_outputs.shape)
    print(type(conv_outputs))
    print(conv_outputs)

    print(predictions.shape)
    print(type(predictions))
    print(predictions)
    '''
    conv_outputs = conv_outputs[0, :, :, :]
    conv_outputs = np.array([np.transpose(np.float32(conv_outputs), (2, 0, 1))])
    conv_outputs = np.squeeze(conv_outputs, axis=None)
    #print(conv_outputs.shape)
    #print(type(conv_outputs))
    #print(conv_outputs.shape)
    #print(conv_outputs)
    #print(conv_outputs[2, :, :].shape)
    #print(class_weights.shape)
    #print(class_weights)
    #print(class_weights[:,1])
    #print(class_weights[:,1].shape)
    
    #Create the class activation map.
    
    
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
    for i, w in enumerate(class_weights[ : , classes[image_class] ]):
        cam += w * conv_outputs[i, :, :]
        
    print("predictions", predictions)
    
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    #cam_gray = cv2.cvtColor(cam , cv2.COLOR_BGR2GRAY)
    #heatmap = cv2.applyColorMap(cam_gray, cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    #print(heatmap.shape)
    #print(heatmap)
    heatmap[np.where(cam < 0.4)] = 0
    cv2.imshow('heat map',heatmap)
    #img = heatmap*0.5 + original_img
    cv2.imshow('orignal image',original_img_copy)
    cv2.waitKey()
    return heatmap , original_img_copy
    #cv2.imwrite(output_path, img)

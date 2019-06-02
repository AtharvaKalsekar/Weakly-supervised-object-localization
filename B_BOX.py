import cv2
import numpy as np
def get_bbox(image , original_image):
    img_copy=image.copy()
    gray_img = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)
    ret,thresh_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    thresh_img = cv2.erode( thresh_img , kernel , 1)
    thresh_img = cv2.dilate( thresh_img , kernel , 1)
    cv2.imshow("threshold",thresh_img)
    edged = cv2.Canny( thresh_img, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_vals = []
    y_vals = []
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        x_vals.append(x)
        y_vals.append(y)
        #cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),1)
    x_top_left = min(x_vals)
    y_top_left = min(y_vals)
    x_bottom_right = max(x_vals)
    y_bottom_right = max(y_vals)
    w = x_bottom_right-x_top_left
    h = y_bottom_right-y_top_left
    cv2.rectangle(original_image,(x_top_left,y_top_left),(x_top_left+w,y_top_left+h),(0,255,0),1)
    cv2.imshow("b_boxes",original_image)
    cv2.waitKey()  
    cv2.destroyAllWindows()
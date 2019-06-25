import cv2
import numpy as np
def get_bbox(image , original_image):
    img_copy=image.copy()
    gray_img = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray",gray_img)
    #cv2.waitKey()
    ret,thresh_img = cv2.threshold(gray_img,150,255,cv2.THRESH_BINARY)
    #cv2.imshow("threshold1",thresh_img)
    #cv2.waitKey()
    kernel = np.ones((3,3),np.uint8)
    thresh_img = cv2.erode( thresh_img , kernel , 1)
    thresh_img = cv2.dilate( thresh_img , kernel , 1)
    #cv2.imshow("threshold",thresh_img)
    #cv2.waitKey()
    edged = cv2.Canny( thresh_img, 30, 200)
    #cv2.imshow("canny",edged)
    #cv2.waitKey()
    _, contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    x_top_vals = []
    y_top_vals = []
    x_bot_vals = []
    y_bot_vals = []
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        x_top_vals.append(x)
        y_top_vals.append(y)
        x_bot_vals.append(x+w)
        y_bot_vals.append(y+h)
        
        #cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),1)
        #print(x,y,x+w,y+h)
        #cv2.imshow("test",img_copy)
        #cv2.waitKey()
    #print(len(x_vals),len(y_vals))
    #print(x_vals)
    #print(y_vals)
    x_top_left = min(x_top_vals)
    y_top_left = min(y_top_vals)
    x_bottom_right = max(x_bot_vals)
    y_bottom_right = max(y_bot_vals)
    
    cv2.rectangle(original_image,(x_top_left,y_top_left),(x_bottom_right,y_bottom_right),(0,255,0),2)
    cv2.imshow("b_boxes",original_image)
    cv2.waitKey()  
    cv2.destroyAllWindows()

import cv2
import numpy as np
import imutils
import math


def nothing(x):
    pass


#load the image, clone it for output, and then convert it to grayscale
img = cv2.imread("test1.png")

clone = img.copy()
gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

#create window
cv2.namedWindow("Treshed")

#create trackbars for threshold for color change
cv2.createTrackbar("Threshold","Treshed", 0, 255, nothing)


while(1):
    clone = img.copy()
    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

    #get current position of four tracker
    r = cv2.getTrackbarPos('Threshold', 'Treshed')

    ret, gray_threshold = cv2.threshold(gray, r, 255,cv2.THRESH_BINARY)
    bilateral_filetered_image = cv2.bilateralFilter(gray_threshold,5,175,175)

    edge_detected_image = cv2.Canny(bilateral_filetered_image,75,200)

    contours, hierarchy=cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    r=0
    rad=0
    radius= []
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if((len(approx) > 8 ) & (area>30)): #10000> area>30
            
            contour_list.append(contour)
            rad = math.sqrt(area/math.pi)
            radius.append(round(rad, 2))
            
        
        for c in contour_list:
            (c_x, c_y), c_r = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            cX = int(M['m10']/M['m00'])
            cY = int(M['m01']/M['m00'])
            if c_r>r:
                if (cX == c_x) | (cY == c_y):
                    
                    c_r = str(c_r)
                    cv2.putText(clone, c_r, (cX - 40, cY - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                    
                
                    

                    
            cv2.circle(clone, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(clone, "center", (cX - 20, cY - 20),
	    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
            
        

    cv2.drawContours(clone, contour_list, -1, (255,0,0),2)
    
    
    #there is an outer boundary and inner boundary for each edge, so countours double
    print('Number of circles:{}'.format(int(len(contour_list)/2)))
    print('radius:{}'.format(radius[::2]))

    #conts = imutils.grab_contours(contour_list)
    
        

    #Displaying the results     
    cv2.imshow('Objects Detected', clone)
    #cv2.imshow("Treshed", gray_threshed)

    # ESC to break
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    
    





cv2.destoryAllWindow()

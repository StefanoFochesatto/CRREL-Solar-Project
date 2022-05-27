from __future__ import print_function
import cv2 as cv
import argparse
import os
import numpy as np
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
path = r'C:\Users\Amanda Barker\Desktop\Stefano\CRREL-Solar-Project\TestTimeLapse' ##Supply Full Path to TimeLapse directory
os.chdir(path) # We have to set the path everytime since cv2 can't handle relative paths without it.
TestSetting = 2
if (TestSetting == 1):
    ## Glare During The DAY
    img = cv.imread('TestImage3.png')
    pts = np.array([(1245, 151), (1535, 138), (1705, 780), (1343, 777)])
elif (TestSetting == 2):
    ## Snow During the Day
    img = cv.imread('TestImage4.png')
    pts = np.array([(631, 156), (913, 154), (924, 788), (572, 797)])
elif (TestSetting == 3):
    ## Snow During the Day with weird coating
    img = cv.imread('TestImage4.png')
    pts = np.array([(944, 159), (1224, 156), (1301, 779), (961, 785)])
elif (TestSetting == 4):
    ## No Snow or glare during the day
    img = cv.imread('TestImage2.png')
    pts = np.array([(987, 145), (1256, 146), (1316, 744), (991, 743)])
elif (TestSetting == 5):
    ## Snow During the Night 
    img = cv.imread('TestImage.png')
    pts = np.array([(986, 145), (1258, 145), (1321, 748), (987, 745)])
elif (TestSetting == 6):
    ## No Snow During the Night
    img = cv.imread('TestImage5.png')
    pts = np.array([(311, 164), (604, 164), (533, 776), (142, 803)])
elif (TestSetting == 7):
    ## Almost Fully Covered Panel in the Day from the old Array
    img = cv.imread('TestImage6.png')
    pts = np.array([(242, 230), (401, 230), (400, 312), (234, 312)])
elif (TestSetting == 8):
    ## Night Time With a tiny bit of snow
    img = cv.imread('TestImage.png')
    pts = np.array([(692, 144), (959, 145), (953, 746), (623, 748)])
elif (TestSetting == 9):
    ## Cropped 8 to have a larger section of snow
    img = cv.imread('TestImage.png')
    pts = np.array([(364, 268), (643, 274), (580, 746), (245, 747)])




## Cropping the bounding rectangle
rect = cv.boundingRect(pts)
x,y,w,h = rect
cropped = img[y:y+h, x:x+w].copy()

## Converting the crop to gray scale
#sampleintensity = img[0:60,870:890]
#sampleintensity = cv.cvtColor(sampleintensity, cv.COLOR_BGR2GRAY)
#sampleintensity = cv.mean(sampleintensity)[0]


#cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

## Testing Contrast 
## cropped = cv2.equalizeHist(cropped)
## This increases the contrast of the image. Use this and TestSetting == 2 to 
## see an example for when We Have SNOW and SUN  

## Generating the Mask
pts = pts - pts.min(axis=0)
mask = np.zeros(cropped.shape[:2], np.uint8)
cv.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)

## Removing everything outside the Mask with bitwise operation
dst = cv.bitwise_and(cropped, cropped, mask=mask)
bg = np.ones_like(cropped, np.uint8)*0
cv.bitwise_not(bg,bg, mask=mask)
dst = bg+dst
### THIS IS VERY IMPORTANT IT DEALS WITH THE LINES AND DOTS ON THE PANELS AND IS The Main Preprocessing ###
dst = cv.medianBlur(dst, ksize=7)
dst = cv.bilateralFilter(dst,9,75,75)
#dst = cv2.GaussianBlur(dst,(5,5),20)
#M = np.ones(dst.shape,  dtype="uint8") * int(sampleintensity/5)
#dst = cv.subtract(dst, M)
#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#dst = clahe.apply(dst)


while True:
    frame_HSV = cv.cvtColor(dst, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    
    cv.imshow(window_capture_name, dst)
    cv.imshow(window_detection_name, frame_threshold)
    
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
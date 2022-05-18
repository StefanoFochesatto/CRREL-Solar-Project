#!/usr/bin/env python3

#################### Dependencies #################### 
# For Data Management
import numpy as np
import pandas as pd
# For Directory Management
import os 
# For Image Processing
from PIL import Image
import cv2  
from matplotlib import pyplot as plt
import copy

# For Optical Character Recognition
import easyocr
from scipy.fft import dstn




#################### Global Variables #################### 
## Supply Full Path to TimeLapse directory
path = "/Users/stefanofochesatto/Desktop/CRREL-Solar-Project/TestTimeLapse" 
# path = r"C:\Users\Amanda Barker\Desktop\Stefano\CRREL-Solar-Project\TestTimeLapse"
reader = easyocr.Reader(['en']) # Initializing OCR Engine
MaskCoordinates = [] # User defined Mask Coordinated global variable. 

## Initilizing Python lists to store our data
TimeStamp = []
SnowCover = []




#################### Helper Functions ####################
# Function for processing the data from an image frame
# takes the openCV image object for each frame as 'img' and the user define points
# for each solar panel as 'Mask'(python list)
def ImageProcess(img):

    ####### Extracting Timestamp Data with easyOCR ####### 
    TimeStampCrop = img[0:60,0:860]
    ## Preprocessing ##
    # Resizing
    TimeStampCrop = cv2.resize(TimeStampCrop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Adding Border
    TimeStampCrop = cv2.copyMakeBorder(TimeStampCrop, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # Converting to GrayScale
    TimeStampCrop = cv2.cvtColor(TimeStampCrop, cv2.COLOR_BGR2GRAY)
    # Dilation and Erosion
    kernel = np.ones((1, 1), np.uint8)
    TimeStampCrop = cv2.dilate(TimeStampCrop, kernel, iterations=10)
    TimeStampCrop = cv2.erode(TimeStampCrop, kernel, iterations=1)
    # Applying Blur
    TimeStampCrop = cv2.threshold(cv2.medianBlur(TimeStampCrop, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    ### Running OCR engine on image ###
    print('Running OCR')
    TimeStampText = reader.readtext(TimeStampCrop, allowlist = '0123456789:-PTSA ')
    TimeStampText = [val[1] for val in TimeStampText]
    print(TimeStampText)


    ### Updating Global Data Lists ###
    global TimeStamp
    TimeStamp.append(TimeStampText)



    ####### Extracting SnowCover Data with User Defined Masks #######
    ### Cropping the Panels ###
    for i in range(0,len(MaskCoordinates),4):
        pts = np.array([MaskCoordinates[i],MaskCoordinates[i+1],MaskCoordinates[i+2],MaskCoordinates[i+3]])
        
        ## Cropping the bounding rectangle
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        cropped = img[y:y+h, x:x+w].copy()

        ## Converting the crop to gray scale
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)


        ## Generating the Mask
        pts = pts - pts.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## Removing everything outside the Mask with bitwise operation
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)

        ############## Use Mask to threshold and pull the snow cover data
        AdaptiveMeanThresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 199, 5)
        
        AdaptiveGaussThresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 199, 5)

        # Otsu's thresholding after Gaussian filtering
        dst = cv2.GaussianBlur(dst,(5,5),0)
        cv2.imwrite('testGaussblur.png', dst)
        linek = np.zeros((11,11),dtype=np.uint8)
        linek[5,...]=1

        x=cv2.morphologyEx(dst, cv2.MORPH_OPEN, linek ,iterations=1)
        dst-=x

        ret3,th3 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



        cv2.imwrite('testAdaptiveMean.png', AdaptiveMeanThresh)
        cv2.imwrite('testAdaptiveGauss.png', AdaptiveGaussThresh)
        cv2.imwrite('testGaussblurGlobalOtsu.png', ret3)
        cv2.imwrite('testGaussblurGlobalOtsu1.png', th3) ## CONTINUE TESTING THIS METHOD





       
    ## # Applying the threshold
    ## threshImage = copy.deepcopy(img)
    ## ret, thresh = cv2.threshold(threshImage,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ## 
    ## # Creating the masks
    ## mask_fore = copy.deepcopy(img)
    ## mask_fore = np.where((img).astype(np.uint8) > ret, 0, 255)  
    ## image_fore = Image.fromarray((mask_fore).astype(np.uint8))
    ## os.chdir(str(path+'/Foreground_Masks'))
    ## image_fore.save(str(i),"JPEG") 
    # importing the module


# Callback function for getting coordinates of the solar panel mask from 
# the click event. This will update the global MaskCoordinates Variable on Click
def click_event(event, x, y, flags, params):
	global MaskCoordinates
	
    # Listening for Left Click
	if event == cv2.EVENT_LBUTTONDOWN:

        # Print and Append Coordinates
		print(x, ' ', y)
		MaskCoordinates.append((x, y))

        # Display Coordinates 
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,'+', (x,y), font, .2, (255,164,0))
		cv2.imshow('image', img)

	# Listening for Right Click	
	if event==cv2.EVENT_RBUTTONDOWN:

        # Print and Append Coordinates
		print(x, ' ', y)
		MaskCoordinates.append((x, y))


		# Display Coordinates 
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,'+', (x,y), font, .2, (255,164,0))
		cv2.imshow('image', img)


# Driver function for MaskCoordinates array generation
# This function will take the initial frame of ever timelapse video
def generateMasks(img):
	# Displaying the image
	cv2.imshow('image', img)

    # Running MouseClick Callback
	cv2.setMouseCallback('image', click_event)

	# Exiting when a key is pressed
	cv2.waitKey(0)

	# Closing the Image Window
	cv2.destroyAllWindows()





#################### Main Script ####################
if __name__ == "__main__":
    #### Directory Management ####
    os.chdir(path) #Change the working directory to the TimeLapse directory
    TimeLapseList = [] #Pull the current list of files in TimeLapse directory
    for file in os.listdir(path):
        if file.endswith(".mp4"): 
            TimeLapseList.append(file)


    for i in TimeLapseList:
        os.chdir(path) # We have to set the path everytime since cv2 can't handle relative paths without it.
        currentVideo = cv2.VideoCapture(i, cv2.IMREAD_GRAYSCALE) # Reading in the current TimeLapse Video
        success, img = currentVideo.read()
        fno = 0
        #Get Points to generate mask for each recording. 
        sample_rate = 1
        generateMasks(img)
        while success:
            if fno % sample_rate == 0:
                ImageProcess(img)
            # read next frame

            print('Next Frame')
            success, img = currentVideo.read()
    
        ## Resetting mask coordinates list
        MaskCoordinates = []


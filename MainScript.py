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
from datetime import datetime 

# For Timing Function 
import time





#################### Global Variables #################### 
## Supply Full Path to TimeLapse directory
# path = "/Users/stefanofochesatto/Desktop/CRREL-Solar-Project/TestTimeLapse" 
path = r"C:\Users\Amanda Barker\Desktop\Stefano\CRREL-Solar-Project\TestTimeLapse"
reader = easyocr.Reader(['en']) # Initializing OCR Engine
MaskCoordinates = [] # User defined Mask Coordinated global variable. 
PanelDictionary = {
    '2':[10, 8, 2, 5],
    '1':[1, 4, 7, 11], 
    '3':[9, 3, 6, 12]
}




## Initilizing Python lists to store our data
TimeStampBuffer = []
TimeStamp = []
SnowCover = []
# Testing video
imgArray = []
prevThresh = 100







#################### Helper Functions ####################
# Function for processing the data from an image frame
# takes the openCV image object for each frame as 'img' and the user define points
# for each solar panel as 'Mask'(python list)

## Perfoming Otsu Global Thresholding inside of Masked area. 
## Count number of zeros outside of mask, remove them from frequency histogram
## Compute Threshold normally. 
def FindThreshold(dst, mask):
    NumberOfZerosinBorder = np.where(mask==0)[0].size
    hist = cv2.calcHist([dst],[0],None,[256],[0,256])
    hist[0] = hist[0] - NumberOfZerosinBorder
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    return thresh



def OCRPreProcess(img):
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Adding Border
        img = cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # Converting to GrayScale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Dilation and Erosion
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=10)
        img = cv2.erode(img, kernel, iterations=1)
        # Applying Blur
        img = cv2.threshold(cv2.medianBlur(img, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return img
















def TimeStampCollation(img):
    global TimeStamp
    global TimeStampBuffer
    ## Add PSA Classification
    if (len(TimeStampBuffer) < 2):
        ####### Extracting Timestamp Data with easyOCR ####### 
        TimeStampCrop = img[0:60,0:620]
        ## Preprocessing ##
        TimeStampCrop = OCRPreProcess(TimeStampCrop)
        ### Running OCR engine on image ###
        print('Running OCR')
        TimingOCR = time.time()
        TimeStampText = reader.readtext(TimeStampCrop, allowlist = '0123456789:- ', width_ths=1)
        TimeStampText = [val[1] for val in TimeStampText]
        TimingOCRELAPSED = time.time() - TimingOCR
        print('Frame TimeStamp: ',TimeStampText)
        print('OCR Processing took: ',str(TimingOCRELAPSED))
        
        ### Updating Global Data Lists ###
        CurrentTimeStamp = datetime.strptime(TimeStampText[0], "%Y-%m-%d %H:%M:%S")
        print(CurrentTimeStamp)
        TimeStamp.append(CurrentTimeStamp)
        TimeStampBuffer.append(CurrentTimeStamp)
    else:    
        TimestampDifference =  TimeStamp[1] - TimeStamp[0]
        print(TimeStamp[-1] + TimestampDifference)
        TimeStamp.append(TimeStamp[-1] + TimestampDifference)
        


def ExtractPanelID(img):
    PanelID = img[0:60,835:860]
    PanelID = OCRPreProcess(PanelID)
    PanelIDText = reader.readtext(PanelID, allowlist = '123PTSA ', width_ths=1)
    PanelIDText = [val[1] for val in PanelIDText]
    return PanelIDText[0]























def ImageProcess(img, PanelID):
    TimeStampCollation(img)

   # ############ REFACTOR THIS ##################################################
   # #########################################################################################################
   # global TimeStamp
   # global TimeStampBuffer
   # ## Add PSA Classification
   # if (len(TimeStampBuffer) < 2):
   #     ####### Extracting Timestamp Data with easyOCR ####### 
   #     TimeStampCrop = img[0:60,0:620]
   #     PanelID = img[0:60,620:860]
   #     ## Preprocessing ##
   #     # Resizing
   #     TimeStampCrop = cv2.resize(TimeStampCrop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
   #     # Adding Border
   #     TimeStampCrop = cv2.copyMakeBorder(TimeStampCrop, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=[0, 0, 0])
   #     # Converting to GrayScale
   #     TimeStampCrop = cv2.cvtColor(TimeStampCrop, cv2.COLOR_BGR2GRAY)
   #     # Dilation and Erosion
   #     kernel = np.ones((1, 1), np.uint8)
   #     TimeStampCrop = cv2.dilate(TimeStampCrop, kernel, iterations=10)
   #     TimeStampCrop = cv2.erode(TimeStampCrop, kernel, iterations=1)
   #     # Applying Blur
   #     TimeStampCrop = cv2.threshold(cv2.medianBlur(TimeStampCrop, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
   #
   #     ### Running OCR engine on image ###
   #     print('Running OCR')
   #     TimingOCR = time.time()
   #     TimeStampText = reader.readtext(TimeStampCrop, allowlist = '0123456789:- ', width_ths=1)
   #     TimeStampText = [val[1] for val in TimeStampText]
   #     TimingOCRELAPSED = time.time() - TimingOCR
   #     print('Frame TimeStamp: ',TimeStampText)
   #     print('OCR Processing took: ',str(TimingOCRELAPSED))
   #     
   #     ### Updating Global Data Lists ###
   #     CurrentTimeStamp = datetime.strptime(TimeStampText[0][0:19], "%Y-%m-%d %H:%M:%S")
   #     print(CurrentTimeStamp)
   #     TimeStamp.append(CurrentTimeStamp)
   #     TimeStampBuffer.append(CurrentTimeStamp)
   # else:    
   #     TimestampDifference =  TimeStamp[1] - TimeStamp[0]
   #     print(TimeStamp[-1] + TimestampDifference)
   #     TimeStamp.append(TimeStamp[-1] + TimestampDifference)
   # ############ REFACTOR THIS ##################################################
   # #########################################################################################################



    SnowCoverFrame = [PanelID]
    ####### Extracting SnowCover Data with User Defined Masks #######
    ### Cropping the Panels ###
    print('Running Image Processing')
    TimingImageProcess = time.time()



    sampleintensity = img[0:60,870:890]
    sampleintensity = cv2.cvtColor(sampleintensity, cv2.COLOR_BGR2GRAY)
    sampleintensity = cv2.mean(sampleintensity)[0]
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
        ## Setting background to gray (Helps with Nightime Thresholding)
        bg = np.ones_like(cropped, np.uint8)*0
        cv2.bitwise_not(bg,bg, mask=mask)
        dst = bg+dst
        ### THIS IS VERY IMPORTANT IT DEALS WITH THE LINES AND DOTS ON THE PANELS AND IS The Main Preprocessing ###
        dst = cv2.medianBlur(dst, ksize=7)
        dst = cv2.bilateralFilter(dst,9,75,75)
        #dst = cv2.GaussianBlur(dst,(5,5),20)
        M = np.ones(dst.shape,  dtype="uint8") * int(sampleintensity/5)
        dst = cv2.subtract(dst, M)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        dst = clahe.apply(dst)




        ## Minimal threshold
        global prevThresh
        Thresh = FindThreshold(dst, mask)
        print(Thresh)
        print(prevThresh)
        if (Thresh <= 30):
            Thresh = prevThresh

        ### Applying Threshold
        ret3,th3 = cv2.threshold(dst,Thresh,255,cv2.THRESH_BINARY)
        ## Pixel math for SnowCover
        SnowCoverPercentage = 1 - ((len(np.extract(mask > 0, th3)) - np.count_nonzero(np.extract(mask > 0, th3)))/len(np.extract(mask > 0, th3)))
        SnowCoverFrame.append((PanelDictionary[PanelID][int(i/4)], SnowCoverPercentage))



        testFrame = cv2.hconcat([cropped, th3, dst])
        global imgArray
        imgArray.append(testFrame)




    global SnowCover
    SnowCover.append(SnowCoverFrame)
    TimingImageProcessELAPSED = time.time() - TimingImageProcess
    print('Image Processing took: ', str(TimingImageProcessELAPSED))
        


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
        generateMasks(img)
        fno = 0 
        sample_rate = 1
        PanelID = ExtractPanelID(img)
        while success:
            if fno % sample_rate == 0:
                ImageProcess(img, PanelID)
            # read next frame

            print('Next Frame')
            success, img = currentVideo.read()
    
        ## Resetting mask coordinates list
        MaskCoordinates = []
        TimeStampBuffer = []

    ############### Data Preparation ###############
    data = {'TimeStamp':TimeStamp, 'SnowCover':SnowCover}
    df = pd.DataFrame(data)
    df.to_csv('test.csv', index=False)
    height, width = imgArray[0].shape
    size = (width,height)
    out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, size, 0)
    for i in range(len(imgArray)):
        out.write(imgArray[i])
    out.release()

    



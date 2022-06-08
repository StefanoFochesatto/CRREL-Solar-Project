#!/usr/bin/env python3

#################### Dependencies #################### 
# For Data Management
from http.server import ThreadingHTTPServer
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
from datetime import datetime 
from datetime import timedelta

# For Timing Function 
import time

# For Sunrise/Sunset Flag
from suntime import Sun, SunTimeException

# For threshold comparison and snowcover calculations
from skimage.metrics import normalized_root_mse
import math


latitude = 64.95156082750202
longitude = -147.6210085622378
sun = Sun(latitude, longitude)


#################### Global Variables #################### 
## Supply Full Path to TimeLapse directory
# path = "/Users/stefanofochesatto/Desktop/CRREL-Solar-Project/TestTimeLapse" 
path = r"C:\Users\AKROStudent2\Desktop\CRREL-Solar-Project\TestTimeLapse"
reader = easyocr.Reader(['en']) # Initializing OCR Engine
PanelDictionary = { 
    '2':[10, 8, 2, 5],
    '1':[1, 4, 7, 11], 
    '3':[9, 3, 6, 12],
    'A':['all', 'all', 'all', 'all']
}
GenerateVid = True



## Initilizing Python lists to store our data
MaskCoordinates = [] # User defined Mask Coordinated global variable. 
TimeStampBuffer = []
TimeStamp = []
SnowCover = []
prevThresh = []


# Testing video
imgArray = []
SunFlag = []







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


def SunriseFlag(CurrentTime):
    SunRiseTime = sun.get_local_sunrise_time(CurrentTime)
    SunSetTime = sun.get_local_sunset_time(CurrentTime)

    CurrentTime = CurrentTime.replace(tzinfo=None)
    SunRiseTime = SunRiseTime.replace(tzinfo=None)
    SunSetTime = SunSetTime.replace(tzinfo=None)
    FlagWindow = 2
    SunRiseStart = SunRiseTime - timedelta(hours = FlagWindow)
    SunRiseEnd = SunRiseTime + timedelta(hours = FlagWindow)
    
    SunSetStart = SunSetTime - timedelta(hours = FlagWindow)
    SunSetEnd = SunSetTime + timedelta(hours = FlagWindow)
    
    if (CurrentTime >= SunRiseStart and CurrentTime <= SunRiseEnd):
        SunFlag.append(1)
    elif (CurrentTime >= SunSetStart and CurrentTime <= SunSetEnd):
        SunFlag.append(1)
    else:
        SunFlag.append(0)


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


def GetPrevThresh(PanelNumber,kernel):
    if (kernel == 0):
        ThresholdsByPanel = prevThresh
        ThreshList = [i[PanelNumber] for i in ThresholdsByPanel]
        return ThreshList
    else:
        ThresholdsByPanel = prevThresh[-kernel:-1]
        ThreshList = [i[PanelNumber] for i in ThresholdsByPanel]
        return ThreshList


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def ImageProcess(img, PanelID):
    TimeStampCollation(img)
    SunriseFlag(TimeStamp[-1])
    SnowCoverFrame = [PanelID]
    prevThreshFrame = []
    

    if (GenerateVid):
        PanelVidArray = []

    ####### Extracting SnowCover Data with User Defined Masks #######
    ### Cropping the Panels ###
    print('Running Image Processing')
    TimingImageProcess = time.time()
    for i in range(0,len(MaskCoordinates),4):
        pts = np.array([MaskCoordinates[i],MaskCoordinates[i+1],MaskCoordinates[i+2],MaskCoordinates[i+3]])
        
        ## Cropping the bounding rectangle
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        cropped = img[y:y+h, x:x+w].copy()

        ## Generating the Mask
        pts = pts - pts.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## Removing everything outside the Mask with bitwise operation
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        ## Setting background to black (is)
        bg = np.ones_like(cropped, np.uint8)*0
        cv2.bitwise_not(bg,bg, mask=mask)
        dst = bg+dst

        ### THIS IS VERY IMPORTANT!!!!! It is The Main Preprocessing ###
        dst = cv2.medianBlur(dst, ksize=9)
        dst = cv2.bilateralFilter(dst,9,75,75)


        # lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # lab[...,0] = clahe.apply(lab[...,0])
        # dst = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        #dst = cv2.GaussianBlur(dst,(5,5),20)
        #print(int(sampleintensity/5))
        #M = np.ones(dst.shape,  dtype="uint8") * int(sampleintensity/5)
        #dst = cv2.add(dst, M)
        #print(int(sampleintensity/5))

        ## Computing HSV Threshold (Color Threshold) ##
        frame_HSV = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        HSV_Threshold = cv2.inRange(frame_HSV, (0, 0, 90), (180, 15, 255)) ## HSV Range for White/Greyish


        ## Computing Otsu Threshold (Intensity Threshold) ##
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        Thresh = FindThreshold(dst, mask)
        prevThreshFrame.append(Thresh)
        print(Thresh)

        ## Gaussian Kernel Smoothing, on Otsu thresholds (kinda)##
        if(len(prevThresh) >= 6):
            ThreshList = GetPrevThresh(int(i/4), 6) 
            CurrentMean = np.mean(ThreshList)
            CurrentStd = np.std(ThreshList)

            if (Thresh < CurrentMean - .5*CurrentStd or Thresh > CurrentMean + .5*CurrentStd):
                Thresh = CurrentMean
                
        else:
            ThreshList = GetPrevThresh(int(i/4), 0) 
            CurrentMean = np.mean(ThreshList)
            CurrentStd = np.std(ThreshList)

            if (Thresh < CurrentMean - .5*CurrentStd or Thresh > CurrentMean + .5*CurrentStd):
                Thresh = CurrentMean

        Otsu_Threshold = cv2.threshold(dst,Thresh,255,cv2.THRESH_BINARY)[1]
        
        ## Comparing Both SnowCover Measurements ##
        score = normalized_root_mse(Otsu_Threshold, HSV_Threshold)
        global SnowCover
        if (score > .90 and len(SnowCover) > 0):
            HSVSnow = math.floor((1 - ((len(np.extract(mask > 0, HSV_Threshold)) - np.count_nonzero(np.extract(mask > 0, HSV_Threshold)))/len(np.extract(mask > 0, HSV_Threshold))))*100)/100
            OTSUSnow = math.floor((1 - ((len(np.extract(mask > 0, Otsu_Threshold)) - np.count_nonzero(np.extract(mask > 0, Otsu_Threshold)))/len(np.extract(mask > 0, Otsu_Threshold))))*100)/100
            if (len(SnowCover) < 5):
                SnowCoverKernel = SnowCover
            else:        
                SnowCoverKernel = SnowCover[-5:-1]

            SnowCoverKernelList1 = [j[int(i/4)+1] for j in SnowCoverKernel]
            SnowCoverKernelList = [j[1] for j in SnowCoverKernelList1]
            meanSnowCover = np.mean(SnowCoverKernelList)
            if (abs(HSVSnow - meanSnowCover) > abs(OTSUSnow - meanSnowCover)):
                Final = Otsu_Threshold
            else:
                Final = HSV_Threshold

        elif (len(SnowCover) == 0):
            Final = HSV_Threshold
        else:
            Final = Otsu_Threshold

        SnowCoverPercentage = math.floor((1 - ((len(np.extract(mask > 0, Final)) - np.count_nonzero(np.extract(mask > 0, Final)))/len(np.extract(mask > 0, Final))))*100)/100
        SnowCoverFrame.append((PanelDictionary[PanelID][int(i/4)], SnowCoverPercentage))

        if (GenerateVid):
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            PanelVidArray.append(cv2.hconcat([cropped, HSV_Threshold, Otsu_Threshold, Final, dst]))
    
    if (GenerateVid):
        global imgArray 
        imgArray.append(vconcat_resize_min(PanelVidArray))


    prevThresh.append(prevThreshFrame)
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


    if (GenerateVid):
        height, width = imgArray[0].shape
        size = (width,height)
        out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, size, 0)
        for i in range(len(imgArray)):
            out.write(imgArray[i])
        out.release()

    



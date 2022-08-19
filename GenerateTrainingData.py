#!/usr/bin/env python3

#################### Dependencies #################### 
# For Data Management
import numpy as np
import pandas as pd
import csv
# For Directory Management
import os 
# For Image Processing
from PIL import Image
import cv2  
import copy

# For Optical Character Recognition
import easyocr
from datetime import datetime 
from datetime import timedelta


# For threshold comparison and snowcover calculations
from skimage.metrics import normalized_root_mse
import math

# GUI and Terminal Elements
from progress.bar import Bar
from progress.spinner import MoonSpinner
import tkinter
from tkinter import filedialog
from tkinter import messagebox

root = tkinter.Tk()
root.withdraw() #use to hide tkinter window




#################### Global Variables #################### 
reader = easyocr.Reader(['en']) # Initializing OCR Engine
PanelDictionary = { 
    '2':[10, 8, 2, 5],
    '1':[1, 4, 7, 11], 
    '3':[9, 3, 6, 12],
    'A':['all', 'all', 'all', 'all']
}
GenerateVid = True



## Initializing Python lists to store our data
MaskCoordinates = [] # User defined Mask Coordinated global variable. 
TimeStampBuffer = []
SnowCover = []
SnowCoverBuffer = []
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


def search_for_file_path ():
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please select a directory')
    if len(tempdir) > 0:
        print ("You chose: %s" % tempdir)
    return tempdir


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
    global TimeStampBuffer
    ## Add PSA Classification
    if (len(TimeStampBuffer) < 2):
        ####### Extracting Timestamp Data with easyOCR ####### 
        TimeStampCrop = img[0:60,0:620]
        ## Preprocessing ##
        TimeStampCrop = OCRPreProcess(TimeStampCrop)
        ### Running OCR engine on image ###
        TimeStampText = reader.readtext(TimeStampCrop, allowlist = '0123456789:- ', width_ths=1)
        TimeStampText = [val[1] for val in TimeStampText]
        
        ### Updating Global Data Lists ###
        CurrentTimeStamp = datetime.strptime(TimeStampText[0], "%Y-%m-%d %H:%M:%S")
        CurrentTimeStamp = CurrentTimeStamp.replace(second=00)
        print(CurrentTimeStamp)
        TimeStampBuffer.append(CurrentTimeStamp)
        return CurrentTimeStamp
    else:    
        TimestampDifference =  TimeStampBuffer[1] - TimeStampBuffer[0]
        TimeStampBuffer.append(TimeStampBuffer[-1] + TimestampDifference)
        return TimeStampBuffer[-1]


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
    CurrentTimeStamp = TimeStampCollation(img)
    SnowCoverFrame = [CurrentTimeStamp]
    prevThreshFrame = []
    

    if (GenerateVid):
        PanelVidArray = []

    ####### Extracting SnowCover Data with User Defined Masks #######
    ### Extracting Sample Intensity ###
    sampleintensity = img[0:60,870:890].copy()
    sampleintensity = cv2.cvtColor(sampleintensity, cv2.COLOR_BGR2GRAY)
    sampleintensity = cv2.mean(sampleintensity)[0]

    for i in range(0,len(MaskCoordinates),4):    
        ######################## Pre Processing ######################## 
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
        CroppedImage = dst
        ## Setting background to black (is)
        bg = np.ones_like(cropped, np.uint8)*0
        cv2.bitwise_not(bg,bg, mask=mask)
        dst = bg+dst
        ### THIS IS VERY IMPORTANT!!!!! It is The Main Preprocessing ###
        dst = cv2.medianBlur(dst, ksize=9)
        dst = cv2.bilateralFilter(dst,9,75,75)
        
        ######################## Main Processing ######################## 
        ## Computing HSV Threshold (Color Threshold) ##
        frame_HSV = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        if sampleintensity > 120:
            HSV_Threshold = cv2.inRange(frame_HSV, (0, 0, 90), (180, 15, 255)) ## HSV Range for White/Greyish
        else:
            HSV_Threshold = cv2.inRange(frame_HSV, (0, 0, 120), (180, 15, 255)) ## HSV Range for White/Greyish


        ## Computing Otsu Threshold (Intensity Threshold) ##
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        Thresh = FindThreshold(dst, mask)
        prevThreshFrame.append(Thresh)

        ## Kernel Smoothing on Otsu Threshold ##
        if(len(prevThresh) >= 6):
            ThreshList = GetPrevThresh(int(i/4), 6) 
            CurrentMean = np.mean(ThreshList)
            CurrentStd = np.std(ThreshList)

            if (Thresh < CurrentMean - .5*CurrentStd or Thresh > CurrentMean + .5*CurrentStd):
                Thresh = CurrentMean
                
        else:
            ThreshList = GetPrevThresh(int(i/4), 0) 
            if (len(ThreshList) > 0):
                CurrentMean = np.mean(ThreshList)
                CurrentStd = np.std(ThreshList)
                if (Thresh < CurrentMean - .5*CurrentStd or Thresh > CurrentMean + .5*CurrentStd):
                    Thresh = CurrentMean

        Otsu_Threshold = cv2.threshold(dst,Thresh,255,cv2.THRESH_BINARY)[1]
        
        ## Comparing Both SnowCover Measurements With Normalized RMSE ##
        score = normalized_root_mse(Otsu_Threshold, HSV_Threshold)
        global SnowCoverBuffer
        if (score > .90 and len(SnowCoverBuffer) > 0):
            HSVSnow = math.floor((1 - ((len(np.extract(mask > 0, HSV_Threshold)) - np.count_nonzero(np.extract(mask > 0, HSV_Threshold)))/len(np.extract(mask > 0, HSV_Threshold))))*100)/100
            OTSUSnow = math.floor((1 - ((len(np.extract(mask > 0, Otsu_Threshold)) - np.count_nonzero(np.extract(mask > 0, Otsu_Threshold)))/len(np.extract(mask > 0, Otsu_Threshold))))*100)/100
            if (len(SnowCoverBuffer) < 5):
                SnowCoverKernel = SnowCoverBuffer
            else:        
                SnowCoverKernel = SnowCoverBuffer[-5:-1]

            SnowCoverKernelList = [j[int(i/4)+1] for j in SnowCoverKernel]
            meanSnowCover = np.mean(SnowCoverKernelList)
            if (abs(HSVSnow - meanSnowCover) > abs(OTSUSnow - meanSnowCover)):
                Final = Otsu_Threshold
            else:
                Final = HSV_Threshold

        elif (len(SnowCoverBuffer) == 0):
            Final = HSV_Threshold
        else:
            Final = Otsu_Threshold

        SnowCoverPercentage = math.floor((1 - ((len(np.extract(mask > 0, Final)) - np.count_nonzero(np.extract(mask > 0, Final)))/len(np.extract(mask > 0, Final))))*100)/100
        SnowCoverFrame.append(SnowCoverPercentage)


        #### Saving Training Data ####
        MaskName = str(CurrentTimeStamp) + '_Panel_' + str(PanelDictionary[PanelID][int(i/4)]) + '_Mask.jpg'
        ImageName = str(CurrentTimeStamp) + '_Panel_' + str(PanelDictionary[PanelID][int(i/4)]) + '_Image.jpg'
        os.chdir(MasksPath)
        cv2.imwrite(MaskName, Final)
        os.chdir(ImagesPath) 
        cv2.imwrite(ImageName, CroppedImage)
        os.chdir(path)



    #### Generating Debugging Video ####
        if (GenerateVid):
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            PanelVidArray.append(cv2.hconcat([cropped, HSV_Threshold, Otsu_Threshold, Final, dst]))
    
    if (GenerateVid):
        global imgArray 
        VideoFrame = vconcat_resize_min(PanelVidArray)
        height, width = VideoFrame.shape
        VideoHeader = np.zeros((75,width), np.uint8)
        cv2.putText(VideoHeader,CurrentTimeStamp.strftime("%Y-%m-%d %H:%M:00"),(0,60),cv2.FONT_HERSHEY_SIMPLEX,2,255,3)
        imgArray.append(vconcat_resize_min([VideoHeader, VideoFrame]))


    prevThresh.append(prevThreshFrame)
    SnowCoverBuffer.append(SnowCoverFrame)








# Callback function for getting coordinates of the solar panel mask from 
# the click event. This will update the global MaskCoordinates Variable on Click
def click_event(event, x, y, flags, params):
	global MaskCoordinates
	
    # Listening for Left Click
	if event == cv2.EVENT_LBUTTONDOWN:

        # Print and Append Coordinates
		#print(x, ' ', y)
		MaskCoordinates.append((x, y))

        # Display Coordinates 
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(imgCopy,'+', (x,y), font, .2, (255,164,0))
		cv2.imshow('image', imgCopy)

	# Listening for Right Click	
	if event==cv2.EVENT_RBUTTONDOWN:

        # Print and Append Coordinates
		#print(x, ' ', y)
		MaskCoordinates.append((x, y))


		# Display Coordinates 
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(imgCopy,'+', (x,y), font, .2, (255,164,0))
		cv2.imshow('image', imgCopy)


# Driver function for MaskCoordinates array generation
# This function will take the initial frame of ever timelapse video
def generateMasks(imgCopy):
    cv2.namedWindow('TimeLapse Frame', cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    # Displaying the image
    cv2.imshow('image', imgCopy)
    # Running MouseClick Callback
    cv2.setMouseCallback('image', click_event)
    # Exiting when a key is pressed
    while True:
        k = cv2.waitKeyEx(0) & 0xFF
        print(k)
        if k == 27:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break

def CollateData():    
    CollateStatus = messagebox.askyesno('Yes|No', 'Collate Data?')
    
    if (CollateStatus):
        path = search_for_file_path()
        os.chdir(path)
        DataFrameList = [] #Pull the current list of files in TimeLapse directory
        for file in os.listdir(path):
            if file.endswith(".csv") and not file.startswith('.'): 
                DataFrameList.append(file)
        
        
        PTSA1 = pd.read_csv(DataFrameList[0])
        PTSA2 = pd.read_csv(DataFrameList[1])
        PTSA3 = pd.read_csv(DataFrameList[2])

        FinalData = pd.merge(PTSA1, PTSA2, how="outer", on=["TimeStamp"])
        FinalData = pd.merge(FinalData, PTSA3, how="outer", on=["TimeStamp"])
        FinalData.to_csv('CollatedFinalData.csv', index=False)




#################### Main Script ####################
if __name__ == "__main__":

    ProcessingStatus = True
    while(ProcessingStatus):
        GenerateVid = messagebox.askyesno('Yes|No', 'Do you want to generate a test video?')
        #### Directory Management ####
        path = search_for_file_path()
        os.chdir(path) #Change the working directory to the TimeLapse directory
        TimeLapseList = [] #Pull the current list of files in TimeLapse directory
        for file in os.listdir(path):
            if file.endswith(".mp4") and not file.startswith('.'): 
                TimeLapseList.append(file)

        #### Generating Training Data Directory ####
        TrainingDataPath = os.path.join(path, 'Train')
        os.mkdir(TrainingDataPath)
        MasksPath = os.path.join(TrainingDataPath, 'Masks')
        os.mkdir(MasksPath)
        ImagesPath = os.path.join(TrainingDataPath, 'Images')
        os.mkdir(ImagesPath)
            
       
        for i in TimeLapseList:
            os.chdir(path) # We have to set the path every time since cv2 can't handle relative paths without it.
            currentVideo = cv2.VideoCapture(i, cv2.IMREAD_GRAYSCALE) # Reading in the current time lapse video. 
            success, frame = currentVideo.read()
            imgCopy = copy.deepcopy(frame)
            generateMasks(imgCopy)
            PanelID = ExtractPanelID(frame)
            fno = 0 
            sample_rate = 1
            with MoonSpinner('Running Image Processing Algorithm ') as bar1:
                while success:
                    if fno % sample_rate == 0:
                        ImageProcess(frame, PanelID)

                    success, frame = currentVideo.read()
                    bar1.next()
        
            ## Resetting mask coordinates list
            MaskCoordinates = []
            TimeStampBuffer = []
            prevThresh = []
            SnowCover.extend(SnowCoverBuffer)
            SnowCoverBuffer = []
            

        ############### Writing Video ###############
        if (GenerateVid):
            height, width = imgArray[0].shape
            size = (width,height)
            out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, size, 0)
            with Bar('Resizing Test Video Frames ...', fill='#', suffix='%(percent).1f%% - %(eta)ds') as bar:
                for i in range(len(imgArray)):  
                    imgArray[i] = cv2.resize(imgArray[i], size) 
                    bar.next()
            
            with MoonSpinner('Exporting Test Video ...  ') as bar:
                for i in range(len(imgArray)):
                    out.write(imgArray[i])
                    bar.next()
                out.release()
            imgArray = []

        ############### Data Preparation/Collation ###############
        header = [i for i in PanelDictionary[PanelID]]
        header.insert(0, 'TimeStamp')
        SnowCover.insert(0, header)

        path = search_for_file_path()
        os.chdir(path)
        with open("PanelID_{}.csv".format(PanelID), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(SnowCover)
        

        SnowCover = []
        ProcessingStatus = messagebox.askyesno('Yes|No', 'Is there more data?')
    
    CollateData()
        


    




    



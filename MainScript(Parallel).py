#!/usr/bin/env python3

#################### Dependencies #################### 
## Data Management
import numpy as np
import pandas as pd
import csv
## Directory Management
import os 
## Image Processing
import cv2  
import copy
import math
import time

## Optical Character Recognition
import easyocr
from datetime import datetime 
from datetime import timedelta

## GUI and Terminal Elements
from progress.bar import Bar
from progress.spinner import MoonSpinner
import tkinter
from tkinter import filedialog
from tkinter import messagebox

## Deeplearning Model
import tensorflow as tf
import segmentation_models as sm
import keras 
from keras.metrics import MeanIoU
from keras.utils import to_categorical
from keras.models import load_model


#################### Global Variables #################### 
## Segmentation Model Variables
model1 = load_model('saved_models/linkNet_arch_res50_backbone_50epochs.hdf5', compile=False)
BACKBONE1 = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE1)
SIZE_X = 416 
SIZE_Y = 640

## Initializing OCR Engine
reader = easyocr.Reader(['en']) 

PanelDictionary = { 
    '2':[10, 8, 2, 5],
    '1':[1, 4, 7, 11], 
    '3':[9, 3, 6, 12],
    'A':['all', 'all', 'all', 'all']
}

## Data storage python lists 
MaskCoordinates = []
TimeStampBuffer = []
SnowCover = []
SnowCoverBuffer = []
prevThresh = []

# Testing video
imgArray = []


## Initialize GUI
root = tkinter.Tk()
root.withdraw() 




#################### Helper Functions ####################

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
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=2)
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


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


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

    PreProcessingArray = []
    mask = []
    croppedArray = []

    for i in range(0,len(MaskCoordinates),4):
        pts = np.array([MaskCoordinates[i],MaskCoordinates[i+1],MaskCoordinates[i+2],MaskCoordinates[i+3]])
        
        ## Cropping the bounding rectangle
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        cropped = img[y:y+h, x:x+w].copy()


        ## Generating the Mask
        pts = pts - pts.min(axis=0)
        maskPanel = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(maskPanel, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        mask.append(maskPanel)

        ## Removing everything outside the Mask with bitwise operation
        dst = cv2.bitwise_and(cropped, cropped, mask=maskPanel)
        ## Setting background to black (is)
        bg = np.ones_like(cropped, np.uint8)*0
        cv2.bitwise_not(bg,bg, mask=maskPanel)
        dst = bg+dst
        croppedArray.append(dst)
        ## Network Prediction  ###To parallelize take this out of the for loop, and pass in a tensor of the four panels. 
        NetworkInput = cv2.resize(dst, (SIZE_Y, SIZE_X))
        PreProcessingArray.append(NetworkInput)

    ## Main Block that handles preprocessing and prediction of tensor. 
    PreProcessingArray = np.array(PreProcessingArray)
    test_img_input = preprocess_input(PreProcessingArray)
    test_pred1 = model1.predict(test_img_input, verbose = False, use_multiprocessing = True)
    test_prediction1 = [np.argmax(i, axis=2) for i in test_pred1]
    NetworkPred = []

    ## Rest of processing that isn't parallelized. 
    for i in range(0,4):
        ResizedandNormalized = cv2.resize(test_prediction1[i], (mask[i].shape[1], mask[i].shape[0]), interpolation = cv2.INTER_NEAREST_EXACT)
        ResizedandNormalized = cv2.normalize(src=ResizedandNormalized, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        NetworkPred.append(ResizedandNormalized)


    for i in range(0,4):
        SnowCoverPercentage = math.floor((1 - ((len(np.extract(mask[i] > 0, NetworkPred[i])) - np.count_nonzero(np.extract(mask[i] > 0, NetworkPred[i])))/len(np.extract(mask[i] > 0, NetworkPred[i]))))*100)/100
        SnowCoverFrame.append(SnowCoverPercentage)
    
    if (GenerateVid):
        for i in range(0,4):
            cropped = cv2.cvtColor(croppedArray[i], cv2.COLOR_BGR2GRAY) 
            PanelVidArray.append(cv2.hconcat([cropped, NetworkPred[i]]))
    
    if (GenerateVid):
        global imgArray 
        VideoFrame = hconcat_resize_min(PanelVidArray)
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
		cv2.putText(img,'+', (x,y), font, .2, (255,164,0))
		cv2.imshow('image', img)

	# Listening for Right Click	
	if event==cv2.EVENT_RBUTTONDOWN:

        # Print and Append Coordinates
		#print(x, ' ', y)
		MaskCoordinates.append((x, y))


		# Display Coordinates 
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,'+', (x,y), font, .2, (255,164,0))
		cv2.imshow('image', img)


# Driver function for MaskCoordinates array generation
# This function will take the initial frame of ever timelapse video
def generateMasks(img):
    cv2.namedWindow('TimeLapse Frame', cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    # Displaying the image
    cv2.imshow('image', img)
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

       
        for i in sorted(TimeLapseList):
            os.chdir(path) # We have to set the path everytime since cv2 can't handle relative paths without it.
            currentVideo = cv2.VideoCapture(i, cv2.IMREAD_GRAYSCALE) # Reading in the current TimeLapse Video
            success, img = currentVideo.read()
            generateMasks(img)
            fno = 0 
            sample_rate = 1
            PanelID = ExtractPanelID(img)
            with MoonSpinner('Running Image Processing Algorithm ') as bar1:
                while success:
                    if fno % sample_rate == 0:
                        ImageProcess(img, PanelID)
                    # read next frame

                    #print('Next Frame')
                    success, img = currentVideo.read()
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

        ############### Data Preparation ###############
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
        


    




    



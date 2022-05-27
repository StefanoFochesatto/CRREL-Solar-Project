################# Dependencies  #################
## Pandas and Csv for Data IO and Manipulation
import csv 
import pandas as pd
## OS for directory management
import os 
## Datetime for timestamp parsing and management
from datetime import datetime 
## Schedule and Time for Scripting
import schedule
import time
# TimeStamp,Water_elev,Water_temp


####### Global Variables for setting when the script runs ##########
## For hourly updating, set scheduleTimer to 'Hourly' like this, 
scheduleTimer = 'Hourly'

#scheduleTimer = 'Daily'


## Set path for DataFile. Use Full path
path = r"C:\Users\Amanda Barker\Desktop\Stefano\CRREL-Solar-Project\Art's Web Server Data Scrub\Data\Data" 
os.chdir(path) #Change the working directory to the Data directory


## Function for Parsing and Changing File Names
def ChangeFileNames():
    DataFileList = GetDataFileList()
    for file in DataFileList:
        if (len(file) == 41):
            newfilename = file[0:31] + file[37:41]
            print(newfilename)
            os.rename(file, newfilename)


## Function for pulling the file list in the datafile. (exludes the master datafile)
def GetDataFileList():
    DataFileList = [] 
    for file in os.listdir(path):
        if file.endswith(".csv"): 
            DataFileList.append(file)
    return DataFileList[1:]



### Global Variable for the current data list. 
### This initilization means that after restarts the first run of the script does nothing,
### So after a restart make sure script starts before any new data is added to the folder. 
CurrentDataFileList = GetDataFileList()

### Use this VVVVVV initilization if that is not feasible
# CurrentDataFileList = []
## The first run of the script will check against all data files in the folder, so it may take a while. 



## Function for collating the data, minding for overlapping data files.
# If the script starts slowing down dramitically this section is what is doing that.  
def CollateData(UpdatedDataFileList, TimeStamp, Column_1, Column_2):
    for file in UpdatedDataFileList:
        ## Pulling Data from each file in the DataFileList
        TimeStampCurrentFile = []
        Column_1CurrentFile = []
        Column_2CurrentFile = []
        CSVDictionary = csv.DictReader(open(file, 'r'), fieldnames=['TimeStamp', 'Water_elev', 'Water_temp'])
        for col in CSVDictionary:
            TimeStampCurrentFile.append(col['TimeStamp'])
            Column_1CurrentFile.append(col['Water_elev'])
            Column_2CurrentFile.append(col['Water_temp'])

        ## Iterating through the current file     
        for i in range(len(TimeStampCurrentFile)):
            ## Pulling newest timestamp from Masterfile and the ith timestamp of the current file
            LatestTimeStamp = datetime.strptime(TimeStamp[-1], "%m/%d/%Y %H:%M:%S")
            NextTimeStamp = datetime.strptime(TimeStampCurrentFile[i], "%m/%d/%Y %H:%M:%S")
            ## Timestamp comparison. Only appending to Masterfile when ith timestamp of the current file is bigger than the newest Masterfile timestamp
            if (NextTimeStamp > LatestTimeStamp):
                TimeStamp.append(TimeStampCurrentFile[i])
                Column_1.append(Column_1CurrentFile[i])
                Column_2.append(Column_2CurrentFile[i])

    ## Writing MasterFile
    MasterFile = pd.DataFrame({'TimeStamp':TimeStamp, 'Water_elev':Column_1, 'Water_temp': Column_2})
    return MasterFile

## Main Script File
def main():
    print('Entering Main')
    ## Updating datafile names.
    ChangeFileNames()
    ## Checking for new files.
    UpdatedDataFileList = GetDataFileList()
    UpdatedDataFileList = [x for x in UpdatedDataFileList if x not in CurrentDataFileList]
    if(len(UpdatedDataFileList) == 0):
        print('no new files')
        return
    
    ## Import MasterFile
    MasterFile = pd.read_csv('CRREL_CSV_Forebay.csv')
    TimeStamp = list(MasterFile['TimeStamp'])
    Column_1  = list(MasterFile['Water_elev'])
    Column_2  = list(MasterFile['Water_temp'])
    
    ## Another check for our MasterFile. Comparing the newest Masterfile timestamp with the newest data file timestamp.   
    LatestFile = pd.read_csv(UpdatedDataFileList[-1], index_col = False, header=None, names = ['TimeStamp','Water_elev','Water_temp' ])
    TimeStampLatestFile = list(LatestFile['TimeStamp']) 
    CurrentTimeStamp = datetime.strptime(TimeStamp[-1], "%m/%d/%Y %H:%M:%S")
    NextTimeStamp = datetime.strptime(TimeStampLatestFile[-1], "%m/%d/%Y %H:%M:%S")
    if (NextTimeStamp > CurrentTimeStamp):
        MasterFile = CollateData(UpdatedDataFileList, TimeStamp, Column_1, Column_2)
        MasterFile.to_csv('CRREL_CSV_Forebay.csv', index = False)

    ## Updating Current Data File List
    global CurrentDataFileList
    CurrentDataFileList.extend(UpdatedDataFileList) 
        
    

if __name__ == "__main__":

    if (scheduleTimer == 'Daily'):
        schedule.every().day.at("09:25").do(main)
    
    if (scheduleTimer == 'Hourly'):
        schedule.every().hour.at(":15").do(main)

    while True:
        schedule.run_pending()
        print('Sleeping')
        print(CurrentDataFileList)
        time.sleep(60) # wait one minute
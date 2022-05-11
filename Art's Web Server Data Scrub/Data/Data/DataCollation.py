## Dependencies
import csv #Pandas for Data IO and Manip
import os #os for directory management
from datetime import datetime #For Datetime Management

path = r"C:\Users\Amanda Barker\Desktop\Stefano\Art's Web Server Data Scrub\Data\Data" # Add full path to data folder
os.chdir(path) #Change the working directory to the Data directory

def main():
    #### Directory Management
    DataFileList = [] #Pull the current list of files in Data directory
    for file in os.listdir(path):
        if file.endswith(".csv"): 
            DataFileList.append(file)

    if (False):        
        #### Renaming the Files 
        ##  CRREL_CSV_Forebay_YEAR-MONTH-DAY_HOUR.csv
        ## Example:
        ##  CRREL_CSV_Forebay_2022-05-10_19.csv
        for file in DataFileList:
            newfilename = file[0:31] + file[37:]
            print(newfilename)
            os.rename(file, newfilename)
    print(DataFileList)
    #### Collating Data
    TimeStamp = []
    Column_1 = []
    Column_2 = []
    for file in DataFileList[0:2]:
            ## Pulling Data from each file in the DataFileList
            TimeStampCurrentFile = []
            Column_1CurrentFile = []
            Column_2CurrentFile = []
            CSVDictionary = csv.DictReader(open(file, 'r'), fieldnames=['TimeStamp', 'Column1', 'Column2'])
            for col in CSVDictionary:
                TimeStampCurrentFile.append(col['TimeStamp'])
                Column_1CurrentFile.append(col['Column1'])
                Column_2CurrentFile.append(col['Column2'])

            # When the Master TimeStampList is not empty, check for overlapping timestamps
            if (len(TimeStamp) != 0):    
                for i in range(len(TimeStampCurrentFile)):
                    LatestTimeStamp = datetime.strptime(TimeStamp[-1], "%m/%d/%Y %H:%M:%S")
                    NextTimeStamp = datetime.strptime(TimeStampCurrentFile[i], "%m/%d/%Y %H:%M:%S")
                    if (NextTimeStamp > LatestTimeStamp):
                        TimeStamp.append(TimeStampCurrentFile[i])
                        Column_1.append(Column_1CurrentFile[i])
                        Column_2.append(Column_2CurrentFile[i])

            else:
                TimeStamp = TimeStamp + TimeStampCurrentFile
                Column_1 = Column_1 + Column_1CurrentFile
                Column_2 = Column_2 + Column_2CurrentFile

    ## Writing the the MasterFile
    DataTranspose = zip(TimeStamp, Column_1, Column_2)
    with open('test.csv', "w") as f:
        writer = csv.writer(f)
        for row in DataTranspose:
            writer.writerow(row)
    

if __name__ == "__main__":
    main()
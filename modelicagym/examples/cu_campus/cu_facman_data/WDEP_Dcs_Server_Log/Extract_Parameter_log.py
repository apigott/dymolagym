# Extracting the parameters from each server logs(.csv) into a data frame and stitching the DF to have a annual log.

import os
import glob
import re as re
import pandas as pd
path = os.getcwd()
import shutil
print(path)


# Function to navigate to folders and extract data and write it into a csv file
# Recheck the row and column number specified for each parameter if the data is not extracting correctly
# Example script to extract 10 parameter specific to CHP and Heating in WDEP

def Extract(M,Y):

    #navigating to each folder
    Folder_path = path +"\\dcs\\dcs_logs_" +str(Y)+"\\"+str(M) + str(Y)

    #Finding Excel files within the folder
    os.chdir(Folder_path)
    files = glob.glob('*.xls')
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print(files)
    combine_df = pd.DataFrame()

    #Reading each file and extracting the data into df
    for F in files:
        #Define data frames
        df1 = pd.DataFrame() #appends data from sheet
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        df_apnd = pd.DataFrame() #df built from each file

        # Navigate to sheets
        sheet1 = pd.read_excel(F, '1.0.0') # Specify the sheet name, eg: 1.0.0
        df1 = df1.append(sheet1)
        d = F[0:-4]
        df_apnd['Hours'] = df1.iloc [11:34,0] # df2{column header] = df.iloc [rows, column (in the server log files)]
        df_apnd['HRSG_FW HTR_Tin'] = df1.iloc [11:34,8]
        df_apnd['HRSG_FW HTR_Tout'] = df1.iloc [11:34,9]
        df_apnd['HRSG_Eco_Tout'] = df1.iloc [11:34,10]
        df_apnd['HRSG_Ste_T'] = df1.iloc [11:34,3]
        df_apnd['HRSG_Ste_p'] = df1.iloc [11:34,2]

        sheet2 = pd.read_excel(F, '1.0.1') # parameters from Sheet 2
        df2 = df2.append(sheet2)
        df_apnd['CTG_output'] = df2.iloc [11:34,1]
        df_apnd['NOx_PPM'] = df2.iloc [11:34,12]
        df_apnd['CO_PPM'] = df2.iloc [11:34,13]


        sheet3 = pd.read_excel(F, '3.2.0') # parameters from Sheet 3
        df3 = df3.append(sheet3)
        df_apnd['CampusLoad_1'] = df3.iloc [10:34,6]
        df_apnd['CampusLoad_2'] = df3.iloc [10:34,7]

       #Combining the df
        df_apnd.insert(0,'Date',d)
        combine_df = combine_df.append(df_apnd, ignore_index=True)

    # writing as .csv
    os.chdir(path)
    combine_df.to_csv(M+Y+'.csv',index=False)



def move(Y):
    os.chdir(path)
    os.mkdir(Y)
    source = path
    des = path+"\\"+Y
    files = os.listdir(source)

    for file in files:
        if file.endswith(Y +'.csv'):
            shutil.move(os.path.join(source,file), os.path.join(des,file))

def combine (Y):
    source = path+"\\"+Y
    os.chdir(source)
    files = os.listdir(source)
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file))
        annual = pd.concat(dfs, ignore_index=True)
        annual.to_csv(Y+'.csv',index=False)



#%% Run year 2017
# months = ["01_jan_","02_feb_","03_mar_","04_apr_","05_may_","06_jun_","07_jul_","08_aug_","09_sep_","10_oct_","11_nov_","12_dec_"]
months = ["01_jan_", "02_feb_"]

year = "2017"
for i in months:
    Extract(i,year)

#%% Move to folders
move(year)
combine(year)

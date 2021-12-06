## Test to see is the cell locations of parameters are correct
## all the dcs log files have the same format 

import sys
import os
import glob 
import string
import re as re
import pandas as pd
import numpy as np
path = os.getcwd()
import shutil
print(path)

df1 = pd.DataFrame()
df_apnd = pd.DataFrame() #df built from each file 
combine_df = pd.DataFrame()

df = pd.read_excel (r'E:\Projects\DOE_GED\WDEP_DCS\test\Sample_dcs.xls', '3.2.0')
df1 = df1.append(df)

df_apnd['CampusLoad_1'] = df1.iloc [10:34,6]
df_apnd['CampusLoad_2'] = df1.iloc [10:34,7]

combine_df = combine_df.append(df_apnd, ignore_index=True)
combine_df.to_csv('test.csv',index=False)

# Importing numpy 
import numpy as np

# Importing Scipy 
import scipy as sp

# Importing Pandas Library 
import pandas as pd

# import glob function to scrap files path
from glob import glob
from IPython.display import display

##ethan##
from s_support import ProgressBar, turn_off_plt, mark_time, mark_milestone, has_flag, prompt_exception, prompt_highlight
mark_time("X_Start")

_Raw_data_paths = sorted(glob("Data/Original-Data/HAPT-Dataset/Raw-Data/*"))
# Selecting acc file paths only
Raw_acc_paths=_Raw_data_paths[0:61]
# Selecting gyro file paths only
Raw_gyro_paths=_Raw_data_paths[61:122]
# Labels file
Labels_path = _Raw_data_paths[-1]

# Raw_skip_ratio = 1
# if has_flag("FlSkipRaw"):
#     Raw_skip_ratio = 20
#     _Raw_data_paths = _Raw_data_paths[::Raw_skip_ratio]
#     Raw_acc_paths = Raw_acc_paths[::Raw_skip_ratio]
#     Raw_gyro_paths = Raw_gyro_paths[::Raw_skip_ratio]

# printing info related to acc and gyro files
print (("RawData folder contains in total {:d} file ").format(len(_Raw_data_paths)))
print (("The first {:d} are Acceleration files:").format(len(Raw_acc_paths)))
print (("The second {:d} are Gyroscope files:").format(len(Raw_gyro_paths)))
print ("The last file is a labels file")
print ("labels file path is:", Labels_path)

#    FUNCTION: import_raw_signals(path,columns)
#    ###################################################################
#    #           1- Import acc or gyro file                            #
#    #           2- convert from txt format to float format            #
#    #           3- convert to a dataframe & insert column names       #
#    ###################################################################                      

def import_raw_signals(file_path,columns):
    ######################################################################################
    # Inputs:                                                                            #
    #   file_path: A string contains the path of the "acc" or "gyro" txt file            #
    #   columns: A list of strings contains the column names in order.                   #
    # Outputs:                                                                           #
    #   dataframe: A pandas Dataframe contains "acc" or "gyro" data in a float format    #
    #             with columns names.                                                    #
    ######################################################################################


    # open the txt file
    opened_file =open(file_path,'r')

    # Create a list
    opened_file_list=[]
    
    # loop over each line in the opened_file
    # convert each element from txt format to float 
    # store each raw in a list
    for line in opened_file:
        opened_file_list.append([float(element) for element in line.split()])

    # convert the list of lists into 2D numpy array(computationally efficient)
    data=np.array(opened_file_list)


    # Create a pandas dataframe from this 2D numpy array with column names
    data_frame=pd.DataFrame(data=data,columns=columns)

    # return the data frame
    return data_frame


########################################### RAWDATA DICTIONARY ##############################################################

# creating an empty dictionary where all dataframes will be stored
raw_dic={}


# creating list contains columns names of an acc file
raw_acc_columns=['acc_X','acc_Y','acc_Z']

# creating list contains gyro files columns names
#raw_gyro_columns=['gyro_X','gyro_Y','gyro_Z']

NumFR = len(Raw_acc_paths)
pgb = ProgressBar(NumFR, "loading_raw_data")
# loop for to convert  each "acc file" into data frame of floats and store it in a dictionnary.
for path_index in range(0,NumFR):
        pgb.inc()
        
        # extracting the file name only and use it as key:[expXX_userXX] without "acc" or "gyro"
        exp_user_key= Raw_acc_paths[path_index][-16:-4]
        
        # Applying the function defined above to one acc_file and store the output in a DataFrame
        raw_acc_data_frame=import_raw_signals(Raw_acc_paths[path_index],raw_acc_columns)
        
        # By shifting the path_index by 61 we find the index of the gyro file related to same experiment_ID
        # Applying the function defined above to one gyro_file and store the output in a DataFrame
        #raw_gyro_data_frame=import_raw_signals(Raw_gyro_paths[path_index],raw_gyro_columns)
        
        # concatenate acc_df and gyro_df in one DataFrame
        #raw_signals_data_frame=pd.concat([raw_acc_data_frame, [raw_gyro_data_frame]], axis=1)
        
        # Store this new DataFrame in a raw_dic , with the key extracted above
        raw_dic[exp_user_key] = raw_acc_data_frame


# %%
# raw_dic is a dictionary contains 61 combined DF (acc_df and gyro_df)
print('raw_dic contains {} DataFrame, each for per person'.format(len(raw_dic)))
all_exp_user_key = raw_dic.keys()
print("raw_dic can be accessed by exp_user_key, typical key are: ", list(all_exp_user_key)[0:3])
mark_time("all_raw_data_loaded_in_raw_dic: {}".format(len(raw_dic)))

# print the first 3 rows of dataframe exp01_user01
display(raw_dic['exp01_user01'].head(3))
mark_milestone("all_raw_data_loaded:raw_dic")

#    FUNCTION: import_raw_labels_file(path,columns)
#    #######################################################################
#    #      1- Import labels.txt                                           #
#    #      2- convert data from txt format to int                         #
#    #      3- convert integer data to a dataframe & insert columns names  #
#    #######################################################################  
def import_labels_file(path,columns):
    ######################################################################################
    # Inputs:                                                                            #
    #   path: A string contains the path of "labels.txt"                                 #
    #   columns: A list of strings contains the columns names in order.                  #
    # Outputs:                                                                           #
    #   dataframe: A pandas Dataframe contains labels  data in int format                #
    #             with columns names.                                                    #
    ######################################################################################
    
    
    # open the txt file
    labels_file =open(path,'r')
    
    # creating a list 
    labels_file_list=[]
    
    
    #Store each row in a list ,convert its list elements to int type
    for line in labels_file:
        labels_file_list.append([int(element) for element in line.split()])
    # convert the list of lists into 2D numpy array 
    data=np.array(labels_file_list)
    
    # Create a pandas dataframe from this 2D numpy array with column names 
    data_frame=pd.DataFrame(data=data,columns=columns)
    
    # returning the labels dataframe 
    return data_frame

# %% [markdown]
# - [**Return to Import Data**](#step1)
# * [**Return Back to the Beginning**](#step0)
# %% [markdown]
# <a id='step15'></a>
# ### I.5. Apply import_labels_file
# 
# - Apply import_raw_labels_file to "labels.txt" path
# - Store the labels in a Pandas Data frame called **Labels_Data_Frame**

# %%
#################################
# creating a list contains columns names of "labels.txt" in order
raw_labels_columns=['experiment_number_ID','user_number_ID','activity_number_ID','Label_start_point','Label_end_point']

# The path of "labels.txt" is last element in the list called "Raw_data_paths"
# labels_path=Raw_data_paths[-1]

# apply the function defined above to labels.txt 
# store the output  in a dataframe 
Labels_Data_Frame=import_labels_file(Labels_path,raw_labels_columns)
print("Labels_Data_Frame", Labels_Data_Frame.shape)
print(Labels_Data_Frame.head(3))
mark_time("import_labels_file")

print(Labels_Data_Frame.shape)
mark_milestone("import_labels_file:Labels_Data_Frame")

# %% [markdown]
# - [**Return to Import Data**](#step1)
# * [**Return Back to the Beginning**](#step0)
# %% [markdown]
# <a id='step16'></a>
# ### I.6. Define Activity Labels Dic

# %%
# Creating a dictionary for all types of activities
# The first 6 activities are called Basic Activities as(BAs) 3 dynamic and 3 static
# The last 6 activities are called Postural Transitions Activities as (PTAs)
Acitivity_labels=AL={
        1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', # 3 dynamic activities
        4: 'SITTING', 5: 'STANDING', 6: 'LIYING', # 3 static activities
        
        7: 'STAND_TO_SIT',  8: 'SIT_TO_STAND',  9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT', 
        11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND',# 6 postural Transitions
       } 


# a list contains the number of rows per dataframe 
rows_per_df=[len(raw_dic[key]) for key in sorted(raw_dic.keys())]

# a list contains exp ids
exp_ids=[i for i in range(1,NumFR+1)]

# useful row is row that was captured while the user was performing an activity
# some rows in acc and gyro files are not associated to an activity id

# list that will contain the number of useful rows per dataframeç
useful_rows_per_df=[]

for i in range(1,NumFR+1):# iterating over exp ids
    # selecting start-end rows of each activity of the experiment
    start_end_df= Labels_Data_Frame[Labels_Data_Frame['experiment_number_ID']==i][['Label_start_point','Label_end_point']]
    # sum of start_labels and sum of end_labels
    start_sum,end_sum=start_end_df.sum()
    # number of rows useful rows in [exp i] dataframe
    useful_rows_number=end_sum-start_sum+len(start_end_df)
    # storing row numbers in a list
    useful_rows_per_df.append(useful_rows_number)


def Windowing_type_1(time_sig, exp_user_key, raw_prefix):

    window_ID = 0  # window unique id
    win_time_sig = {}  # output dic

    BA_array = np.array(Labels_Data_Frame[(
        Labels_Data_Frame["activity_number_ID"] < 7)])  # Just Basic activities

    for line in BA_array:
        # Each line in BA_array contains info realted to an activity

        # extracting the dataframe key that contains rows related to this activity [expID,userID]
        file_key = "exp{:02d}_user{:02d}".format(int(line[0]), int(line[1]))
        if file_key == exp_user_key:

            # extract the activity id in this line
            act_ID = line[2]  # The activity identifier from 1 to 6 (6 included)

            # starting point index of an activity
            start_point = line[3]

            # from the cursor we copy a window that has 128 rows
            # the cursor step is 64 data point (50% of overlap) : each time it will be shifted by 64 rows
            for cursor in range(start_point, line[4] - 127, 64):

                # end_point: cursor(the first index in the window) + 128
                end_point = cursor + 128  # window end row

                data = time_sig.iloc[cursor:end_point]

                # creating the window
                key = '{}_t_W{:05d}_{}_act{:02d}'.format(raw_prefix, window_ID, file_key, act_ID)
                win_time_sig[key] = np.array(data)

                # incrementing the windowID by 1
                window_ID = window_ID + 1
            return win_time_sig

    return win_time_sig  # return a dictionary including time domain windows type I


for eu in ((1,1), (2,1), (3,2), (4,2)):
    exp_no = eu[0]
    user_no = eu[1]    
    exp_user_key = "exp{:02d}_user{:02d}".format(exp_no, user_no)
    if exp_user_key in raw_dic:
        raw_time_sigs = raw_dic[exp_user_key]
        time_sig_x = raw_time_sigs["acc_X"]
        time_sig_y = raw_time_sigs["acc_Y"]
        time_sig_z = raw_time_sigs["acc_Z"]
        prompt_highlight("win_time_sigs for", exp_user_key)
        win_time_sig_x = Windowing_type_1(time_sig_x, exp_user_key, "aX")
        display(win_time_sig_x.keys())

        win_time_sig_y = Windowing_type_1(time_sig_y, exp_user_key, "aY")
        display(win_time_sig_y.keys())

        win_time_sig_z = Windowing_type_1(time_sig_z, exp_user_key, "aZ")
        display(win_time_sig_z.keys())

prompt_highlight("done")

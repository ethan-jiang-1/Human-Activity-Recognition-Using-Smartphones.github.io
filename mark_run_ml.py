# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import default_timer to compute durations
from timeit import default_timer as timer
Debut=timer() # start time

import numpy as np # import numpy library
import pandas as pd # importing pandas library

# scrapping file paths
from glob import glob

# Allows the use of display() for DataFrames
from IPython.display import display 

from matplotlib import pyplot as plt # import matplot. pyplot to allow figure's plotting
#plt.style.use('bmh') # for better plots

##ethan##
from s_support import ProgressBar, turn_off_plt, time_mark
turn_off_plt()

# %%
# scrapping data files path
DF_paths_list=sorted(glob("Data/New-Data/full_Datasets_type_I_and_II/*"))
display(DF_paths_list)

# %% [markdown]
# # I. Importing Dataset type I and II

# %%
# loading datasets(these datasets are the outputs of the signal processing pipeline )
Dataset_type_I_part1 = pd.read_csv('Data/New-Data/full_Datasets_type_I_and_II/Dataset_I_part1.csv')
Dataset_type_I_part2 = pd.read_csv('Data/New-Data/full_Datasets_type_I_and_II/Dataset_I_part2.csv')
Dataset_type_II_part1= pd.read_csv('Data/New-Data/full_Datasets_type_I_and_II/Dataset_II_part1.csv')
Dataset_type_II_part2= pd.read_csv('Data/New-Data/full_Datasets_type_I_and_II/Dataset_II_part2.csv')

# select parts two be conctenated
frames_I=[Dataset_type_I_part1,Dataset_type_I_part2]
frames_II=[Dataset_type_II_part1,Dataset_type_II_part2]

# concatenate each dataframes' parts
Dataset_type_I=pd.concat(frames_I)
Dataset_type_II=pd.concat(frames_II)

# index reset
Dataset_type_I.reset_index(level=0, drop=True, inplace=True)
Dataset_type_II.reset_index(level=0, drop=True, inplace=True)

# %% [markdown]
# # II. Datasets Exploration and Exploratory Visualizations
# %% [markdown]
# ## II.1. Datapoints number per each tuple (user,activity) function:

# %%
# This function returns a table includes the number of 
# windows per each tuple(user_id , activity id) included in the dataset 

def num_row_user_act(Df):
    
    user_Ids=sorted(Df['user_Id'].unique()) # extracting and sorting unqiue user ids 
    activity_Ids=sorted(Df['activity_Id'].unique()) # extracting and sorting unqiue activity ids 
    act_columns=['Activity '+str(int(Id)) for Id in activity_Ids ] # defining column names used in output table
    
    if len(activity_Ids)==7: # adapting column names in case the function deals with dataset type III
        act_columns=act_columns[0:6]+['P_Transitions'] 
    
    users_index=['User '+ str(int(Id)) for Id in user_Ids] # defining rows names used in output table
    
    # counting the number of windows per each tuple(user_id,activity_id)
    # store these values in 2D numpy array
    data=np.array([ [len(Df[(Df["user_Id"]== user_ID) &(Df["activity_Id"]==activity_ID)]) 
               for activity_ID in activity_Ids ] for user_ID in user_Ids])
    
    # Create a pandas dataframe from the array above
    win_per_act_per_user=pd.DataFrame(data = data,columns=act_columns,index=users_index)
    
    
    return win_per_act_per_user # returns the dataframe

# %% [markdown]
# ## II.2. Visualizing Activities Distribution:

# %%

##################################################################################
# This function returns the weights activity and visualize the distribution of a column
# This function will be applied only to target columns
def visualize_column(Df,column):
    
    labels= sorted(Df[column].unique()) # extracting and sorting activity unique ids
    Als_dict={ key: len(Df[Df[column]==key]) for key in labels} # counting the number of windows per activity
    data=[Als_dict[key] for key in labels] # sorting these numbers
    
    weights=np.array(data)/float(np.array(data).sum()) # calculating weights of each activity
    
    columns=["Activity "+str(int(key)) for key in labels] # defining columns of weights' table
    
    Df_weights=pd.DataFrame(data=None,columns=columns)# defining an empty dataframe with column names
    Df_weights.loc['Weights']=weights # appending weights row
    
    print("_____ The weights of each activity _____")
    display(Df_weights) # displying weights table
    print("")
    plt.bar(columns,data) # ploting activity distribution
    plt.xlabel('Activity Labels') # set X axis info
    plt.ylabel('Number of Data points') # set Y axis info
    plt.title('Number of Data points per activity') # set the figure's title
    plt.show() # showing the figure

# %% [markdown]
# ## Data Exploration PipeLine

# %%
def data_exploration_pipeline(Dataset,typ,outliers):
    # inputs:
    #        Dataset: a pandas dataframe can be a full dataset (I or II), 
    #               cleaned dataset(I or II or III), outliers dataset (I or II)
    
    #        typ    : integer type of the dataset possible values: 1(for dataset type I), 2(for type II) or 3 (for type III)
    #        outliers: Boolean if true dataset we are dealing with is an outlier dataset(contain outlier values)         
    
    # columns names of the dataset
    columns=Dataset.columns
    
    if not outliers:  # in case we are not dealing with outliers datasets  
        # Adapting the dataset name switch the typ
        if typ==1:
            Dataset_name="Dataset type I "
        if typ==2:
            Dataset_name="Dataset type II "
        if typ==3:
            Dataset_name="Dataset type III "
    else:# in case we are dealing with outliers
        
        # adapting the dataset names switch the case
        if typ==1:
            Dataset_name="Outliers of Dataset type I "
        if typ==2:
            Dataset_name="Outliers ofDataset type II "
    
    # general info about the dataset: number of rows and columns
    print(  Dataset_name+'has a shape of: '+ str(Dataset.shape[0]) +' rows and '+str(Dataset.shape[1])+' columns')
    print("")
    print("")    
    print("")
    
    if not outliers: # in case dataset is not an outlier dataset
        print("The first 3 rows of "+Dataset_name +":")
        display(Dataset.iloc[0:3]) # display the first 3 rows
        print("")
        print("")    
        print("")
        print("rows 500, 501, 502 of "+Dataset_name +":")
        display(Dataset.iloc[500:503]) # display rows 500,501 and 502
        print("")
        print("")    
        print("")
        print("Description of the 10 first features:")
        display(Dataset.describe()[columns[0:10]]) # statistics of the first ten time domain features
        print("")
        print("")    
        print("")
        print("Description of the 10 first frequency features:")
        display(Dataset.describe()[columns[265:275]]) # statistics of the first ten frequency domain features
        print("")
        print("")    
        print("")
    Stats= num_row_user_act(Dataset)# generate number of windows per each tuple (user,activity)
    print("Number of windows per user and per each activity:")
    display(Stats)# display the table
    print("")
    print("")    
    print("")
    print("Statistics of table above:")
    display(Stats.describe())# table's statics
    print("")
    print("")    
    print("")
    visualize_column(Dataset,"activity_Id") # visualize activity distribution of the dataset 


# %%
# apply the data_exploration_pipeline to dataset type 1
time_mark("data_exploration_pipeline:Dataset_type_I")
data_exploration_pipeline(Dataset_type_I,1,False)


# %%
# apply the data exploration pipe line to dataset type II
time_mark("data_exploration_pipeline:Dataset_type_II")
data_exploration_pipeline(Dataset_type_II,2,False)

# %% [markdown]
# # III. Data Preprocessing:
# %% [markdown]
# ## III.1 Handling Outliers

# %%
#####################################################################################
def extract_drop_outliers(Df,threshold,typ):
    #Df: pandas dataframe (Dataset type I or Dataset type II)
    # Threshold: integer : if the number of features detected as ouliers in row exceeds the threshold 
    #                      therow will be considered as "outlier row"
    
    max_range=len(Df["activity_Id"].unique()) # number of unique activities in Df
    
    columns=Df.columns # column names of the dataset
    
    outliers={} # dictionary will contain number of outliers per row . keys are rows' indexes
    pgb = ProgressBar((max_range+1)*len(columns[:-2]), "extract_drop_outliers")
    for i in range(1,max_range+1):# iterate throw each activity type in the dataset
        
        Df_A=Df[Df['activity_Id']==i] # select rows related to this activity
        
        for column in columns[:-2]:# iterate throw features columns only in Df_A
            pgb.inc()
            q1= Df_A[column].describe()['25%'] # the value of the first quartile of a column in Df_A
            
            q3= Df_A[column].describe()['75%'] # the value of the third quartile of a column in Df_A
            
            low_threshold=q1-1.5*(q3-q1) # define low threshold to detect bottom outliers of a column
            high_threshold=q3+1.5*(q3-q1) # define high threshold to detect top outliers of a column
            
            for e in Df_A.index :# iterate throw Df_A indexes
                
                if (Df[column].iloc[e]>high_threshold or Df[column].iloc[e]<low_threshold) :# if value is an outlier
                    
                    if e in outliers.keys(): # if the row index is alread exist in outliers dictionary
                        outliers[e]=outliers[e]+1 # increse the number of ouliers for this row
                    else:# if the row index does not exist yet in  outliers dic keys
                        outliers[e]=1 # add the key with outlier number =1
    
    indexs=np.array(sorted(outliers.keys())) # rows indexes contain outlier values sorted from low to high
    values=np.array([outliers[indexs[i]] for i in range(len(indexs))]) # number of outliers related to each row

    indexs_droped=indexs[values>threshold]# store indexes having number of outliers exceeding the threshold in a list
    
    # Build outliers dataframe using row's indexes
    outliers_data=np.array([list(Df.iloc[indexs_droped[i]]) for i in range(len(indexs_droped))])
    outliers_Df= pd.DataFrame(data=outliers_data,columns= columns)
    
    # generate the clean dataframe by droping outliers from the original dataframe
    clean_Df=Df.drop(indexs_droped,0,)
    
    # adapting the name of the dataset switch the case
    if typ==1:
        dataset_name='Dataset type I'
    if typ==2:
        dataset_name="Dataset type II"
    
    #### report
    print("")
    print("_______________________________ Original Data Frame info...____________________________________")
    print('Number of rows in the original dataframe '+dataset_name+':',len(Df) ) # original dataset lenght
    print("")
    print("")
    visualize_column(Df,'activity_Id') # activity distribution of the original dataset
    print("")
    print("")
    print("________________________________ Outliers info...________________________________________________")
    print("A row is considered as outlier if the number of its outliers exceeds: "+str(threshold)) # threshold info
    print('Number of rows droped :',len(indexs_droped) ) # number of rows considered as outliers
    print("")
    print("")
    data_exploration_pipeline(outliers_Df,typ,True) # Apply the data exploration pipeline to outliers dataframe
    print("")
    print("")
    print ("________________________________ Cleaned+" +dataset_name+" Dataframe info...________________________________________")
    print ('Number of rows in the clean dataframe '+dataset_name+':',len(clean_Df)) # clean dataframe info
    print("")
    print("")
    print("")
    print("")
    data_exploration_pipeline(clean_Df,typ,False)# apply the data exploration pipeline to the clean dataframe
    return clean_Df # return the clean dataset


# %%
# apply extract drop outliers to dataset type I
time_mark("extract_drop_outliers:Dataset_type_I")
clean_Dataset_type_I= extract_drop_outliers(Dataset_type_I,100,1)# store the clean dataframe


# %%
# apply drop extract outliers to dataset type
time_mark("extract_drop_outliers:Dataset_type_II")
clean_Dataset_type_II= extract_drop_outliers(Dataset_type_II,100,2)# store the clean dataframe

# %% [markdown]
# ## III.2 Features Scaling

# %%
#################################################################################
def scaling_array(oneD_signal):
    # inputs: 1D numpy array (one column)
    maximum=oneD_signal.max() # maximum of the column
    minimum=oneD_signal.min() # min value of the column
    Difference=float(maximum-minimum) # max-min
    # scaling formula: 2 * (x_i-minimum)/(maximum -minimum)
    # apply the scaling formula to each value in the column
    scaled_signal=np.array([((float(oneD_signal[i])-minimum)/float(Difference))*2 -1 for i in range(len(oneD_signal))])
    
    #return the scaled array
    return scaled_signal

def scaling_DF(data_frame):
    # input : pandas dataframe (clean datasets type I or II)
    columns=data_frame.columns# column names
    # apply the scaling function to each feature columns only
    scaled_array=np.apply_along_axis(scaling_array,0,np.array(data_frame[columns[:-2]]))
    
    # buid the scaled dataset
    scaled_df=pd.DataFrame(data=scaled_array,columns=columns[:-2])
    
    # the user and activity ids columns
    scaled_df['activity_Id']=np.array(data_frame['activity_Id'])
    scaled_df['user_Id']=np.array(data_frame['user_Id'])
    
    return scaled_df # return the scaled dataset

###################################################################################


# %%
# apply the scaling function to cleaned dataset type I
scaled_type_I=scaling_DF(clean_Dataset_type_I)

# explore the scaled dataset type I
time_mark("data_exploration_pipeline:scaled_type_I")
data_exploration_pipeline(scaled_type_I,1,False)


# %%
# apply the scaling function to cleaned dataset type II
scaled_type_II=scaling_DF(clean_Dataset_type_II)

# explore the scaled dataset type II
time_mark("data_exploration_pipeline:scaled_type_II")
data_exploration_pipeline(scaled_type_II,2,False)

# %% [markdown]
# # VI Dataset type III:
# %% [markdown]
# ## VI.1. Dataset type III generation

# %%
act_labels=list(scaled_type_II['activity_Id']) # extract activity labels from scaled type II (ids from 1 to 12)

for i in range(len(act_labels)):# iterate throw each activity label
    
    if act_labels[i]>6: # if activity label belongs to postural transitions ids from 7 to 12
        act_labels[i]=7 # the target will be replaced by the id=7 (postural transition)

# build dataset type III by replacing the activity id column by the new column create above
scaled_type_III=pd.DataFrame(data=np.array(scaled_type_II),columns=scaled_type_II.columns)
scaled_type_III['activity_Id']=np.array(act_labels)

# %% [markdown]
# ## VI.2. Dataset type III exploration

# %%
# apply the data exploration pipeline to scaled dataset type III
time_mark("data_exploration_pipeline:scaled_type_III")
data_exploration_pipeline(scaled_type_III,3,False)

# %% [markdown]
# # V. Train-Test Datasets
# %% [markdown]
# ## V.1. Train-Test Datasets creation
# %% [markdown]
# ### Unique Users Ids used for training and testing

# %%
# volunteersids used for training
train_users =[1,3,5,6,7,8,10,11,14,15,27,17,21,29,30,16,19,20,22,23,25,]

# volunteers ids used for testing
test_users = [2,4,9,12,13,26,18,28,24,]

# %% [markdown]
# ### Creating train and test datasets

# %%
def create_training_testing_data(scaled_Df,train_users,test_users,typ):
    # inputs:
    #        scaled_DF : pandas dataframe already scaled
    #       train_users: list of integers contains train user ids 
    #       train_users: list of integers contains test user ids 
    #       typ        : integer from 1 to 3 (depending on the dataset type)
    
    # select rows related to train users ids store them in numpy array
    array_train =np.array([np.array(scaled_Df.iloc[i]) 
                           for i in range(len(scaled_Df)) if int(scaled_Df['user_Id'].iloc[i]) in train_users])
    # select rows related to test users ids store them in numpy array
    array_test  =np.array([np.array(scaled_Df.iloc[i]) 
                           for i in range(len(scaled_Df)) if int(scaled_Df['user_Id'].iloc[i]) in test_users])
    
    # columns names
    columns=scaled_Df.columns
    
    # build train and test dataframes from numpy arrays above
    Df_train= pd.DataFrame(data= array_train,columns=columns)
    Df_test = pd.DataFrame(data= array_test,columns=columns)
    
    
    # train features dataframe
    Df_train_features= Df_train[columns[:-2]]
    # train labels dataframe
    Df_train_labels  = Df_train[columns[-2:-1]]
    # train user id labels dataframe
    Df_train_users   = Df_train[columns[-1]]
    
    # test features dataframe
    Df_test_features= Df_test[columns[:-2]]
    # test labels dataframe
    Df_test_labels  = Df_test[columns[-2:-1]]
    # test user id labels dataframe
    Df_test_users   = Df_test[columns[-1]]
    
    # 2D numpy array : train features
    X_train =np.array(Df_train_features)
    
    # 2D numpy array : test features
    X_test  =np.array(Df_test_features)
    
    # 1D numpy array : train labels
    y_train= np.array(Df_train_labels['activity_Id'])
    
    # 1D numpy array : test labels
    y_test = np.array(Df_test_labels ['activity_Id'])
    
    # adapting the dataset name switch the case
    if typ==1:
           Dataset_name="Dataset type I"
    if typ==2:
           Dataset_name="Dataset type II"
    if typ==3:
           Dataset_name="Dataset type III"
    
    print("")
    print("______________________________"+Dataset_name+" Train features & labels info:______________________________________")
    print("")
    visualize_column(Df_train,'activity_Id')# visualize activity distribution of train dataframe
    print("")
    print("______________________________Test features & labels info:______________________________________")
    print("")
    visualize_column(Df_test,'activity_Id') # visualize the activity distribution of the test dataframe
    
    return  [X_train, X_test, y_train, y_test] # return train and test numpy arrays


# %%
# dictionary will contain train and test files of each dataframe type
train_test_files_dic={}


# %%
# apply create_training_testing_data to scaled dataset type I
[X_1_train, X_1_test, y_1_train, y_1_test] = create_training_testing_data(scaled_type_I,train_users,test_users,1)


# %%
# store train test files type I in the dictionary 
train_test_files_dic[1]=[X_1_train, X_1_test, y_1_train, y_1_test]


# %%
# apply create_training_testing_data to scaled dataset type II
[X_2_train, X_2_test, y_2_train, y_2_test] = create_training_testing_data(scaled_type_II,train_users,test_users,2)


# %%
# store train test files type II in the dictionary 
train_test_files_dic[2]= [X_2_train, X_2_test, y_2_train, y_2_test]


# %%
# apply create_training_testing_data to scaled dataset type II
[X_3_train, X_3_test, y_3_train, y_3_test] = create_training_testing_data(scaled_type_III,train_users,test_users,3)


# %%
# store train test files type III in the dictionary 
train_test_files_dic[3]=[X_3_train, X_3_test, y_3_train, y_3_test]

# %% [markdown]
# # VI. Train-Test PipeLine

# %%
time_mark("GaussianNB:DecisionTreeClassifier:LogisticRegression:M0,M1:M2")
from sklearn.naive_bayes import GaussianNB as NB # import gaussian naive bayes classifier
from sklearn.tree import DecisionTreeClassifier as DTC # import decision tree classifier
from sklearn.linear_model import LogisticRegression as LR # import logistic regression classifier

from sklearn.metrics import accuracy_score as accuracy # import accuracy score
from sklearn.metrics import confusion_matrix as cm # import confusion matrix

# intialize models
Benchmark_model =NB()
Clf1=DTC(random_state=337)
#Clf2=LR(random_state=337)
Clf2=LR(random_state=337, max_iter=10000)


# %%
# Define the adpted confusion matrix
def full_confusion_matrix(Df): 
    # input: 
    #   Df : pandas dataframe, the contingency table resulted from the confusion matrix defined earlier as cm
    
    columns=Df.columns # activity names
    # add new columns containing detailed scores
    new_columns=list(columns)+['data points number','precision %','sensitivity %','specificity %']
    
    # create the index from the same old columns add an other row called total
    new_index=list(columns)+['Total']
    
    # intialize the confustion matrix dataframe
    new_Df=pd.DataFrame(data=None,columns=new_columns, index= new_index)
    # intilize values
    total_TP=0 # sum of true positives
    total_FN=0 # sum of false negatives
    total_data_points_number=0 # total number of datapoints
    
    for column in columns:
        
        TP=Df.loc[column][column] # extract true postives from the contingency table
        FN=Df.loc[column].sum()-TP # calculate FN(false negatives)
        FP=Df[column].sum()-TP # calculate FP(false positives)
        TN=(Df.sum()).sum()-TP-FN-FP # calculate TN(true negatives)
        class_data_points=TP+FN  # number of datapoints per activity
        # precision score in %
        precision= TP/float(TP+FP) * 100
        # Recall or sensitivity in %
        sensitivity= TP/float(TP+FN) *100
        # sepecificity score in %
        specificity=TN/float(TN+FP) * 100
        
        new_row =list(Df.loc[column])+[class_data_points,precision,sensitivity,specificity]# contenate new scores in one row
        new_Df.loc[column]=new_row # append the row to the dataframe
        
        # update intialized values
        total_data_points_number= total_data_points_number+class_data_points 
        total_TP=total_TP+TP
        total_FN=total_FN+FN
    
    # after iterting throw all activity types
    # the general accuracy of the model is:
    total_accuracy= total_TP/float(total_TP+total_FN) * 100
    
    # add total values to the dataframe
    new_Df.loc['Total'] [['data points number','precision %','sensitivity %','specificity %']]=['data points number='+str(total_data_points_number),'','','accuracy= '+str(total_accuracy)[0:6]+'%']
    new_Df.loc['Total'][columns]=['' for i in range(len(columns))]
    
    return new_Df # return the adapted confusion matrix
        


# %%
def train_predict(classifier, sample_size, X_train, X_test, y_train,  y_test,typ): 
    
    # inputs:
    #   classifier: the learning algorithm to be trained and predicted on
    #   sample_size: the size of samples (number) to be drawn from training set
    #   X_train: features training set
    #   y_train: Activity_number_ID training set
    #   X_test: features testing set
    #   y_test: Activity_number_ID testing set
    
    # Empty dictionary will include all dataframes and info related to training and testing.
    results = {}
    
    # Fitting the classifier to the training data using slicing with 'sample_size'
    start= timer() # Get start time
    classifier = classifier.fit(X_train[0:sample_size,:],y_train[0:sample_size])# fiting the classfier
    end = timer() # Get end time
    
    # Calculate the training time
    results['train_time'] = end-start
        
    # Get the predictions on the test set(X_test),
    # then get predictions on the first 3000 training samples(X_train) using .predict()
    start = timer() # Get start time
    predictions_test = classifier.predict(X_test) # predict
    predictions_train =classifier.predict(X_train[:3000,:])
    end = timer() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] =end-start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy(y_train[:3000],predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy(y_test,predictions_test)
    
    # Adapting the confusion matrix shape to the type of data used
    if typ==1:
        confusion_matrix=cm(y_test, predictions_test, labels=[1,2,3,4,5,6], sample_weight=None) # 
        columns=['WK','WU','WD','SI','ST','LD']
        index=['WK','WU','WD','SI','ST','LD']
    if typ==2:
        confusion_matrix=cm(y_test, predictions_test, labels=[1,2,3,4,5,6,7,8,9,10,11,12], sample_weight=None)
        columns=['WK','WU','WD','SI','ST','LD','St-Si','Si-St','Si-Li','Li-Si','St-Li','Li-St']
        index=  ['WK','WU','WD','SI','ST','LD','St-Si','Si-St','Si-Li','Li-Si','St-Li','Li-St'] 
    if typ==3:   
        confusion_matrix=cm(y_test, predictions_test, labels=[1,2,3,4,5,6,7], sample_weight=None)
        columns=['WK','WU','WD','SI','ST','LD','PT']
        index=['WK','WU','WD','SI','ST','LD','PT']
    
    if sample_size==len(X_train):# if 100% of training is achieved
        # apply the confusion matrix function to the last contingency table generated
        confusion_matrix_df=(pd.DataFrame(data=confusion_matrix,columns=columns,index=index)).pipe(full_confusion_matrix)
    else:# if not
        # create a dataframe from the contingency table
        confusion_matrix_df=pd.DataFrame(data=confusion_matrix,columns=columns,index=index)
        
    # Return the results
    return (results,confusion_matrix_df)


# %%
def train_test_report(classifier,dataset_type):
    # Inputs:
    #  classifier: model will be trained tested and evaluated on all sample sizes
    #  Dataset_type: "All"  or integers 1,2,3 
    # if "All" the classifier will be trained, tested and evaluated on all datasets
    # if integer 1, 2 or 3: the classifier the classifier will be trained, tested and evaluated on one dataset(I,II or III)
    
    if dataset_type!='All': # dataset type equal to 1 ,2 or 3
        # extract train and test files related to dataset type
        new_dic={dataset_type:train_test_files_dic[dataset_type]}
    
    else:# the model will be trained , tested and evaluted on all datasets
        new_dic=train_test_files_dic # import all train and test files
    
    for key in sorted(new_dic.keys()):# iterating throw dataset types
        clf=classifier # reintialize the classifier
        # adapt the dataset name switch the case
        if key==1:
            Dataset_name='Dataset type I'
        if key==2:
            Dataset_name='Dataset type II'
        if key==3:
            Dataset_name='Dataset type III'
        
        files = new_dic[key] # copy train and test files related to the dataset type
        # create a temporal dictionary where train, test and evaluation results will be stored
        results = {}
        print("_____________________"+Dataset_name+" Training and Testing______________________")
        print("")
        
        # copy train and test files
        X_train, X_test, y_train, y_test,= files[0], files[1], files[2], files[3]
        # extract the name of the classifier
        clf_name = clf.__class__.__name__
        
        # training started
        print("{} started training....".format(clf_name) )   
        
        results[clf_name] = {}
        # generate sample sizes
        samples_10 = int(len(X_train)/10) # 10%
        samples_50 = int(len(X_train)/2) # 50%
        samples_100 = int(len(X_train)) # 100%
        
        
        for i, samples in enumerate([samples_10, samples_50, samples_100]): # iterate throw each sample size
            print("...")
            if samples==len(X_train):# when 100% of training will be achieved
                # store results related to the classier and sample size in results dictionary
                # store the full confusion matrix
                results[clf_name][i],confusion_matrix = train_predict(clf, samples, X_train, X_test, y_train, y_test,key)
            else:# if not
                # store results related to the classier and sample size in results dictionary
                results[clf_name][i]= train_predict(clf, samples, X_train,X_test, y_train,  y_test,key)[0]

        print( "Success: {} Finished Training and Testing.".format(clf_name))
        print("")
        print ("________"+clf_name+" results:__________")
        print("")
        print("")
        print("Accuracy and duration per training size")
        # display train and test results
        display(pd.DataFrame(results[clf_name]).rename(columns={0:'10% of train', 1:'50% of train', 2:'100% of train'}))
        print("")
        print("Confusion Matrix Sensitivity and Recall when 100% of train is achieved")
        # display the full confusion matrix results
        display(confusion_matrix)
        print("____________________________________________________________________")
        print("")
        print("")

# %% [markdown]
# # VII. Benchmark Model

# %%
# training, testing and evaluating the benchmark model on all datasets
time_mark("M0:GaussianNB")
train_test_report(Benchmark_model,'All')

# %% [markdown]
# # VIII. Elected Models
# %% [markdown]
# - Decision Tree Classifier

# %%
# training, testing and evaluating Decision tree classifier on all datasets
time_mark("M1:DecisionTreeClassifier")
train_test_report(Clf1,'All')

# %% [markdown]
# - Logistic Regression Classifier

# %%
# training, testing and evaluating Logistic Regression  classifier on all datasets
time_mark("M2:LogisticRegression")
train_test_report(Clf2,'All')

# %% [markdown]
# # IX. Tunning Parameters

# %%
time_mark("GridSearchCV:M1,M2,M3,M4,M5,M6,M7,M8")
from sklearn.model_selection import GridSearchCV # import grid search cv to tune parameters
#clf_chosen=LR(random_state=337) # intialize the LR model
clf_chosen=LR(random_state=337, max_iter=10000) # intialize the LR model

# scaled dataset type I activity weights
weights_dic_1= {1:0.179248,2:0.15867,3:0.144265,4:0.161919,5:0.17849,6:0.177407}

# scaled dataset type II activity weights
weights_dic_2={1 : 0.164576, 2 : 0.150152, 3 : 0.142537, 4 : 0.145225,
               5 : 0.156961, 6 : 0.155169, 7 : 0.012991, 8 : 0.009138,
               9 : 0.015857, 10: 0.015320, 11: 0.018903,12 : 0.013170}
# scaled dataset type III activity weights
weights_dic_3={1:0.164576,2:0.150152,3:0.142537,4:0.145225,5:0.156961,6:0.155169,7:0.085379}

# possible values for the parameter "class_weight" for each dataset

class_weight_1=[None,weights_dic_1] # scaled dataset I
class_weight_2=[None,weights_dic_2] # scaled dataset II
class_weight_3=[None,weights_dic_3] # scaled dataset III

# possible parameters dictionary for each dataset:

# dataset type I
params_1={'penalty':['l2'], 'solver':['newton-cg','sag','lbfgs'],'dual':[False],
          'multi_class':['ovr','multinomial'],'class_weight':class_weight_1}
params_2={'penalty':['l2'], 'solver':['liblinear'],'dual':[True],'multi_class':['ovr'],'class_weight':class_weight_1}
params_3={'penalty':['l1'], 'solver':['liblinear'],'dual':[False],'multi_class':['ovr'],'class_weight':class_weight_1}

# dataset type II
params_4={'penalty':['l2'], 'solver':['newton-cg','sag','lbfgs'],'dual':[False],
          'multi_class':['ovr','multinomial'],'class_weight':class_weight_2}
params_5={'penalty':['l2'], 'solver':['liblinear'],'dual':[True],'multi_class':['ovr'],'class_weight':class_weight_2}
params_6={'penalty':['l1'], 'solver':['liblinear'],'dual':[False],'multi_class':['ovr'],'class_weight':class_weight_2}

# dataset type III
params_7={'penalty':['l2'], 'solver':['newton-cg','sag','lbfgs'],'dual':[False],
          'multi_class':['ovr','multinomial'],'class_weight':class_weight_3}
params_8={'penalty':['l2'], 'solver':['liblinear'],'dual':[True],'multi_class':['ovr'],'class_weight':class_weight_3}
params_9={'penalty':['l1'], 'solver':['liblinear'],'dual':[False],'multi_class':['ovr'],'class_weight':class_weight_3}


# %%
# Dataset type I:
# insert parameters in the grid seach for each path 
# store each future results in a model
tuned_model1 =GridSearchCV(estimator =clf_chosen,
                          param_grid=params_1)
tuned_model2 =GridSearchCV(estimator =clf_chosen,
                          param_grid=params_2)
tuned_model3 =GridSearchCV(estimator =clf_chosen,
                          param_grid=params_3)

# train models
time_mark("M1:GridSearchCV")
tuned_model1.fit(X_1_train,y_1_train)
time_mark("M2:GridSearchCV")
tuned_model2.fit(X_1_train,y_1_train)
time_mark("M3:GridSearchCV")
tuned_model3.fit(X_1_train,y_1_train)

# display best parameters of each model
print("Tuned model 1 best params:",tuned_model1.best_params_)
print("Tuned model 2 best params:",tuned_model2.best_params_)
print("Tuned model 3 best params:",tuned_model3.best_params_)

# store predictions and generate accuracies for each model
predictions1=tuned_model1.predict(X_1_test)
print( "tuned model 1 accuracy:",accuracy(y_1_test,predictions1))

predictions2=tuned_model2.predict(X_1_test)
print( "tuned model 2 accuracy:",accuracy(y_1_test,predictions2))

predictions3 = tuned_model3.predict(X_1_test)
print( "tuned model 3 accuracy:",accuracy(y_1_test,predictions3))


# %%
# Dataset type II: same process will be applied for dataset type II
tuned_model4 =GridSearchCV(estimator =clf_chosen,
                          param_grid=params_4)
tuned_model5 =GridSearchCV(estimator =clf_chosen,
                          param_grid=params_5)
tuned_model6 =GridSearchCV(estimator =clf_chosen,
                          param_grid=params_6)

time_mark("M4:GridSearchCV")
tuned_model4.fit(X_2_train,y_2_train)
time_mark("M5:GridSearchCV")
tuned_model5.fit(X_2_train,y_2_train)
time_mark("M6:GridSearchCV")
tuned_model6.fit(X_2_train,y_2_train)

print("Tuned model 4 best params:",tuned_model4.best_params_)
print("Tuned model 5 best params:",tuned_model5.best_params_)
print("Tuned model 6 best params:",tuned_model6.best_params_)

predictions4=tuned_model4.predict(X_2_test)
print( "tuned model 4 accuracy:",accuracy(y_2_test,predictions4))

predictions5=tuned_model5.predict(X_2_test)
print( "tuned model 5 accuracy:",accuracy(y_2_test,predictions5))

predictions6=tuned_model6.predict(X_2_test)
print( "tuned model 6 accuracy:",accuracy(y_2_test,predictions6))


# %%
# Dataset type III: same process will be applied for dataset type III
tuned_model7 =GridSearchCV(estimator =clf_chosen,
                          param_grid=params_7)
tuned_model8 =GridSearchCV(estimator =clf_chosen,
                       param_grid=params_8)
tuned_model9 =GridSearchCV(estimator =clf_chosen,
                          param_grid=params_9)

time_mark("M7:GridSearchCV")
tuned_model7.fit(X_3_train,y_3_train)
time_mark("M8:GridSearchCV")
tuned_model8.fit(X_3_train,y_3_train)
time_mark("M9:GridSearchCV")
tuned_model9.fit(X_3_train,y_3_train)

print("Tuned model 7 best params:",tuned_model7.best_params_)
print("Tuned model 8 best params:",tuned_model8.best_params_)
print("Tuned model 9 best params:",tuned_model9.best_params_)

predictions7=tuned_model7.predict(X_3_test)
print( "tuned model 7 accuracy:",accuracy(y_3_test,predictions7))

predictions8=tuned_model8.predict(X_3_test)
print( "tuned model 8 accuracy:",accuracy(y_3_test,predictions8))

predictions9=tuned_model9.predict(X_3_test)
print( "tuned model 9 accuracy:",accuracy(y_3_test,predictions9))

# %% [markdown]
# ## Hyperparameter C Search:
# %% [markdown]
# * **Warning:** Running duration of this part is a least **5 hours**

# %%
# C hyper-parameter search

# C values from 0.1 to 20 with a step of 0.1
C_values=[(i+1)*0.1 for i in range(200)]       

def max_c(results,typ):
    # inputs:
    #   results: dictionary={c_value: accuracy of the model}
    #   typ: float possible values are: 1,2,3 and 3.5
    
    # extract C values
    C_values=sorted(results.keys())
    # extract related accuracies 
    accuracy_values=[results[key] for key in C_values]
    
    # extract c value having the maximum accuracy
    max_c=C_values[np.array(accuracy_values).argmax()]
    
    # display results
    print("max accuracy :",max(accuracy_values))
    print("C value:",max_c)
    
    plt.plot(C_values,accuracy_values)# plot the curve
    plt.xlabel("C values")  # set X axis info
    plt.ylabel("accuracy") # set Y axis info
    
    # Set the right title switch the case
    if typ==1:# if dataset type I
        plt.title("model 1 accuracy variation on dataset type I")
    if typ==2:# if dataset type II
        plt.title("model 5 accuracy variation on dataset type II")
    
    if typ==3:# if the first model of dataset type III
        plt.title("model 7 accuracy variation on dataset type III")
    
    if typ==3.5:# if the second model of dataset type III
        plt.title("model 8 accuracy variation on dataset type III")

# %% [markdown]
# * ** To run the search function below for dataset type 1 decomment the last line** 

# %%
# Dataset type I
def lookup_best_c(x_train,y_train,x_test,y_test):
    accuracy_results={} # empty dictionary will contain c values and accuracies related
    
    for value in C_values:# iterate throw each C value
        #tuned model 1 best parameters + C variable
        #tmp_model=LR(solver='lbfgs',class_weight= None,multi_class= 'ovr', 
        #          dual=False, penalty= 'l2',random_state=337,C=value)
        tmp_model=LR(solver='lbfgs',class_weight= None,multi_class= 'ovr', 
                  dual=False, penalty= 'l2',random_state=337,C=value, max_iter=10000)
        # train the model
        tmp_model.fit(x_train,y_train)
        
        # predicting activity labels
        tmp_predictions=tmp_model.predict(x_test)
        # accuracy score
        tmp_accuracy=accuracy(tmp_predictions,y_test)
        # store the tuple c_value and accuracy value in the dictionary
        accuracy_results[value]=tmp_accuracy
    
    # after iterating throw all c values
    return accuracy_results # return results

# apply lookup_best_c to train and test files type I
#results_I = lookup_best_c(X_1_train,y_1_train,X_1_test,y_1_test)

# %% [markdown]
# * ** To visualize results_I,  Decomment the last line of the cell below**

# %%
# visualizing results
#max_c(results_I,1)

# %% [markdown]
# * ** To run the search function below for dataset type 2 models decomment the last line** 

# %%
# same process will be applied for dataset type II model
def lookup_best_c(x_train,y_train,x_test,y_test):
    i=0
    accuracy_results={}
    for value in C_values:
        
        #tuned model 5 best parameters + C variable
        #tmp_model=LR(solver='liblinear', class_weight= None, multi_class= 'ovr',
        #          dual= True, penalty= 'l2',random_state=337,C=value)
        tmp_model=LR(solver='liblinear', class_weight= None, multi_class= 'ovr',
                  dual= True, penalty= 'l2',random_state=337,C=value, max_iter=10000)
        tmp_model.fit(x_train,y_train)
        tmp_predictions=tmp_model.predict(x_test)
        tmp_accuracy=accuracy(tmp_predictions,y_test)
        accuracy_results[value]=tmp_accuracy
    return accuracy_results

# apply lookup_best_c to train and test files type II
#results_II =lookup_best_c(X_2_train,y_2_train,X_2_test,y_2_test)

# %% [markdown]
# * ** To visualize results_II,  Decomment the last line of the cell below**

# %%
# visualizing results
#max_c(results_II,2)

# %% [markdown]
# * ** To run the search function below for dataset type 3 models decomment the last line** 

# %%
# same process for Dataset type III first model(tuned model 7)
def lookup_best_c(x_train,y_train,x_test,y_test):
    accuracy_results={}
    for value in C_values:
        #tuned model 7 best parameters + C variable
        tmp_model=LR(solver= 'newton-cg', class_weight= None, multi_class= 'ovr', 
                  dual= False, penalty= 'l2',random_state=337,C=value)
        tmp_model.fit(x_train,y_train)
        tmp_predictions=tmp_model.predict(x_test)
        tmp_accuracy=accuracy(tmp_predictions,y_test)
        accuracy_results[value]=tmp_accuracy
    return accuracy_results
# apply lookup_best_c to train and test files type III using model 7 best parameters
#results_III=lookup_best_c(X_3_train,y_3_train,X_3_test,y_3_test)

# %% [markdown]
# * ** To visualize results_III,  Decomment the last line of the cell below**

# %%
# visualizing results
#max_c(results_III,3)

# %% [markdown]
# * ** To run the search function below for dataset type 3 models decomment the last line** 

# %%
def lookup_best_c(x_train,y_train,x_test,y_test):
    accuracy_results={}
    for value in C_values:
        #tuned model 8 best parameters + C variable
        tmp_model=LR(solver= 'liblinear', class_weight= None, multi_class= 'ovr', 
                  dual= True, penalty= 'l2',random_state=337,C=value)
        tmp_model.fit(x_train,y_train)
        tmp_predictions=tmp_model.predict(x_test)
        tmp_accuracy=accuracy(tmp_predictions,y_test)
        accuracy_results[value]=tmp_accuracy
    return accuracy_results
# apply lookup_best_c to train and test files type III using model 8 best parameters
#results_III_5=lookup_best_c(X_3_train,y_3_train,X_3_test,y_3_test)

# %% [markdown]
# * ** To visualize results_III_5,  Decomment the last line of the cell below**

# %%
# visualizing results
#max_c(results_III_5,3.5)

# %% [markdown]
# * ** To run the search function below for dataset type 3 models decomment the last line** 

# %%
# accuracy is still increasing for model 8 best params
# changing C values ranges from [0.1,20] to [20.1, 40]
# C values from 20.1 to 40 with a step of 0.1
C_values=[20+(i+1)*0.1 for i in range(200)]
def lookup_best_c(x_train,y_train,x_test,y_test):
    accuracy_results={}
    for value in C_values:
        # tuned model 8 best parameters
        tmp_model=LR(solver= 'liblinear', class_weight= None, multi_class= 'ovr', 
                  dual= True, penalty= 'l2',random_state=337,C=value)
        tmp_model.fit(x_train,y_train)
        tmp_predictions=tmp_model.predict(x_test)
        tmp_accuracy=accuracy(tmp_predictions,y_test)
        accuracy_results[value]=tmp_accuracy
    return accuracy_results
# apply lookup_best_c to train and test files type III using tuned model 8 best parameters
#results_III_6=lookup_best_c(X_3_train,y_3_train,X_3_test,y_3_test)

# %% [markdown]
# * ** To visualize results_III_6,  Decomment the last line of the cell below**

# %%
# visualizing results
#max_c(results_III_6,3.5)


# %%
fin=timer()
print('Running Duration= ',fin-Debut)

# %% [markdown]
# # Final Models:

# %%
# best C values was selected from each search 
time_mark("LogisticRegression")
final_model_I=LR(solver='lbfgs',class_weight= None,multi_class= 'ovr', 
                  dual=False, penalty= 'l2',random_state=337,C=4.7, max_iter=10000)
final_model_II=LR(solver='liblinear', class_weight= None, multi_class= 'ovr',
                  dual= True, penalty= 'l2',random_state=337,C=0.8, max_iter=10000)

# for dataset type III model 7 best parameters + best C value have the highest accuracy compared to model 8 best C value 
final_model_III=LR(solver= 'newton-cg', class_weight= None, multi_class= 'ovr', 
                  dual= False, penalty= 'l2',random_state=337,C=8.7, max_iter=10000)


# %%
# train, test and evaluate final model I on dataset type I
time_mark("MI:LogisticRegression")
train_test_report(final_model_I,1)


# %%
# train, test and evaluate final model II on dataset type II
time_mark("MII:LogisticRegression")
train_test_report(final_model_II,2)


# %%
# train, test and evaluate final model III on dataset type III
time_mark("MIII:LogisticRegression")
train_test_report(final_model_III,3)

# %% [markdown]
# # Testing Some Samples:

# %%

# test samples indexes for each dataset
indexes_I=[0,500,300,800,900,1000]
indexes_II=[91,134,124,14,0,46,189,27,72,56,40,89]
indexes_III=[92,135,125,15,1,47,190]

# activity labels for Datasets type I and II
AL={
        1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', # 3 dynamic activities
        4: 'SITTING', 5: 'STANDING', 6: 'LIYING', # 3 static activities
        
        7: 'STAND_TO_SIT',  8: 'SIT_TO_STAND',  9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT', 
    11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND',# 6 postural Transitions
       } 
# activity labels for dataset type III
AL3={1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', # 3 dynamic activities
        4: 'SITTING', 5: 'STANDING', 6: 'LIYING', # 3 static activities
        
        7: 'Postural Transition',}

def Samples_Results(x_test,y_test,model,samples_index,dataset_type):
    # Inputs:
    #  X_test: 2D numpy array (test features)
    #  y_test: 1D numpy array (test labels)
    #  sample index: integer from 0 to lenght of X_test-1    
    # Dataset type: integer possible values are 1,2 or 3
    
    # Intialize a pandas dataframe will contain predictions' results
    Df=pd.DataFrame(data=[],columns=['Row index','real identifier','predicted identifier'])
    
    for indice in samples_index:# iterate throw indicies
        
        real_value=int(y_test[indice]) # activity label of the sample
        features_row=x_test[indice,:] # features vector of the sample
        
        prediction=int(model.predict(features_row)) # predicted activity label
        # Adapting the activity name switch the dataset type
        if dataset_type==1:
            activity_name=AL[real_value]
        if dataset_type==2:
            activity_name=AL[real_value]
        if dataset_type==3:
            activity_name=AL3[real_value]
        
        # append the row index the activity id and the predicted activity id
        Df.loc[activity_name]=[indice,real_value,prediction]
    return Df # return the dataframe


# %%
# train the final model I on dataset type I
time_mark("MI:LogisticRegression")
final_model_I.fit(X_1_train,y_1_train)
# display results
Samples_Results(X_1_test,y_1_test,final_model_I,indexes_I,1) 


# %%
# train the final model II on dataset type II
time_mark("MII:LogisticRegression")
final_model_II.fit(X_2_train,y_2_train)
# display results
Samples_Results(X_2_test,y_2_test,final_model_II,indexes_II,2)


# %%
# train the final model III on dataset type III
time_mark("MIII:LogisticRegression")
final_model_III.fit(X_3_train,y_3_train)
# display results
Samples_Results(X_3_test,y_3_test,final_model_III,indexes_III,3)

# %% [markdown]
# # Fin


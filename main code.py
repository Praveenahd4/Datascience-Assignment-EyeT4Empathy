#!/usr/bin/env python
# coding: utf-8

# # Eloring Data

# In[1]:


#Importing Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

#Filtering Warnings
warnings.filterwarnings('ignore')

#Importing machine learning libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


# In[153]:


#Reading CSV files into DataFrames
df = pd.read_csv("EyeT/EyeT_group_dataset_III_image_name_letter_card_participant_11_trial_0.csv")
df_1 = pd.read_csv("EyeT/EyeT_group_dataset_III_image_name_letter_card_participant_11_trial_1.csv")
df_questionnaire_1 = pd.read_csv("Questionnaire_datasetIA.csv",encoding='latin-1')


# In[3]:


#Printing the shape of the DataFrame
print( df.shape)
#Comparing columns of two DataFrames
list(df.columns) == list(df_1.columns)


# In[4]:


#Displaying the first 5 rows of the DataFrame
df.head()


# In[5]:


# Check for null values.
for i in df.columns:
    print(i,":",df[i].isnull().sum()/df.shape[0]*100)


# In[6]:


#check for Data type
df.info()


# In[7]:


#loading the taget value file
df_questionnaire_1.head()


# In[8]:


df.describe()


# In[9]:


#Creating histogram plots for the columns in the DataFrame
df.hist(bins=60, figsize=(20, 15));


# # Data Pre-procesing
# * some of the columns that may not be necessary for model training.
#    * Unnamed: 0: This appears to be an index column and is not relevant for the analysis.
#    * Sensor: This column may not be relevant for the model as it is not clear what kind of sensor it is.
#    * Project name, Export date, Participant name, Recording name, Recording date, Recording date UTC, Recording start time, Recording start time UTC, Recording duration, Timeline name, Recording Fixation filter name, Recording software version: These columns may not be directly related to the gaze behavior and may not be necessary for model training.
#    * Recording resolution height, Recording resolution width, Recording monitor latency: These columns may not be directly related to the gaze behavior beacuse their std is zero and may not be necessary for model training.
#    * Event and Event value: These columns may not be relevant for predicting gaze behavior.
#    * Presented Stimulus name, Presented Media name: These columns may not be necessary for predicting gaze behavior unless the stimuli or media presented to the participant is of interest.
#    * Eye movement type, Eye movement type index: These columns may not be relevant for predicting gaze behavior as they describe the type of eye movement rather than the gaze behavior itself.
#    * validity left and validity right: keeping data which are assigned to be valid. dropping remaining values.
#    * Mouse position X, Mouse position Y: These columns may not be necessary for predicting gaze behavior as the focus is on the gaze behavior and not the mouse behavior and it has 98% NaN values.
# * replace all the , to . in the number values.
# * One commonly used method for filling missing values in time series is to use interpolation, which can help preserve the underlying patterns and trends in the data.

# In[154]:


def preprocess_data(data):
    """
    Preprocesses eye-tracking data by dropping unnecessary columns,
    filtering valid data, converting data types, interpolating missing values,
    and renaming columns.

    Args:
        data (DataFrame): Input eye-tracking data as a Pandas DataFrame.

    Returns:
        DataFrame: Preprocessed eye-tracking data as a Pandas DataFrame.
    """
    columns_dropped =['Unnamed: 0','Sensor','Project name', 'Export date', 'Recording name', 'Recording date',
                  'Recording date UTC','Recording start time', 'Recording start time UTC', 'Recording duration', 'Timeline name',
                  'Recording Fixation filter name','Recording software version','Recording resolution height', 'Recording resolution width',
                  'Recording monitor latency','Event', 'Event value','Presented Stimulus name', 'Presented Media name','Eye movement type',
                  'Eye movement type index','Mouse position X', 'Mouse position Y']

    # drop the columns
    df_preprocessed = data.drop(columns=columns_dropped)

    # keeping data which are assigned to be valid. dropping remaining values
    df_preprocessed = df_preprocessed[(df_preprocessed['Validity left'] == 'Valid') & (df_preprocessed['Validity right'] == 'Valid')]
    df_preprocessed.drop(columns=['Validity left', 'Validity right'], inplace=True)
    
    # replacing all commas to dots in the number values
    df_preprocessed = df_preprocessed.replace(to_replace=r',', value='.', regex=True)

    columns_to_convert = ['Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z', 'Gaze direction right X', 
                          'Gaze direction right Y', 'Gaze direction right Z', 'Pupil diameter left', 'Pupil diameter right', 
                          'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)', 
                          'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)', 
                          'Gaze point left X (DACSmm)', 'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)', 
                          'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)', 'Gaze point left X (MCSnorm)', 
                          'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)', 'Gaze point right Y (MCSnorm)', 
                          'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)']
    
    # converting selected columns to float data type
    df_preprocessed[columns_to_convert] = df_preprocessed[columns_to_convert].astype(float)

    # linear interpolation
    df_interpolated = df_preprocessed.interpolate(method='linear', limit_direction='forward')

    # filling remaining NaN values with forward fill method
    df_preprocessed = df_interpolated.fillna(method='ffill')

    # dropping rows with NaN values in 'Pupil diameter left' and 'Pupil diameter right' columns
    df_preprocessed.dropna(subset=['Pupil diameter left', 'Pupil diameter right'], inplace=True)
    
    # Convert 'Participant name' column to integer and rename it to 'Participant nr'
    df_preprocessed['Participant name'] = df_preprocessed['Participant name'].str[-2:].astype(int)
    df_preprocessed.rename(columns={'Participant name': 'Participant nr'}, inplace=True)
    
    return df_preprocessed


# In[155]:


# Call the preprocess_data() function to preprocess the data
df_preprocessed = preprocess_data(df)


# In[12]:


df_preprocessed.info()


# In[13]:


df_preprocessed.head()


# Time series Plot

# In[14]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Recording timestamp'], df['Gaze point left X'], label='X')
ax.plot(df['Recording timestamp'], df['Gaze point left Y'], label='Y')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Gaze point (DACSmm)')
ax.legend()
plt.show()


# Plotting the time series of "Gaze event duration" for fixations longer than a certain duration (e.g., 500 ms):

# In[15]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
mask = df['Gaze event duration'] > 500
ax.plot(df.loc[mask, 'Recording timestamp'], df.loc[mask, 'Gaze event duration'])
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Fixation duration (ms)')
plt.show()


# In[16]:


def plot_dataset(df_preprocessed, x="Computer timestamp", y="Gaze point Y"):
    plt.plot( df_preprocessed[x],df_preprocessed[y], label=y)
    plt.title(y + " VS Computer Timestamp")
    plt.xlabel("Computer Timestamp")
    plt.ylabel(y)
    plt.legend()
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.show()
plot_dataset(df_preprocessed)


# In[17]:


def plot_timeseries(df):
    """
    Plot time series data from DataFrame using matplotlib.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be plotted.

    Returns:
        None
    """
    data = df_preprocessed
    plt.rcParams["figure.figsize"] = (20, 20)
    df_columns = list(data.columns)[1:]
    fig, axs = plt.subplots(nrows=6, ncols=8)
    fig.subplots_adjust(hspace=0.6, wspace=0.8)
    for i in range(len(df_columns)):
        col = df_columns[i]
        try:
            ax = axs.flat[i]
            ax.set_title(col)
            ax.plot(data["Recording timestamp"], data[col], label=col)
        except ValueError:
            print("Found value error in column: ", col)
        except KeyError:
            pass
        except TypeError:
            pass
    plt.legend()


# In[18]:


plot_timeseries(df_preprocessed)


# After plotting the time-series graphs, I got to know we can delete the columns which have a constant graph with respect to the "Eyetracker timestamp", "Computer timestamp", "Recording timestamp" column as they do not provide any useful information.

# In[158]:


#Identify constant columns
constant_cols = []
for col in df_preprocessed.columns:
    if col not in ["Eyetracker timestamp", "Computer timestamp", "Recording timestamp"]:
        if df_preprocessed[col].nunique() == 1:
            constant_cols.append(col)

#Drop constant columns from the DataFrame
df_preprocessed.drop(columns=constant_cols, inplace=True)


# In[20]:


plot_timeseries(df_preprocessed)


# In[160]:


#Creating histogram plots for the columns in the DataFrame
df_preprocessed.hist(bins=60, figsize=(20, 15));


# In[161]:


#correlation matrix for columns after preprocessing 
correlation_matrix = df_preprocessed.corr()
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',linewidth=.5)
plt.show()


# Analyzing gaze behavior during tasks

# Analyzing gaze behavior during tasks typically involves processing the raw eye tracker data to extract meaningful gaze metrics that can provide insights into the participant's visual attention patterns. Here are some common methods that can be used to analyze gaze behavior during tasks:
# 
# Fixation detection: Fixations are periods of stable gaze where the eyes are relatively stationary. Fixation detection algorithms can be used to identify fixations from raw gaze data, typically by setting thresholds for parameters such as gaze velocity or dispersion. Fixation duration, fixation count, and fixation locations can be calculated as metrics of gaze behavior.
# 
# Saccade detection: Saccades are rapid eye movements between fixations that are used to shift gaze from one location to another. Saccade detection algorithms can be used to identify saccades from raw gaze data, typically by detecting rapid changes in gaze position. Saccade amplitude, saccade duration, and saccade count can be calculated as metrics of gaze behavior.

# In[21]:


# Extract gaze point X and Y coordinates
gaze_x = df_preprocessed['Gaze point X']
gaze_y = df_preprocessed['Gaze point Y']

# Define fixation and saccade threshold values
fixation_threshold = 30
saccade_threshold = 30 

# Initialize fixation and saccade lists
fixations = []
saccades = []

# Loop through the gaze data to detect fixations and saccades
for i in range(1, len(gaze_x)):
    # Calculate gaze velocity as the Euclidean distance between consecutive gaze points
    velocity = np.sqrt((gaze_x.iloc[i] - gaze_x.iloc[i-1]) ** 2 + (gaze_y.iloc[i] - gaze_y.iloc[i-1]) ** 2)
    # Check if the velocity falls below the fixation threshold, indicating a fixation
    if velocity < fixation_threshold:
        # Add the gaze point coordinates to the fixations list
        fixations.append((gaze_x.iloc[i], gaze_y.iloc[i]))
    # Check if the velocity exceeds the saccade threshold, indicating a saccade
    elif velocity > saccade_threshold:
        # Add the gaze point coordinates to the saccades list
        saccades.append((gaze_x.iloc[i], gaze_y.iloc[i]))

# Convert fixations and saccades to numpy arrays
fixations = np.array(fixations)
saccades = np.array(saccades)

print("fixations : ",fixations)
print()
print("saccades : ",saccades)


# In[22]:


# Calculate fixation duration
fixation_durations = []
for i in range(len(fixations) - 1):
    fixation_duration = df_preprocessed['Recording timestamp'].iloc[i+1] - df_preprocessed['Recording timestamp'].iloc[i]
    fixation_durations.append(fixation_duration)

# Calculate saccade amplitude
saccade_amplitudes = []
for i in range(len(saccades) - 1):
    saccade_amplitude = np.sqrt((saccades[i+1][0] - saccades[i][0]) ** 2 + (saccades[i+1][1] - saccades[i][1]) ** 2)
    saccade_amplitudes.append(saccade_amplitude)

#print("fixation duration: ",fixation_duration)
#print("sccade amplitudes: ",saccade_amplitudes)

# Generate fixation duration histogram
plt.figure(figsize=(8, 6))
plt.hist(fixation_durations, bins=10)
#plt.figure(figsize=(10, 7))
plt.xlabel('Fixation Duration (ms)')
plt.ylabel('Frequency')
plt.title('Fixation Duration Histogram')
plt.show()

# Generate saccade amplitude histogram
plt.figure(figsize=(8, 6))
plt.hist(saccade_amplitudes, bins=10)
plt.xlabel('Saccade Amplitude (pixels)')
plt.ylabel('Frequency')
plt.title('Saccade Amplitude Histogram')
plt.show()


# Heatmaps and scanpaths: Heatmaps and scanpaths are visual representations of gaze behavior during tasks. Heatmaps are created by aggregating gaze data into a 2D grid and calculating the density of fixations at each grid location, while scanpaths are the sequential order of fixations and saccades. Heatmaps and scanpaths can provide insights into areas of interest, gaze patterns, and exploration strategies during the tasks.

# In[23]:


# Generate gaze heatmap
heatmap, xedges, yedges = np.histogram2d(gaze_x, gaze_y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
fig = plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
plt.colorbar(label='Frequency')
plt.xlabel('Gaze Point X (pixels)')
plt.ylabel('Gaze Point Y (pixels)')
plt.title('Gaze Heatmap')
plt.show()


# In[24]:


# Generate a heatmap of fixations
heatmap, xedges, yedges = np.histogram2d(fixations[:, 0], fixations[:, 1], bins=(100, 100))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
plt.colorbar(label='Fixation count')
plt.xlabel('Gaze point X')
plt.ylabel('Gaze point Y')
plt.title('Fixation Heatmap')
plt.show()

# Generate a scanpath plot of fixations and saccades
plt.figure(figsize=(8, 6))
plt.plot(gaze_x, gaze_y, color='gray', alpha=0.5, label='Gaze')
plt.scatter(fixations[:, 0], fixations[:, 1], color='blue', label='Fixations')
plt.scatter(saccades[:, 0], saccades[:, 1], color='red', label='Saccades')
plt.xlabel('Gaze point X')
plt.ylabel('Gaze point Y')
plt.legend()
plt.title('Scanpath')
plt.show()

# Print analysis results
print('Fixation duration:', fixation_duration, 'fixations')
print('Saccade amplitude:', saccade_amplitude, 'pixels')


# Understanding the significance of the signals from the data set and how they affect the model's prediction of empathy comes through data exploration.

# # Using 60 files for modling

# * We looked at one file out of 502 in order to better understand the data it contained. Now, for 60 participants, considering one trail file per participant will be used to develop the model. 
# 

# Reading CSV files into DataFrames

# In[2]:


data_path = "EyeT/"

participants = range(1, 61)
trial_num = 1

selected_files = []

for participant in participants:
    participant_files = os.listdir(data_path)
    participant_files = [f for f in participant_files if f.endswith(f"_participant_{participant}_trial_{trial_num}.csv")]
    if len(participant_files) == 0:
        print(f"No files found for participant {participant}")
        continue
    selected_files.append(os.path.join(data_path, participant_files[0]))


# In[3]:


len(selected_files)


# In[4]:


# Initialize an empty list to store the dataframes
dfs = []

#loop through each selected files
for files in selected_files:
  df = pd.read_csv(files)
  dfs.append(df)

#Concatenate all the DataFrames into a single one
df = pd.concat(dfs)
print(df.shape)
df.head()


# Data Pre-procesing

# * Added the number of columns to the drop-down list in the preceding function. Following a time series plot check, certain columns with constant values were dropped. Therefore, I have added the names of those columns with the previous list of columns dropped.
# 

# In[33]:


def preprocess_data(data):
    """
    Preprocesses eye-tracking data by dropping unnecessary columns,
    filtering valid data, converting data types, interpolating missing values,
    and renaming columns.

    Args:
        data (DataFrame): Input eye-tracking data as a Pandas DataFrame.

    Returns:
        DataFrame: Preprocessed eye-tracking data as a Pandas DataFrame.
    """
    columns_dropped =['Recording timestamp', 'Computer timestamp',
                      'Recording duration', 'Eyetracker timestamp','Event value', 'Sensor', 'Recording name', 'Eye movement type index', 
                      'Eye movement type', 'Recording resolution width', 'Recording resolution height', 
                      'Timeline name', 'Export date', 
                      'Recording date UTC', 'Mouse position X', 'Recording start time', 
                      'Recording Fixation filter name', 'Presented Stimulus name', 
                      'Recording software version', 'Original Media height', 'Presented Media width', 
                      'Presented Media name', 'Recording date', 'Recording start time UTC', 'Original Media width', 
                      'Presented Media position X (DACSpx)', 'Unnamed: 0', 'Event', 'Presented Media position Y (DACSpx)', 
                      'Mouse position Y', 'Recording monitor latency', 'Project name', 'Presented Media height']

    # drop the columns
    df_preprocessed = data.drop(columns=columns_dropped)

    # keeping data which are assigned to be valid. dropping remaining values
    df_preprocessed = df_preprocessed[(df_preprocessed['Validity left'] == 'Valid') & (df_preprocessed['Validity right'] == 'Valid')]
    df_preprocessed.drop(columns=['Validity left', 'Validity right'], inplace=True)
    
    # replacing all commas to dots in the number values
    df_preprocessed = df_preprocessed.replace(to_replace=r',', value='.', regex=True)

    columns_to_convert = ['Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z', 'Gaze direction right X', 
                          'Gaze direction right Y', 'Gaze direction right Z', 'Pupil diameter left', 'Pupil diameter right', 
                          'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)', 
                          'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)', 
                          'Gaze point left X (DACSmm)', 'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)', 
                          'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)', 'Gaze point left X (MCSnorm)', 
                          'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)', 'Gaze point right Y (MCSnorm)', 
                          'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)']
    
    # converting selected columns to float data type
    df_preprocessed[columns_to_convert] = df_preprocessed[columns_to_convert].astype(float)

    # linear interpolation
    df_interpolated = df_preprocessed.interpolate(method='linear', limit_direction='forward')

    # filling remaining NaN values with forward fill method
    df_preprocessed = df_interpolated.fillna(method='ffill')

    # dropping rows with NaN values in 'Pupil diameter left' and 'Pupil diameter right' columns
    df_preprocessed.dropna(subset=['Pupil diameter left', 'Pupil diameter right'], inplace=True)
    
    # Convert 'Participant name' column to integer and rename it to 'Participant nr'
    df_preprocessed['Participant name'] = df_preprocessed['Participant name'].str[-2:].astype(int)
    df_preprocessed.rename(columns={'Participant name': 'Participant nr'}, inplace=True)
    
    return df_preprocessed


# In[34]:


# Call the preprocess_data() function to preprocess the data
df_preprocessed = preprocess_data(df)


# In[35]:


df_questionnaire_1 = pd.read_csv("Questionnaire_datasetIA.csv",encoding='latin-1')

# questionnaire could be used as the ground truth
df_questionnaire = df_questionnaire_1[['Participant nr','Total Score original']]
# there are 40 questions and each question worth 5 points
#scores.describe()
df_questionnaire.head()


# In[36]:


# Merge the dataframes on 'Participant nr'
df_merged = pd.merge(df_preprocessed, df_questionnaire, on='Participant nr')


# In[37]:


df_merged.head()


# In[10]:


#correlation matrix for columns after preprocessing 
correlation_matrix = df_merged.corr()
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',linewidth=.5)
plt.show()


# In[11]:


df_merged.columns


# # Feature selection

# To predict empathy scores using the eye test data, we need to identify relevant features that could potentially impact empathy levels. Here are some possible important features to consider:
# 
#     'Gaze direction left X' and 'Gaze direction left Y': These columns represent the X and Y components of the gaze direction vector for the left eye, which could be indicative of the direction of the participant's gaze. The direction of gaze could potentially be related to empathy, as previous research has shown that gaze patterns can be associated with social cognition and emotional processing.
# 
#     'Gaze direction right X' and 'Gaze direction right Y': Similar to the gaze direction for the left eye, these columns represent the X and Y components of the gaze direction vector for the right eye, which could also be informative for predicting empathy scores.
# 
#     'Pupil diameter left' and 'Pupil diameter right': These columns represent the diameter of the pupil for the left and right eyes, respectively. Pupil dilation has been linked to emotional arousal, and changes in pupil diameter could potentially reflect differences in emotional responsiveness, which could be relevant for empathy.
# 
#     'Gaze event duration': This column represents the duration of gaze events, which could provide information on the duration of fixations or sustained attention to specific areas of interest. Gaze event duration could potentially be related to empathy, as it could reflect differences in attentional processing during social interactions.
# 
#     'Gaze point X' and 'Gaze point Y': These columns represent the X and Y coordinates of the gaze point on the screen, which could provide information on the spatial distribution of gaze during the eye test. The pattern of gaze points could potentially be associated with empathy, as it could reflect differences in visual attention to social cues.
# 
#     'Fixation point X' and 'Fixation point Y': Similar to the gaze point coordinates, these columns represent the X and Y coordinates of the fixation point on the screen, which could also provide information on the spatial distribution of fixations during the eye test. The pattern of fixation points could potentially be relevant for empathy, as it could reflect differences in visual attention to specific areas of interest.
#     
# 

# * picking these features to train a model, evaluating its performance, and contrasting it with another model that includes all features.

# In[12]:


X_selected = df_merged[['Gaze direction left X', 'Gaze direction left Y', 'Gaze direction right X', 'Gaze direction right Y',
                        'Pupil diameter left', 'Pupil diameter right', 'Gaze event duration', 'Gaze point X', 'Gaze point Y',
                        'Fixation point X', 'Fixation point Y']]


# In[13]:


X_selected.shape


# In[14]:


#correlation matrix for columns after preprocessing 
correlation_matrix = X_selected.corr()
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',linewidth=.5)
plt.show()


# Multiple features have varying scales, thus the data is scaled using the MIn-max scaler by linearly translating the values to a certain range, often [0,1]. It is suitable for data that has different scales across different features.

# In[15]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Perform min-max scaling on the remaining columns
X_scaled = scaler.fit_transform(X_selected)

# Convert the result back to a dataframe
X_scaled_df = pd.DataFrame(X_scaled, columns=X_selected.columns)

X_scaled_df.head()


# In[16]:


X_scaled_df.shape


# In[17]:


X = X_scaled_df
y = df_merged['Total Score original']

# Perform train-test split on normalized feature matrix and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

len(X_train), len(X_test), len(y_train), len(y_test)


# In[18]:


X.shape


# In[19]:


y.shape


# Multilayer Perceptron (MLP)

# In[20]:


from keras.models import Sequential
from keras.layers import Dense

# Create an MLP model
model = Sequential()

# Add input layer with the number of features as input_dim
model.add(Dense(units=64, activation='relu', input_dim=11))  # 11 features in X_selected

# Add one or more hidden layers with desired number of units and activation functions
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=16, activation='relu'))

# Add output layer with a single output neuron for regression task or multiple neurons for multi-class classification
model.add(Dense(units=1))  #regression task

# Compile the model by specifying loss function, optimizer, and evaluation metric(s)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Train the model on your training data
history = model.fit(X_train, y_train, epochs=10, batch_size=32)  

# Evaluate the model on your test data
loss, mse = model.evaluate(X_test, y_test)  

# Make predictions with the trained model
predictions = model.predict(X_test)  


# Here's a function that encapsulates the code for creating scatter plots, histograms, density plots, and box plots of predicted vs. actual values for empathy scores using Matplotlib and Seaborn.

# In[21]:


#Scatter Plot Function

def create_scatter_plot(y_test, predictions):
    """
    Creates a scatter plot of predicted vs. actual values for empathy scores.

    Parameters:
        -- y_test (array-like): Actual empathy scores.
        -- predictions (array-like): Predicted empathy scores.
    """
    plt.scatter(y_test, predictions, c='blue', label='Predicted')
    plt.scatter(y_test, y_test, c='red', label='Actual')
    plt.xlabel('Actual Empathy Score')
    plt.ylabel('Predicted Empathy Score')
    plt.title('Actual vs. Predicted Empathy Score')
    plt.legend()
    plt.show()


# In[22]:


#Histogram Function

def create_histograms(y_test, predictions):
    """
    Creates histograms of predicted and actual values for empathy scores.

    Parameters:
        -- y_test (array-like): Actual empathy scores.
        -- predictions (array-like): Predicted empathy scores.
    """
    plt.hist(y_test, bins=20, alpha=0.5, label='Actual Empathy Score')
    plt.hist(predictions, bins=20, alpha=0.5, label='Predicted Empathy Score')
    plt.xlabel('Empathy Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Actual vs. Predicted Empathy Score')
    plt.legend()
    plt.show()


# In[23]:


#Density Plot Function

def create_density_plots(y_test, predictions):
    """
    Creates density plots of predicted and actual values for empathy scores.

    Parameters:
        -- y_test (array-like): Actual empathy scores.
        -- predictions (array-like): Predicted empathy scores.
    """
    sns.kdeplot(y_test, label='Actual Empathy Score', color='red')
    sns.kdeplot(predictions, label='Predicted Empathy Score')
    plt.xlabel('Empathy Score')
    plt.ylabel('Density')
    plt.title('Density Plot of Actual vs. Predicted Empathy Score')
    plt.legend()
    plt.show()


# In[24]:


#Box Plot Function

def create_box_plot(y_test, predictions):
    """
    Creates a box plot of predicted vs. actual values for empathy scores.

    Parameters:
        -- y_test (array-like): Actual empathy scores.
        -- predictions (array-like): Predicted empathy scores.
    """
    sns.boxplot(data=[y_test, predictions], width=0.5)
    plt.xticks(ticks=[0, 1], labels=['Actual', 'Predicted'])
    plt.ylabel('Empathy Score')
    plt.title('Box Plot of Actual vs. Predicted Empathy Score')
    plt.show()


# In[25]:


#loss curve function

def plot_loss_curve(history):
    """
    Plots the training loss over epochs from the history object.

    Parameters:
        -- history (object): History object returned by the model.fit() function.
    """
    # Extract loss values and number of epochs
    loss = history.history['loss']
    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='Training Loss')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()


# In[26]:


create_density_plots(y_test, predictions)


# In[27]:


create_box_plot(y_test, predictions)


# In[28]:


create_histograms(y_test, predictions)


# In[29]:


create_scatter_plot(y_test, predictions)


# In[30]:


plot_loss_curve(history)


# In[50]:


X_scaled_df.shape


# In[51]:


predictions


# In[52]:


import numpy as np
import pandas as pd

# concatenate y_test and predictions
df01 = pd.DataFrame(np.concatenate([y_test.values.reshape(-1,1), predictions.reshape(-1,1)], axis=1), columns=['y_test', 'predictions'])

df01['predictions'] = df01['predictions'].round()

# print the dataframe
df01.head(10)


# In[32]:


def calculate_mismatch_percentage(y_test, predictions, df):
    """
    Calculates and prints the number and percentage of values that do not match in y_test and predictions.

    Parameters:
        -- y_test (array-like): Actual empathy scores.
        -- predictions (array-like): Predicted empathy scores.
        -- df (DataFrame): DataFrame containing y_test and predictions as columns.
    """
    # Calculate number of values that do not match in y_test and predictions
    mismatch_predictions = (df['predictions'] != df['y_test']).sum()

    # Calculate percentage of values that do not match in y_test and predictions
    percent_mismatch_predictions = (mismatch_predictions / len(df)) * 100

    # Calculate number of values that do match in y_test and predictions
    match_predictions = len(df) - mismatch_predictions

    # Calculate percentage of values that do match in y_test and predictions
    percent_match_predictions = (match_predictions / len(df)) * 100

    # Print results
    print(f'Number of values that do not match in predictions: {mismatch_predictions} out of {len(df)}')
    print(f'Percentage of values that do not match in predictions: {percent_mismatch_predictions.round(2)}%')
    print(f'Number of values that do match in predictions: {match_predictions} out of {len(df)}')
    print(f'Percentage of values that do match in predictions: {percent_match_predictions.round(2)}%')


# In[128]:


calculate_mismatch_percentage(y_test, predictions, df01)


# Changing epochs=100 for same model

# In[56]:


from keras.models import Sequential
from keras.layers import Dense

# Create an MLP model
model1 = Sequential()

# Add input layer with the number of features as input_dim
model1.add(Dense(units=64, activation='relu', input_dim=11))  # Assuming 11 features in X_selected

# Add one or more hidden layers with desired number of units and activation functions
model1.add(Dense(units=32, activation='relu'))
model1.add(Dense(units=16, activation='relu'))

# Add output layer with a single output neuron for regression task or multiple neurons for multi-class classification
model1.add(Dense(units=1))  #regression task

# Compile the model by specifying loss function, optimizer, and evaluation metric(s)
model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Train the model on your training data
history1 = model1.fit(X_train, y_train, epochs=100, batch_size=32)  

# Evaluate the model on your test data
loss, mse = model1.evaluate(X_test, y_test)  

# Make predictions with the trained model
predictions1 = model1.predict(X_test)  


# In[57]:


plot_loss_curve(history1)


# In[58]:


create_scatter_plot(y_test, predictions1)


# In[59]:


create_histograms(y_test, predictions1)


# In[60]:


create_density_plots(y_test, predictions1)


# In[61]:


# concatenate y_test and predictions
df02 = pd.DataFrame(np.concatenate([y_test.values.reshape(-1,1), predictions1.reshape(-1,1)], axis=1), columns=['y_test', 'predictions'])

# Round the predicted scores to 2 decimal places
df02['predictions'] = df02['predictions'].round()

# print the dataframe
df02.head(10)


# In[129]:


calculate_mismatch_percentage(y_test, predictions, df02)


# * One trail file for 60 participants with 11 characteristics was examined. 
# 
# * For training and testing, we will now utilise the same amount of files with 37 features.
# 
# 

# In[38]:


X_all_features = df_merged.drop(['Participant nr', 'Total Score original'], axis=1)
X_all_scaled = scaler.fit_transform(X_all_features)

# Convert the result back to a dataframe
X_all_features_df = pd.DataFrame(X_all_scaled, columns=X_all_features.columns)

X_all_features_df.head()


# In[39]:


X_all_features_df.shape


# In[40]:


from sklearn.model_selection import train_test_split
X = X_all_features_df
y = df_merged['Total Score original']

# Perform train-test split on normalized feature matrix and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

len(X_train), len(X_test), len(y_train), len(y_test)


# In[41]:


from keras.models import Sequential
from keras.layers import Dense

# Create an MLP model
model2 = Sequential()

# Add input layer with the number of features as input_dim
model2.add(Dense(units=64, activation='relu', input_dim=35))  #35 features in X_selected

# Add one or more hidden layers with desired number of units and activation functions
model2.add(Dense(units=32, activation='relu'))
model2.add(Dense(units=16, activation='relu'))

# Add output layer with a single output neuron for regression task or multiple neurons for multi-class classification
model2.add(Dense(units=1))  #regression task

# Compile the model by specifying loss function, optimizer, and evaluation metric(s)
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Train the model on your training data
history2 = model2.fit(X_train, y_train, epochs=10, batch_size=32)  

# Evaluate the model on your test data
loss, mse = model2.evaluate(X_test, y_test)  

# Make predictions with the trained model
predictions2 = model2.predict(X_test)  


# In[42]:


# concatenate y_test and predictions
df02 = pd.DataFrame(np.concatenate([y_test.values.reshape(-1,1), predictions2.reshape(-1,1)], axis=1), columns=['y_test', 'predictions'])

# Round the predicted scores to 2 decimal places
df02['predictions'] = df02['predictions'].round()

# print the dataframe
df02.head(10)


# In[45]:


calculate_mismatch_percentage(y_test, predictions, df02)


# With just for 10 iterations, our loss is equal to that of model 1. The model is picking up patterns as more features are introduced, so let's try training it for 50 more epochs.

# In[68]:


# Create an MLP model
model3 = Sequential()

# Add input layer with the number of features as input_dim
model3.add(Dense(units=64, activation='relu', input_dim=35))  #35 features in X_selected

# Add one or more hidden layers with desired number of units and activation functions
model3.add(Dense(units=32, activation='relu'))
model3.add(Dense(units=16, activation='relu'))

# Add output layer with a single output neuron for regression task or multiple neurons for multi-class classification
model3.add(Dense(units=1))  #regression task

# Compile the model by specifying loss function, optimizer, and evaluation metric(s)
model3.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Train the model on your training data
history3 = model3.fit(X_train, y_train, epochs=50, batch_size=32)  

# Evaluate the model on your test data
loss, mse = model3.evaluate(X_test, y_test)  

# Make predictions with the trained model
predictions3 = model3.predict(X_test)  


# In[69]:


create_scatter_plot(y_test, predictions3)


# In[70]:


create_histograms(y_test, predictions3)


# In[71]:


create_density_plots(y_test, predictions3)


# In[72]:


create_box_plot(y_test, predictions3)


# In[73]:


plot_loss_curve(history3)


# In[74]:


# concatenate y_test and predictions
df03 = pd.DataFrame(np.concatenate([y_test.values.reshape(-1,1), predictions3.reshape(-1,1)], axis=1), columns=['y_test', 'predictions'])

# Round the predicted scores to 2 decimal places
df03['predictions'] = df03['predictions'].round()

# print the dataframe
df03.head(10)


# In[130]:


calculate_mismatch_percentage(y_test, predictions3, df03)


# # Full dataset

# Let's make predictions for the entire dataset, which consists of 502 files from 60 people. 
# 

# In[76]:


import os
from collections import defaultdict

data_path = "EyeT/"
participants = range(1, 61)

selected_files = []
participant_trial_count = defaultdict(int)

for participant in participants:
    for trial_num in range(0, 61):
        participant_files = os.listdir(data_path)
        participant_files = [f for f in participant_files if f.endswith(f"_participant_{participant}_trial_{trial_num}.csv")]
        if len(participant_files) == 0:
            continue
        selected_files.extend([os.path.join(data_path, f) for f in participant_files])
        participant_trial_count[participant] += 1
        #print(participant_files)

# Print the list of participants with the number of trials they have taken
for participant, trial_count in participant_trial_count.items():
    print(f"Participant {participant}: Number of Trials = {trial_count} ")


# In[77]:


# Calculate total number of trials across all participants
total_trials = sum(participant_trial_count.values())

# Print total number of trials
print("Total number of trials: ", total_trials)


# In[78]:


import glob
#Using Glob module for loading
path = r'EyeT/'
files = glob.glob(os.path.join(path, "*.csv"))    
data=[]
#looping to all files and storing in data
for i in files:
    df = pd.read_csv(i)
    data.append(df)
#concatenating all files stored in data
data = pd.concat(data, ignore_index=True)
data.shape # (rows, columns)


# In[79]:


data.head()


# In[80]:


value_counts = data['Participant name'].value_counts()
value_counts


# In[81]:


# Call the preprocess_data() function to preprocess the data
df_preprocessed = preprocess_data(data)


# In[82]:


# Merge the dataframes on 'Participant nr'
df_merged = pd.merge(df_preprocessed, df_questionnaire, on='Participant nr')


# In[83]:


df_merged.head()


# In[84]:


X_all_features = df_merged.drop(['Participant nr', 'Total Score original'], axis=1)
X_all_scaled = scaler.fit_transform(X_all_features)

# Convert the result back to a dataframe
X_all_features_df = pd.DataFrame(X_all_scaled, columns=X_all_features.columns)

X_all_features_df.head()


# In[85]:


X_all_features_df.shape


# In[126]:


#correlation matrix for columns after preprocessing 
correlation_matrix = df_merged.corr()
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',linewidth=.5)
plt.show()


# In[86]:


from sklearn.model_selection import train_test_split
X = X_all_features_df
y = df_merged['Total Score original']

# Perform train-test split on normalized feature matrix and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

len(X_train), len(X_test), len(y_train), len(y_test)


# In[87]:


# Create an MLP model
model4 = Sequential()

# Add input layer with the number of features as input_dim
model4.add(Dense(units=64, activation='relu', input_dim=35))  #35 features in X_selected

# Add one or more hidden layers with desired number of units and activation functions
model4.add(Dense(units=32, activation='relu'))
model4.add(Dense(units=16, activation='relu'))

# Add output layer with a single output neuron for regression task or multiple neurons for multi-class classification
model4.add(Dense(units=1))  # Assuming a regression task

# Compile the model by specifying loss function, optimizer, and evaluation metric(s)
model4.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Train the model on your training data
history4 = model4.fit(X_train, y_train, epochs=10, batch_size=32)  

# Evaluate the model on your test data
loss, mse = model4.evaluate(X_test, y_test)  

# Make predictions with the trained model
predictions4 = model4.predict(X_test)  


# In[88]:


create_scatter_plot(y_test, predictions4)


# In[89]:


create_density_plots(y_test, predictions4)


# In[90]:


create_histograms(y_test, predictions4)


# In[91]:


# concatenate y_test and predictions
df04 = pd.DataFrame(np.concatenate([y_test.values.reshape(-1,1), predictions4.reshape(-1,1)], axis=1), columns=['y_test', 'predictions'])

# Round the predicted scores to 2 decimal places
df04['predictions'] = df04['predictions'].round()

# print the dataframe
df04.head(10)


# In[92]:


len(df04)


# In[131]:


calculate_mismatch_percentage(y_test, predictions, df04)


# Although the predictions are near, we will achieve much better outcomes if we train for more epochs. Let's practise for 50 epochs.
# 

# In[95]:


# Create an MLP model
model5 = Sequential()

# Add input layer with the number of features as input_dim
model5.add(Dense(units=64, activation='relu', input_dim=35))  #40 features in X_selected

# Add one or more hidden layers with desired number of units and activation functions
model5.add(Dense(units=32, activation='relu'))
model5.add(Dense(units=16, activation='relu'))

# Add output layer with a single output neuron for regression task or multiple neurons for multi-class classification
model5.add(Dense(units=1))  # Assuming a regression task

# Compile the model by specifying loss function, optimizer, and evaluation metric(s)
model5.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Train the model on your training data
history5 = model5.fit(X_train, y_train, epochs=50, batch_size=32)  

# Evaluate the model on your test data
loss, mse = model5.evaluate(X_test, y_test)  

# Make predictions with the trained model
predictions5 = model5.predict(X_test)  

# Print the model summary
model5.summary()


# In[96]:


create_scatter_plot(y_test, predictions5)


# In[97]:


create_density_plots(y_test, predictions5)


# In[98]:


create_histograms(y_test, predictions5)


# In[99]:


plot_loss_curve(history5)


# In[100]:


from keras.models import load_model

# Load the saved model
my_model = load_model('model5.h5')
my_model.summary()


# In[101]:


# concatenate y_test and predictions
df05 = pd.DataFrame(np.concatenate([y_test.values.reshape(-1,1), predictions5.reshape(-1,1)], axis=1), columns=['y_test', 'predictions'])

# Round the predicted scores to 2 decimal places
df05['predictions'] = df05['predictions'].round()

# print the dataframe
df05.head(10)


# In[132]:


calculate_mismatch_percentage(y_test, predictions, df05)


# LinearRegression

# In[103]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# Initialize the Linear Regression model
model6 = LinearRegression()

# Train the model
model6.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_0 = model6.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred_0)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_0)
r2 = r2_score(y_test, y_pred_0)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared Score:", r2)


# Random Forest Regressor

# In[104]:


from sklearn.ensemble import RandomForestRegressor

# Initialize the Linear Regression model
model7 = RandomForestRegressor(n_estimators=5)

# Train the model
model7.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_1 = model7.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred_1)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_1)
r2 = r2_score(y_test, y_pred_1)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared Score:", r2)


# In[113]:


create_density_plots(y_test, y_pred_1)


# In[114]:


create_histograms(y_test, y_pred_1)


# In[115]:


create_scatter_plot(y_test, y_pred_1)


# In[116]:


# concatenate y_test and predictions
df07 = pd.DataFrame(np.concatenate([y_test.values.reshape(-1,1), y_pred_1.reshape(-1,1)], axis=1), columns=['y_test', 'predictions'])

# Round the predicted scores to 2 decimal places
df07['predictions'] = df07['predictions'].round()

# print the dataframe
df07.head(10)


# In[133]:


calculate_mismatch_percentage(y_test, predictions, df07)


# Based on the MSE, RMSE, MAE, and R-squared score values, the model's performance appears to be very good, with highly accurate predictions and a high degree of explanation of the variance in the target variable.
# 
# Let's interpret the different evaluation metrics:
# 
#     * Mean Squared Error (MSE): The MSE is a measure of the average squared difference between the predicted values and the actual values. A lower MSE indicates better accuracy, with 0 being a perfect score. In this case, the MSE value of 0.17803307948256644 is very close to 0, which indicates that the model's predictions are very accurate.
# 
#     * Root Mean Squared Error (RMSE): The RMSE is the square root of MSE and provides an estimate of the standard deviation of the residuals (i.e., the prediction errors). Like MSE, a lower RMSE indicates better accuracy, with 0 being a perfect score. In this case, the RMSE value of 0.42193966331996624 is also very low, indicating that the model's predictions are highly accurate.
# 
#     * Mean Absolute Error (MAE): The MAE is a measure of the average absolute difference between the predicted values and the actual values. Similar to MSE and RMSE, a lower MAE indicates better accuracy, with 0 being a perfect score. The MAE value of 0.028568332995942754 is very close to 0, which indicates that the model's predictions are highly accurate.
# 
#     * R-squared Score: The R-squared score, also known as the coefficient of determination, represents the proportion of the variance in the target variable (y_test) that is explained by the model's predictions. R-squared ranges from 0 to 1, with 1 being a perfect score indicating that the model explains all of the variance in the target variable. In this case, the R-squared score of 0.998432510314932 is very close to 1, indicating that the model's predictions explain almost all of the variance in the target variable, and thus, the model's performance is excellent.

# In[ ]:





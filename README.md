# ARMBR
Version 1.0.0 

This repository is the original implementation of ARMBR:    
Artifact-Reference Multivariate Backward Regression (ARMBR) Outperforms Common EEG Blink Artifact Removal Methods

by Ludvik Alkhoury, Giacomo Scanavini, Samuel Louviot, Ana Radanovic, Sudhin A. Shah, and N. Jeremy Hill

Below, we provide instructions on how to: 
1) download the package 
2) implement the code in Matlab and Python

# Download Package
1) Select a directory to save this repository to. This can be done by creating a directory on your Desktop (for example: C:/MyPC/Desktop/GitRepo/).
Open the Git bash and type:
```
cd "C:/MyPC/Desktop/GitRepo/"
```

2) Download (clone) this repository in the directory that we previously created. This can be done by opening Git bash inside the directory we just created (for example: C:/MyPC/Desktop/GitRepo/) and typing:
```
git clone "https://github.com/S-Shah-Lab/ARMBR.git"
```
and then go the repository directory by typing:
```
cd ARMBR/
```

3) Now that you have the repository, run setup.py to install all dependent packages:
```
python -m pip install -e  ./Python
```



# Matlab Implementation 

ARMBR can be used in Python as follows:

## Option 1: A generic script
Here is how to implement ARMBR using Matlab on the semi-synthetic data used in the paper. 
This implementation will work with any EEG array.

```
clc; clear; close all;

fs = 128; % Set sampling rate
sub = 1;  % Set subject number

% Load clean and blink-contaminated EEG signals
Clean_EEG = load("..\SemiSyntheticData\Sub"+num2str(sub)+"\Sub"+num2str(sub)+"_Clean_EEG.mat").Clean_EEG;
Sythentic_Blink_Contaminated_EEG = load("..\SemiSyntheticData\Sub"+num2str(sub)+"\Sub"+num2str(sub)+"_Synthetic_Blink_Contaminated_EEG.mat").Sythentic_Blink_Contaminated_EEG;

% Bandpass filter the data from 1 to 40
Sythentic_Blink_Contaminated_EEG = BPF(Sythentic_Blink_Contaminated_EEG, fs, [1 40]);
Clean_EEG                        = BPF(Clean_EEG, fs, [1 40]);

% Run ARMBR
ref_chan_nbr = [80, 93]; %indices for Fp1 and Fp2
[ARMBR_EEG, Set_IQR_Thresh, Blink_Ref, Blink_Artifact] = ARMBR(Sythentic_Blink_Contaminated_EEG, ref_chan_nbr, fs);

% Compute performance metrics
[PearCorr, RMSE, SNR] = PerformanceMetrics(Clean_EEG, ARMBR_EEG);

% Display computed metrics
disp(['========================================='])
disp(['Pearson correlation for subject ',num2str(sub),': ', num2str(round(mean(PearCorr), 2))])
disp(['SNR                 for subject ',num2str(sub),': ', num2str(round(mean(SNR), 2))])
disp(['RMSE                for subject ',num2str(sub),': ', num2str(round(mean(RMSE), 2))])
disp(['========================================='])

```


## Option 2: EEGLAB structure




# Python Implementation


ARMBR can be used in Python as follows:

## Option 1: Run from terminal
Open your terminal and use one of the following commands:

If you want to use the indices of the blink reference channels then use below, where -c "90" represents index 90: 
```
python -m ARMBR -p "YOUR_PATH\Sub1_Synthetic_Blink_Contaminated_EEG.fif" -c "90"   --plot
```

If you want to use the name of the blink reference channels then use below, where -c "C16" represents channel name C16: 
```
python -m ARMBR -p "YOUR_PATH\Sub1_Synthetic_Blink_Contaminated_EEG.fif" -c "C16"  --plot
```
At this point, this command line supports data of .fif, .edf, and .dat type.


## Option 2: A generic script
You can use a numpy array EEG with ARMBR. Here is a script:

```
from ARMBR_Library import *
import mne

EEG_Blink	= np.array([.....])
Clean		  = np.array([.....])
fs = 128


ChannelsName = ['EEG1', EEG2', ...'EEGn']

myARMBR = ARMBR(EEG=EEG_w_Blink,
                Fs=fs,
                ChannelsName=ChannelsName,
                EEGGT=Clean)

myARMBR.ARMBR(blink_chan=['EEG1','EEG2']).PerformanceMetrics().DispMetrics().Plot()

```
in this script `EEGGT` is the EEG ground truth if you are working with synthetic signals or you know what the signal should look like without blinks.
When `EEGGT` is available, you can run methods like `PerformanceMetrics()` and `DispMetrics()`. `PerformanceMetrics()` will compute the Pearson correlation, RMSE, and SNR for all channels. `DispMetrics()` will display an average across channels.


## Option 3: Work with mne raw object
You can also use ARMBR with mne raw objects. Here is a script:

```
from ARMBR_Library import *
import mne

raw = mne.io.read_raw_fif("YOUR_RAW_PATH.fif", preload=True)
raw.filter(l_freq=1, h_freq=40)

myARMBR = ARMBR()
myARMBR.ImportFromRaw(raw)
myARMBR.ARMBR(blink_chan=['EEG1','EEG2'])
myARMBR.UnloadIntoRaw(raw)

raw.plot()
raw.save("SAVE_PATH.fif")

```
With this code you can process the raw data using ARMBR and load it back to the raw object.




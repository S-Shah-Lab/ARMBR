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



# Python Implementation


ARMBR could be used as follows:

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




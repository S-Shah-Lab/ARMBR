# ARMBR: Artifact-reference multivariate backward regression
Version 2.0.7

Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements

ARMBR is a lightweight and easy-to-use method for blink artifact removal from EEG signals using multivariate backward regression. 
The algorithm detects the times at which eye blinks occur and then estimates their linear scalp projection by regressing a simplified, 
time-locked reference signal against the multichannel EEG. This projection is used to suppress blink-related components while preserving 
underlying brain signals. ARMBR requires minimal training data, does not depend on dedicated EOG channels, and operates robustly in both 
offline and real-time (online) settings, including BCI applications.

This module implements the ARMBR algorithm, described in:
  
> **Alkhoury L**, Scanavini G, Louviot S, Radanovic A, Shah SA & Hill NJ (2025). *Artifact-Reference Multivariate Backward Regression (ARMBR): A Novel Method for EEG Blink Artifact Removal with Minimal Data Requirements.* *Journal of Neural Engineering*, 22(3). [DOI: 10.1088/1741-2552/ade566](https://doi.org/10.1088/1741-2552/ade566) [PubMed: 40527334](https://www.ncbi.nlm.nih.gov/pubmed/40527334)



(see ARMBR.bibtex for the BibTeX entry)


The core algorithm supports both standalone and MNE integration via the `ARMBR` class:
```
from armbr import run_armbr   # core non-MNE-dependent code (just needs numpy)
from armbr import ARMBR       # MNE-compatible wrapper class
```

Below, we provide instructions on how to: 
1) Download the package 
2) Implement the code in MATLAB and Python

# Download Package

You can download the package directly from this repository. If so, we recommend downloading a stable release. 

Alternatively, you can use git:
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

4) The semi-synthetic data used in the ARMBR paper, along with example codes that allow testing ARMBR of the semi-synthetic data used in the paper, are available here in OSF at:
[https://osf.io/th2g6/](https://osf.io/th2g6/)


Download folder `SemiSyntheticData` and add it to the main ARMBR directory.

# MATLAB Implementation 

ARMBR can be used in Python as follows. First, make sure that your working directory is `MATLAB`.


## Option 1: A generic script
Here is how to implement ARMBR using MATLAB on the semi-synthetic data used in the paper. 
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
[ARMBR_EEG, Set_IQR_Thresh, Blink_Ref, Blink_Artifact, BlinkSpatialPattern] = ARMBR(Sythentic_Blink_Contaminated_EEG, ref_chan_nbr, fs);

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

Here, make sure you install EEGLAB and add the path to your directory. 
We will create an EEGLAB structure and apply ARMBR to the EEG structure using `pop_ARMBR`. The blink spatial pattern can be accessed from the EEG structure: `EEG.BlinkSpatialPattern`. You can use the already-trained blink spatial pattern on new EEG data. This is done by: `EEG_without_Blinks = EEG_with_Blinks * BlinkSpatialPattern;`.
```
clc
close all
clear

fs = 128; % Set sampling rate
sub = 1;  % Set subject number

% Load clean and blink-contaminated EEG signals
Clean_EEG = load("..\SemiSyntheticData\Sub"+num2str(sub)+"\Sub"+num2str(sub)+"_Clean_EEG.mat").Clean_EEG;
Sythentic_Blink_Contaminated_EEG = load("..\SemiSyntheticData\Sub"+num2str(sub)+"\Sub"+num2str(sub)+"_Synthetic_Blink_Contaminated_EEG.mat").Sythentic_Blink_Contaminated_EEG;

% Bandpass filter the data from 1 to 40
Sythentic_Blink_Contaminated_EEG = BPF(Sythentic_Blink_Contaminated_EEG, fs, [1 40]);
Clean_EEG                        = BPF(Clean_EEG, fs, [1 40]);


EEG = pop_importdata('dataformat','MATLAB','nbchan',128,'data', Sythentic_Blink_Contaminated_EEG', 'srate',fs, 'chanlocs','Biosemi128.ced');
eeglab redraw


blink_chan = ["C16", "C29"]; %or you can use blink_chan = [80, 93]; the index number
[EEG] = pop_ARMBR(EEG, blink_chan);

% Compute performance metrics
[PearCorr, RMSE, SNR] = PerformanceMetrics(Clean_EEG, EEG.data);

% Display computed metrics
disp(['========================================='])
disp(['Pearson correlation for subject ',num2str(sub),': ', num2str(round(mean(PearCorr), 2))])
disp(['SNR                 for subject ',num2str(sub),': ', num2str(round(mean(SNR), 2))])
disp(['RMSE                for subject ',num2str(sub),': ', num2str(round(mean(RMSE), 2))])
disp(['========================================='])
```









# Python Implementation


ARMBR can be used in Python as follows. First, make sure that your working directory is `Python`.

## Option 1: Run from terminal
Open your terminal and use one of the following commands:

If you want to use the indices of the blink reference channels then use below, where -c "79, 92" represents indices 79 and 92: 
```
python -m ARMBR -p "..\SemiSyntheticData\Sub1\Sub1_Synthetic_Blink_Contaminated_EEG.fif" -c "79,92" --plot
```

If you want to use the name of the blink reference channels then use below, where -c "C16,C29" represents channel name C16 and C29: 
```
python -m ARMBR -p "..\SemiSyntheticData\Sub1\Sub1_Synthetic_Blink_Contaminated_EEG.fif" -c "C16,C29" --plot
```
At this point, this command line supports data of .fif, .edf, and .dat type.


## Option 2: A generic script
You can use a numpy array EEG with ARMBR. Here is a script:

```
from ARMBR.armbr import *
import matplotlib.pyplot as plt

EEG_blink	= np.array([.....])
blink_ch_idx = [0,1] # indices of the blink reference channels
sfreq = 128


EEG_clean, optimal_alpha, blink_mask, blink_comp, blink_pattern, blink_projection_matrix = run_armbr(EEG_clink, blink_ch_idx, sfreq, alpha=-1.0)

```
In this example, `EEG_blink` is the blink contaminated EEG matrix. `blink_ch_idx` is a list of channel indices used as blink reference. `EEG_clean` is the blink-suppressed EEG matrix. 


## Option 3: Work with mne `Raw` object
You can also use ARMBR with mne raw objects, `raw1`. You can apply the blink pattern to the same raw object or even another one. 
The steps are shown below (and more detailed are availble in the example codes):

```
from ARMBR import ARMBR
import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_fif(r"..\SemiSyntheticData\Sub1\Sub1_Synthetic_Blink_Contaminated_EEG.fif", preload=True)
raw.filter(l_freq=1, h_freq=40, method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)

myarmbr = ARMBR(ch_name=['C16','C29'])
myarmbr.fit(raw)
myarmbr.apply(raw)

# Note that the blink spatial pattern could be applied to another raw object.
# For example myarmbr.apply(raw2); assuming both raw2 and raw have the same montage.

myarmbr.plot_blink_patterns() # To plot the blink spatial pattern

raw.plot()
plt.show()

```
With this code you can process the raw data using ARMBR and load it back to the raw object.


# ARMBR Result Example
Below is an example showing the blink contaminated region of a subject 1 before and after running ARMBR. After training ARMBR, we can use `myarmbr.plot()` to generate the following plot:
![BeforeAfterARMBR](https://github.com/S-Shah-Lab/ARMBR/assets/66024269/2b374eb0-47d6-4d6d-84dc-864bae5e35bf)







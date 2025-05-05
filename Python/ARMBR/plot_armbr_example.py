import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
#from mne.preprocessing import ARMBR
from armbr import ARMBR

# Load MNE sample EEG data
data_path = sample.data_path()
raw_path = str(data_path) + "/MEG/sample/sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(raw_path, preload=True)
raw.pick_types(meg=False, eeg=True)

# Add synthetic annotation for ARMBR to process
#annot = mne.Annotations(onset=[10.0], duration=[80.0], description=['armbr_fit'])
#raw.set_annotations(annot)

# Plot before ARMBR
raw.copy().plot(title="Before ARMBR", start=9, duration=4, n_channels=10, scalings='auto')


# Run ARMBR using first EEG channel as blink reference
ch_name = raw.ch_names[0:3]
armbr = ARMBR(ch_name=ch_name)
armbr.fit(raw, start = 0, stop = int(100*raw.info['sfreq']) )
armbr.apply(raw)

# Plot after ARMBR
raw.plot(title="After ARMBR", start=9, duration=4, n_channels=10, scalings='auto')

from ARMBR_Library import *
import scipy
import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_fif(r"..\SemiSyntheticData\Sub1\Sub1_Synthetic_Blink_Contaminated_EEG.fif", preload=True)
raw.filter(l_freq=1, h_freq=40)

myARMBR = ARMBR()
myARMBR.ImportFromRaw(raw)
myARMBR.ARMBR(blink_chan=['C16','C29'])
myARMBR.UnloadIntoRaw(raw)

raw.plot()
plt.show()



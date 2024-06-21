from ARMBR_Library import *
import scipy
import mne

sub = 1
fs = 128

raw = mne.io.read_raw_fif(r"C:\Users\lua4006\Shah Laboratory Dropbox\proj-LalorNatLang\derivatives\lalor_auto_processed\sub-HCnatlang001\ses-01\eeg\sub-HCnatlang001_ses-01_task-NatLang_run-01_eeg.fif", preload=True)
raw.filter(l_freq=1, h_freq=40)

myARMBR = ARMBR()
myARMBR.ImportFromRaw(raw)
myARMBR.ARMBR(blink_chan=['C16','C29']).Plot()
myARMBR.UnloadIntoRaw(raw)







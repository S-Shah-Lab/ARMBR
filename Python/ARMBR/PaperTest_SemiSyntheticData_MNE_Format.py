from armbr import ARMBR
import scipy
import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_fif(r"..\..\SemiSyntheticData\Sub1\Sub1_Synthetic_Blink_Contaminated_EEG.fif", preload=True)
raw.filter(l_freq=1, h_freq=40, method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)

raw_before = raw.copy()
raw_after  = raw.copy()

myarmbr = ARMBR(ch_name=['C16','C29'])
myarmbr.fit(raw_before)
myarmbr.apply(raw_after)

myarmbr.plot_blink_patterns()

raw_after.plot()
plt.show()



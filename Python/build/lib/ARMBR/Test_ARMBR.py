import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.axes_grid1 import (make_axes_locatable, ImageGrid, inset_locator)
import os
import scipy.io as spio

from Library import ProjectOut, ARMBR


import colorednoise as cn
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)


# ==================== GENERATE SYNTHETIC SIGNAL ===============================
fs = 200
np.random.seed(2)

Sig_Len = 10
t = np.linspace(0, Sig_Len, Sig_Len*fs)
Noise_Comp = cn.powerlaw_psd_gaussian(1, len(t))
Blink = np.zeros((1, len(t)))

f, mu, sig = 1, 0.9, 0.1

for i in range(1, int(np.ceil(Sig_Len*1.2)), 2):
    Blink = Blink + np.exp(-0.5*(t-i*mu)**2/(sig**2))

a1, a2, a3, a4 = 3, 8, 10, 20

x1 = 1.06*Noise_Comp
x2 = 1.00*Noise_Comp
x3 = 0.90*Noise_Comp
x4 = 1.01*Noise_Comp
x5 = 0.98*Noise_Comp

y1 = x1
y2 = x2 + a1*Blink
y3 = x3 + a2*Blink
y4 = x4 + a3*Blink
y5 = x5 + a4*Blink

X   = np.vstack((x1, x2, x3, x4, x5))
EEG = np.vstack((y1, y2, y3, y4, y5))

# ==============================================================================



# ================================= ARMBR Method ===============================
[Our_Back_Reg, Set_IQR_Thresh, Blink_Ref, Blink_Artifact] = ARMBR(EEG, [4], 128, -1)
# ==============================================================================



# ================================= Plot Results  ===============================
for i in range(0,5):
    plt.plot(t, X[i,:] + (i+1)*15, 'g', linewidth=0.5)
    plt.plot(t, EEG[i,:] + (i+1)*15, 'r', linewidth=0.5)
    plt.plot(t, Our_Back_Reg[:,i] + (i+1)*15, 'b', linewidth=0.5)
plt.legend(['Brain-wave','Brain-wave + Blink','Brain-wave - Blink'])
plt.show()
# ===============================================================================





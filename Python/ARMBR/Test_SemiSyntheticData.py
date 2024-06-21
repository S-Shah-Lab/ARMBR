from ARMBR_Library import *
import scipy
import mne

sub = 1
fs = 128
EEG_Blink	= scipy.io.loadmat(r"..\..\SemiSyntheticData\Sub"+str(sub)+"\Sub"+str(sub)+"_Synthetic_Blink_Contaminated_EEG.mat")
Clean		= scipy.io.loadmat(r"..\..\SemiSyntheticData\Sub"+str(sub)+"\Sub"+str(sub)+"_Clean_EEG.mat")




ChannelsName = ['Cz','A2','A3','A4','P1','A6','P3','A8','A9','PO7','A11','A12','A13','A14','A15','A16','PO3','CMS',
'Pz','A20','POz','A22','Oz','A24','Iz','A26','A27','O2','A29','PO4','DRL','P2','B1','CP2','B3',
'P4','B5','B6','PO8','B8','B9','P10','P8','B12','P6','TP8','B15','CP6','B17','CP4','B19','C2',
'B21','C4','B23','C6','B25','C8','FT8','B28','FC6','B30','FC4','B32','C1','C2','C3','F4','F6','C6',
'F8','AF8','C9','C10','FC2','F2','C13','C14','AF4','Fp2','Fpz','C18','AFz','C20','Fz','C22','FCz',
'FC1','F1','C26','C27','AF3','Fp1','AF7','C31','C32','D1','D2','D3','F3','F5','D6','F7','FT7','D9',
'FC5','D11','FC3','D13','C1','D15','CP1','D17','D18','C3','D20','C5','D22','T7','TP7','D25',
'CP5','D27','CP3','P5','D30','P7','P9']


EEG_w_Blink = mne.filter.filter_data(EEG_Blink['Sythentic_Blink_Contaminated_EEG'].T, sfreq=fs, l_freq=1, h_freq=40, method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)
EEGGT = mne.filter.filter_data(Clean['Clean_EEG'].T, sfreq=fs, l_freq=1, h_freq=40, method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)


myARMBR = ARMBR(EEG = EEG_w_Blink, Fs  = fs, ChannelsName = ChannelsName, EEGGT=EEGGT)
myARMBR.ARMBR(blink_chan=['fp1', 'fp2']).PerformanceMetrics().DispMetrics().Plot()





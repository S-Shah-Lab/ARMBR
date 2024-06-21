from ARMBR_Library import *
import scipy
import mne

sub = 1
fs = 128
EEG_Blink	= scipy.io.loadmat(r"..\SemiSyntheticData\Sub"+str(sub)+"\Sub"+str(sub)+"_Synthetic_Blink_Contaminated_EEG.mat")
Clean		= scipy.io.loadmat(r"..\SemiSyntheticData\Sub"+str(sub)+"\Sub"+str(sub)+"_Clean_EEG.mat")


ChannelsName = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 
'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 
'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 
'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 
'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 
'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 
'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 
'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']


EEG_w_Blink = mne.filter.filter_data(EEG_Blink['Sythentic_Blink_Contaminated_EEG'].T, sfreq=fs, l_freq=1, h_freq=40, method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)
EEGGT = mne.filter.filter_data(Clean['Clean_EEG'].T, sfreq=fs, l_freq=1, h_freq=40, method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)


myARMBR = ARMBR(EEG = EEG_w_Blink, Fs  = fs, ChannelsName = ChannelsName, EEGGT=EEGGT)
myARMBR.ARMBR(blink_chan=['C16','C29']).PerformanceMetrics().DispMetrics().Plot()





import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, signal
from math import nan
from scipy.stats import pearsonr

class ARMBR:
	Version = '1.0.0'  # @VERSION_INFO@
	
	def __init__(self, 
	EEG				=	np.array([]), 
	Fs				=	None, 
	ChannelsName	=	[],
	ChannelsNameInx	=	[],
	BlinkChannels	=	[],
	Alpha			=	-1):
		
		self.EEG				= EEG
		self.Fs					= Fs
		self.ChannelsName		= ChannelsName
		self.ChannelsNameInx	= ChannelsNameInx
		self.BlinkChannels		= BlinkChannels
		self.BlinkChannelsInx	= BlinkChannels
		self.Alpha 				= Alpha
		

	def import_raw(self, raw):
		# Load EEG data from mne object
		eeg_indices				= [index for index, word in enumerate(raw.get_channel_types()) if word == 'eeg']
		eeg_data				= raw._data[eeg_indices, :]
		
		self.ChannelsName		= raw.ch_names(eeg_indices)
		self.ChannelsNameInx	= raw.ch_names.index(eeg_indices)
		self.EEG 				= reeg_data
		self.Fs  				= raw.info['sfreq']
		
		return self
		
		
	def GetBlinkChannels(self, blink_chan):
		
		self.BlinkChannels = blink_chan
		LowerChannelsName = [ch_name.lower() for ch_name in self.ChannelsName]
			
		blink_ch_id = []
		blink_ch    = []
		for bc in range(len(blink_chan)):
			if blink_chan[bc].lower() in LowerChannelsName:
				index = LowerChannelsName.index(blink_chan[bc].lower())
				blink_ch_id.append(index)
				blink_ch.append(blink_chan[bc].lower())
				
		self.BlinkChannels		= blink_ch
		self.BlinkChannelsInx	= blink_ch_id
		print("Blink channel(s): " + str(self.BlinkChannels))
		
		
		
		
	def ARMBR(self, blink_chan):
		
		# Get blink channels
		self.GetBlinkChannels(blink_chan)
		
		# Run ARMBR
		print('Performing ARMBR')
		X_purged, best_alpha, Bmask, Bc = armbr(self.EEG, self.BlinkChannelsInx, self.Fs, self.Alpha)
		
		self.CleanedEEG	= X_purged
		self.Alpha0		= best_alpha
		self.BlinkComp 	= Bc
		self.BlinkMask	= Bmask
		
		return self
		
		
		
	def PerformanceMetrics(self):
		
		PearCorr = []
		RMSE     = []
		SNR      = []
		
		EEG 	 = rotate_arr(self.EEG)
		CleanedEEG = rotate_arr(self.CleanedEEG) 
				
		for chn in range( np.size(EEG, axis=1) ):
			RMSE.append( np.sqrt(np.mean((EEG[:,chn] - CleanedEEG[:,chn])**2)) )
			SNR.append( 10*np.log10(np.std(EEG[:,chn]) / np.std(EEG[:,chn] - CleanedEEG[:,chn])) )
			correlation, _ = pearsonr(EEG[:,chn], CleanedEEG[:,chn])
			PearCorr.append(correlation)     			
			
		self.RMSE		= RMSE
		self.SNR		= SNR
		self.PearCorr	= PearCorr
		
		return self
		
		
		
		
		
		
	def DispMetrics(self):
		
		print("====================================")
		print("RMSE: " + str(np.round(np.mean(self.RMSE), 2)))
		print("SNR: " + str(np.round(np.mean(self.SNR), 2)))
		print("Pearson correlation: " + str(np.round(np.mean(self.PearCorr), 2)))
		("====================================")
		
		return self
		
		
		
		
		
		
		
def rotate_arr(X):
    if len(X.shape) == 1: X = X[:, np.newaxis]  # Convert 1D array to a column vector
    if X.shape[0] < X.shape[1]:
        return X.T  # Rotate array if channels are on rows
    else:
        return X  # Otherwise, do nothing


def ProjectOut(X, X_reduced, Bmask, maskIn=None):
    # Provides cleaned EEG and blink artifact component
    # X: time series provided to the function
    # X_reduced: time series provided to the function (removed values if Bref is < quartile Qa)
    # Bmask: mask produced by Blink_Selection
    # maskIn: sets some channels to 0 if they are not X channels (could be stim)

    # Ensure that arrays are in the format (data-points x channels)
    X = rotate_arr(X)
    X_reduced = rotate_arr(X_reduced)
    Bmask = rotate_arr(Bmask)

    nSamples, nChannels = X_reduced.shape
    nRefs = Bmask.shape[1]
    
    # This statement should be always True as there are only 4 arguments: if nargin < 5, maskIn = []; end
    if maskIn is None: maskIn = np.ones(nChannels, dtype=bool)
    if maskIn.any() == False: maskIn = np.ones(nChannels, dtype=bool)
    if type(maskIn[0]) != np.bool_:
        if maskIn.min() > 0 and (maskIn.max() > 1 or maskIn.size != nChannels):
            ind = maskIn
            maskIn = np.zeros(nChannels, dtype=bool)
            maskIn[ind] = True
        maskIn = maskIn.astype(bool)
    maskOut = ~maskIn
    if maskIn.size != nChannels: raise ValueError(
        f"the number of channels implied by MASK_IN ({maskIn.size}) does not match the number of columns in X ({nChannels})")
    if Bmask.shape[0] != nSamples: raise ValueError(
        f"the number of rows in REF ({Bmask.shape[0]}) does not match the number of rows in X ({nSamples})")



    I = np.eye(nChannels)  # Identity matrix
    Sigma = np.cov(X_reduced, rowvar=False)  # Covariance matrix
    Sigma[:, maskOut] = I[:, maskOut]  # If not X channels replace columns with I columns
    Sigma[maskOut, :] = I[maskOut, :]  # If not X channels replace rows with I rows
    if not isinstance(X_reduced[0][0], (np.float64, float, np.ndarray)): Bmask = Bmask.astype(float)
    
    
    X_in = np.hstack((X_reduced[:, maskIn], np.ones((nSamples, 1))))
    solution = np.linalg.lstsq(X_in, Bmask, rcond=None)[0]

    bias = solution[-1]
    w = np.zeros((nChannels, nRefs))
    
    # Reshape solution to include excluded channels by maskIn
    w[maskIn, :] = solution[:-1]
    
    # Normalize solution
    rescale = np.sum((Sigma.dot(w)) * (w), axis=0)
    rescale = np.diag(rescale ** -0.5)
    w = w * rescale
    
    # Spatial pattern
    a = Sigma.dot(w)
    
    # Blink artifact spatial components
    M_est = w * a.T
    
    # Blink-suppressed spatial components
    M_purge = I - M_est
    
    # Blink component
    Bc = X.dot(w)
    
    # Blink-suppressed time-series (cleaned output)
    X_purged = X.dot(M_purge)
    
    return M_purge, w, a, Sigma, Bc, X_purged





def Blink_Selection(X, blink_ch_id, alpha0):
    
    # Identify blinks, create blink mask, and obtain projected data set to given EEG time series (C channels, N samples each)
    # X: time series provided to the function
    # blink_ch_id: list of ID of channels tagged for containing eye blinks
    # alpha0: parameter used in automatic blink idenfitication

    # Ensure that X is in the format (data-points x channels)
    # This is not needed as it happens in ARMBR already
    # X = rotate_arr(X)
    
    # Mean across blink channels
    Bref = np.mean(X[:, blink_ch_id], axis=1)

    # if skewed LEFT ==> mirror image to make it skewed RIGHT
    if np.median(Bref) > np.mean(Bref): Bref = -Bref
    
    # Compute the Inter-quartile Range
    Qa = np.quantile(Bref, 0.159)  # -1 sigma
    Qb = np.quantile(Bref, 0.841)  # +1 sigma
    Q2 = np.quantile(Bref, 0.500)  # Median
    StD = (Qb - Qa) / 2  # Semi-IQR
    
    # Threshold for blink detection
    T0 = Q2 + alpha0 * StD  # Median + weighted semi-IQR
    
    # Build a blink mask, which is a train of 1 (blink) and 0 (no blink)
    X_reduced = X[Bref > Qa, :]  # remove data up to -1 sigma
    Bref_reduced = Bref[Bref > Qa]  # remove data up to -1 sigma
    Bmask = Bref_reduced > T0
    Bmask = Bmask.reshape((len(Bmask), 1))
    
    # Obtain cleaned X and blink component
    if np.sum(Bmask) != 0:
        [_, _, _, _, Bc, X_purged] = ProjectOut(X, X_reduced, Bmask)
    else:
        Bc = np.nan
        X_purged = np.nan


    return X_purged, Bc, Bmask



def armbr(X, blink_ch_id, fs, alpha=-1):

    # Apply ARMBR method to given EEG time series (C channels, N samples each)
    # X: time series provided to the function
    # blink_ch_id: list of ID of channels tagged for containing eye blinks
    # fs: X sampling rate
    # alpha: parameter used in automatic blink idenfitication (if -1 then values between 0.01 and 10 are gonna be tested)

    # Ensure that X is in the format (data-points x channels)
    # It's assumed that there are more data-points than channels
    X = rotate_arr(X)
    
    # Used if alpha is not specified (= None)
    if alpha == -1:
        # Store ratio of energies to choose best alpha value
        Delta = []
        alpha_range = np.arange(0.01, 10, 0.1)
        for alpha in alpha_range:
            X_purged, Bc, _ = Blink_Selection(X, blink_ch_id, alpha)
            # if there are no np.nan the statement is True
            if np.isnan(np.sum(Bc)) == False:
                # Bandpass filter between 1 and 8 Hz (FIR filter design using the window method)
                # First parameter in firwin specifies the number of coefficients to use in the filter
                # pass_zero=False: DC component gain set to 0
                LPF = signal.firwin(10, [1, 8], pass_zero=False, fs=fs)
                LPF = signal.filtfilt(LPF, 1, Bc.T).T
                # Energies ratio
                Delta.append(np.sum(LPF ** 2) / (np.sum((Bc - LPF) ** 2)))
            else:
                break
        Delta = np.array(Delta)
        alpha_range = alpha_range[0:len(Delta)]
        # Find optimal alpha
        if len(Delta) > 0:
            best_alpha = alpha_range[np.argmax(Delta)]
            [X_purged, Bc, Bmask] = Blink_Selection(X, blink_ch_id, best_alpha)
        else:
            X_purged = X
            Bc = np.array([])
            Bmask = np.array([])
            best_alpha = np.array([])
    # Used if alpha is specified
    else:
        X_purged, Bc, Bmask = Blink_Selection(X, blink_ch_id, alpha)
        best_alpha = alpha
        
    return X_purged, best_alpha, Bmask, Bc



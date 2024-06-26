from math import nan

import numpy as np
import scipy
from tqdm import tqdm

class ARMBR:
	Version = '1.0.0'  # @VERSION_INFO@
	
	def __init__(self, 
	EEG				=	np.array([]), 
	EEGGT           = None,
	CleanedEEG      = None,
	Fs				=	None, 
	ChannelsName	=	[],
	ChannelsNameInx	=	[],
	BlinkChannels	=	[],
	Alpha			=	-1):
		
		self.EEG				= EEG
		self.EEGGT       	    = EEGGT
		self.CleanedEEG    		= EEGGT
		self.Fs					= Fs
		self.ChannelsName		= ChannelsName
		self.ChannelsNameInx	= ChannelsNameInx
		self.BlinkChannels		= BlinkChannels
		self.BlinkChannelsInx	= BlinkChannels
		self.Alpha 				= Alpha
		

	def ImportFromRaw(self, raw):
		# Load EEG data from mne object
		eeg_indices				= [index for index, word in enumerate(raw.get_channel_types()) if word == 'eeg']
		eeg_data				= raw._data[eeg_indices, :]
		
		ChannelsName = []
		for i in eeg_indices:
			ChannelsName.append(raw.ch_names[i])
		self.ChannelsName	 	= ChannelsName
		self.ChannelsNameInx	= eeg_indices
		eeg_data 	   = rotate_arr(eeg_data)
		self.EEG 				= eeg_data
		self.Fs  				= raw.info['sfreq']
		
		return self
		
	def UnloadIntoRaw(self, raw):
		raw._data = self.CleanedEEG.T
		
		return self, raw

		
	def GetBlinkChannels(self, blink_chan):
		
		# check if blink_chan contains only integers or only strings
		
		
		all_int = all(s.isdigit() for s in blink_chan)
		all_str = all(isinstance(element, str) for element in blink_chan)
		
		if all_int:
			blink_ch_id = []
			for bc in range(len(blink_chan)):
				blink_ch_id.append(int(blink_chan[bc]))
			self.BlinkChannelsInx = blink_ch_id
			print("Blink channel(s): " + str(self.BlinkChannelsInx))

		elif all_str:
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
		if len(self.BlinkChannelsInx) > 0:
			X_purged, best_alpha, Bmask, Bc, BlinkSpatialPattern = armbr(self.EEG, self.BlinkChannelsInx, self.Fs, self.Alpha)
			
			X_purged = rotate_arr(X_purged)
			self.CleanedEEG	= X_purged
			self.Alpha0		= best_alpha
			self.BlinkComp 	= Bc
			self.BlinkMask	= Bmask
			self.BlinkSpatialPattern = BlinkSpatialPattern
			
		else:
			print('No blink channels were identified. ARMBR was not performed.\n' )
		
		return self
		
		
	def ApplyBlinkSpatialPattern(self, BlinkSpatialPattern):
		
		EEG = self.EEG 
		EEG = rotate_arr(EEG)
		
		self.CleanedEEG = EEG.dot(BlinkSpatialPattern)

		return self
		


		
	def PerformanceMetrics(self):
		
		if self.EEGGT is None:
			print('EEG ground truth is not available')
		
		else: 
			PearCorr = []
			RMSE     = []
			SNR      = []
			
			CleanedEEG 	   = rotate_arr(self.CleanedEEG)
			EEGGT          = rotate_arr(self.EEGGT) 
					
			for chn in range( np.size(EEGGT, axis=1) ):
				RMSE.append( np.sqrt(np.mean((EEGGT[:,chn] - CleanedEEG[:,chn])**2)) )
				SNR.append( 10*np.log10(np.std(EEGGT[:,chn]) / np.std(EEGGT[:,chn] - CleanedEEG[:,chn])) )
				correlation, _ = scipy.stats.pearsonr(EEGGT[:,chn], CleanedEEG[:,chn])
				PearCorr.append(correlation)     			
				
			self.RMSE		= RMSE
			self.SNR		= SNR
			self.PearCorr	= PearCorr
			
		return self
		
		
		
		
		
		
	def DispMetrics(self):
		if self.EEGGT is None:
			print('No ground truth or metrics.')
			
		else:
			print("====================================")
			print("RMSE: " + str(np.round(np.mean(self.RMSE), 2)))
			print("SNR: " + str(np.round(np.mean(self.SNR), 2)))
			print("Pearson correlation: " + str(np.round(np.mean(self.PearCorr), 2)))
			("====================================")
		
		return self
		
		
		
	def Plot(self):
		
		import matplotlib.pyplot as plt
		
		mvup=np.max(np.std(self.EEG))*10
		CleanedEEG = rotate_arr(self.CleanedEEG)
		EEG = rotate_arr(self.EEG)
		
		for chn in range( np.size(CleanedEEG, axis=1) ):
			plt.subplot(1,2,1)
			plt.plot(EEG[:,chn] - mvup*chn, 'r')
			#plt.ylim([-0.011, -0.007])	
			plt.yticks([])
			plt.xlabel('time (s)')
			plt.title('Before ARMBR')
			
			plt.subplot(1,2,2)
			plt.plot(CleanedEEG[:,chn] - mvup*chn, 'k')
			#plt.ylim([-0.011, -0.007])	
			plt.yticks([])
			plt.xlabel('time (s)')
			plt.title('After ARMBR')
			
			
		
		plt.show()
			
		
		
		
		
		
#=============================================

def MaxAmp(data_vector, fs, window_size=15, shift_size=15):
	"""
	Calculate the absolute maximum amplitude of window_size second long data segments.

	Parameters:
	- data_vector: numpy array or list, time-series data of N samples
	- fs: float, sampling frequency in Hz
	- window_size: float, size of data window to be processed in seconds (default: 15 seconds)
	- shift_size: float, shift amount to get to the next window in seconds (default: 15 seconds)

	Returns:
	- max_amp: list of M values, absolute maximum of the window_size second long data segments
	"""

	# Convert window sizes from seconds to points
	window_size_pts = int(window_size * fs)
	shift_size_pts = int(shift_size * fs)

	max_amp = []

	# Iterate over the data_vector in segments defined by window_size and shift_size
	for i in range(0, len(data_vector), shift_size_pts + 1):
		start = i
		finish = min(i + window_size_pts, len(data_vector))

		window_data = data_vector[start:finish]
		max_amp.append(np.max(np.abs(window_data)))

	return max_amp

	

def Segment(data_vector, fs, window_size=15, shift_size=15):
	"""
	Segment the data_vector into window_size second long data segments.

	Parameters:
	- data_vector: numpy array or list, time-series data of N samples
	- fs: float, sampling frequency in Hz
	- window_size: float, size of data window to be processed in seconds (default: 15 seconds)
	- shift_size: float, shift amount to get to the next window in seconds (default: 15 seconds)

	Returns:
	- segmented_data: list of M arrays, window_size second long data segments
	"""

	# Convert window sizes from seconds to points
	window_size_pts = int(window_size * fs)
	shift_size_pts = int(shift_size * fs)

	segmented_data = []

	# Iterate over the data_vector in segments defined by window_size and shift_size
	for i in range(0, len(data_vector), shift_size_pts + 1):
		start = i
		finish = min(i + window_size_pts, len(data_vector))

		window_data = data_vector[start:finish]
		segmented_data.append(window_data)

	return segmented_data
	
	
def DataSelect(data_vector, init_size=3, std_dev_threshold=5):
	"""
	Filter out outliers from the data vector based on a given threshold and initial size.

	Parameters:
	- data_vector: numpy array or list, vector of N values from which outliers are to be eliminated
	- init_size: int, number of data points used for initialization (default: 3)
	- std_dev_threshold: float, threshold over which points are considered outliers (default: 5)

	Returns:
	- filtered_data: list of values excluding the outliers
	- filtered_data_inx: list of indices of the values excluding the outliers
	- excluded_points: list of the excluded outliers
	- excluded_points_inx: list of indices of the excluded outliers
	"""

	if len(data_vector) == 0:
		raise ValueError('The input data_vector must not be empty.')

	# Initialize arrays to store filtered data
	filtered_data = []
	filtered_data_inx = []

	excluded_points = []
	excluded_points_inx = []

	data_vector_ = np.array(data_vector)

	# Initialize with the first few points
	for j in range(min(init_size, len(data_vector))):
		filtered_data.append(data_vector_[j])
		filtered_data_inx.append(j)

	# Iterate through the data
	for i in range(init_size, len(data_vector_)):
		# Extract the previous points
		previous_points = data_vector_[:i]

		# Calculate mean and standard deviation of the previous points
		mean_prev = np.mean(previous_points)
		std_dev_prev = np.std(previous_points)

		# Check if the current point is within the standard deviation threshold
		if abs(data_vector_[i] - mean_prev) <= std_dev_threshold * std_dev_prev:
			# Include the current point in the filtered data
			filtered_data.append(data_vector_[i])
			filtered_data_inx.append(i)
		else:
			# Exclude the current point
			data_vector_[i] = mean_prev
			excluded_points.append(data_vector_[i])
			excluded_points_inx.append(i)

	return filtered_data, filtered_data_inx, excluded_points, excluded_points_inx
	
	
	
	
def Data_Prep(eeg, fs, blink_chan_nbr):
	"""
	Prepare EEG data by filtering out segments affected by blinks.

	Parameters:
	- eeg: numpy array, multi-channel time-series (samples by channels)
	- fs: float, sampling frequency in Hz
	- blink_chan_nbr: list of int, indices of channels most affected by blinks

	Returns:
	- GoodEEG: numpy array, multi-channel time-series of good EEG signals
	- OrigEEG: numpy array, original multi-channel time-series as input EEG
	- GoodBlinks: numpy array, time-series of good blink signal
	"""
	if eeg.shape[0] < eeg.shape[1]:
		eeg = eeg.T
	OrigEEG = eeg

	Blink = np.mean(eeg[:, blink_chan_nbr], axis=1)

	if np.median(Blink) > np.mean(Blink):
		Blink = -Blink

	BlinkAmp = MaxAmp(np.diff(Blink), fs)
	Blink_epochs = Segment(Blink, fs)
	eeg_epochs = Segment(eeg, fs)
	_, filtered_data_pts, _, _ = DataSelect(BlinkAmp)

	GoodBlinks = []
	GoodEEG = []

	for g in range(len(Blink_epochs)):
		if g in filtered_data_pts:
			GoodBlinks.append(Blink_epochs[g])
			GoodEEG.append(eeg_epochs[g])

	GoodBlinks = np.concatenate(GoodBlinks)
	GoodEEG = np.concatenate(GoodEEG, axis=0)

	return GoodEEG, OrigEEG, GoodBlinks
	
	
	

def rotate_arr(X):
	if len(X.shape) == 1: X = X[:, np.newaxis]  # Convert 1D array to a column vector
	if X.shape[0] < X.shape[1]:
		return X.T  # Rotate array if channels are on rows
	else:
		return X  # Otherwise, do nothing


def projectout(X, X_reduced, Bmask, maskIn=None):
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



def Blink_Selection(eeg_orig, eeg_filt, Blink_filt, alpha0, maskIn=None):
	"""
	Selects and suppresses blink artifacts from EEG data.

	Parameters:
	- eeg_orig: numpy array, multi-channel time-series (samples by channels) containing all EEG data
	- eeg_filt: numpy array, multi-channel time-series (samples by channels) containing a reduced set of the original EEG
	- Blink_filt: numpy array, time-series containing a reduced set of the blink reference signal
	- alpha0: float, optimal blink level threshold
	- maskIn: numpy array, vector of the same size as the number of channels (1 for included, 0 for excluded)

	Returns:
	- no_blink_eeg: numpy array, multi-channel time-series after blinks are suppressed
	- Blink_Artifact: numpy array, time-series of the blink component that is removed from the EEG data matrix
	- ref: numpy array, train of pulses at the blink location
	"""
	nChannels = eeg_orig.shape[1]

	if maskIn is None:
		maskIn = np.ones(nChannels, dtype=int)  # all channels are used

	# Compute the Inter-quartile Range
	Qa = np.quantile(Blink_filt, 0.159)
	Qb = np.quantile(Blink_filt, 0.841)
	Q2 = np.quantile(Blink_filt, 0.5)

	StD = (Qb - Qa) / 2
	T0 = Q2 + alpha0 * StD

	# Build a reference signal, which is a train of 1 and 0, where 1 indicates a blink and 0 indicates no blink
	eeg_reduced = eeg_filt[Blink_filt > Qa, :]
	Blink_reduced = Blink_filt[Blink_filt > Qa]
	ref = Blink_reduced > T0

	# Project out blink components
	if np.sum(ref) != 0:
		BlinkSpatialPattern, _, _, _, Blink_Artifact, no_blink_eeg = projectout(eeg_orig, eeg_reduced, ref, maskIn)
	else:
		Blink_Artifact = np.array([])
		no_blink_eeg = np.array([])
		BlinkSpatialPattern = np.array([])

	return no_blink_eeg, Blink_Artifact, ref, BlinkSpatialPattern




def armbr(X, blink_ch_id, fs, alpha=-1):

	# Apply ARMBR method to given EEG time series (C channels, N samples each)
	# X: time series provided to the function
	# blink_ch_id: list of ID of channels tagged for containing eye blinks
	# fs: X sampling rate
	# alpha: parameter used in automatic blink idenfitication (if -1 then values between 0.01 and 10 are gonna be tested)

	# Ensure that X is in the format (data-points x channels)
	# It's assumed that there are more data-points than channels
	X = rotate_arr(X)
	
	# Segment the EEG data into smaller segments and remove bad segments
	good_eeg, _, good_blinks =  Data_Prep(X, fs, blink_ch_id)

	
	# Used if alpha is not specified (= None)
	if alpha == -1:
		# Store ratio of energies to choose best alpha value
		Delta = []
		alpha_range = np.arange(0.01, 10, 0.1)
		for alpha in tqdm(alpha_range, desc="Running ARMBR"):
			
		#for alpha in alpha_range:
		
		
			X_purged, Bc, _ , _= Blink_Selection(X, good_eeg, good_blinks, alpha)

			
			# if there are no np.nan the statement is True
			if (np.isnan(np.sum(Bc)) == False) and (len(Bc)>0):
				# Bandpass filter between 1 and 8 Hz (FIR filter design using the window method)
				# First parameter in firwin specifies the number of coefficients to use in the filter
				# pass_zero=False: DC component gain set to 0
				LPF = scipy.signal.firwin(10, [1, 8], pass_zero=False, fs=fs)
				LPF = scipy.signal.filtfilt(LPF, 1, Bc.T).T
				# Energies ratio
				Delta.append(np.sum(LPF ** 2) / (np.sum((Bc - LPF) ** 2)))
			else:
				break
		Delta = np.array(Delta)
		alpha_range = alpha_range[0:len(Delta)]
		# Find optimal alpha
		if len(Delta) > 0:
			best_alpha = alpha_range[np.argmax(Delta)]
			[X_purged, Bc, Bmask, BlinkSpatialPattern] = Blink_Selection(X, good_eeg, good_blinks, best_alpha)
		else:
			X_purged = X
			Bc = np.array([])
			Bmask = np.array([])
			best_alpha = None
			BlinkSpatialPattern = np.array([])
	# Used if alpha is specified
	else:
		X_purged, Bc, Bmask, BlinkSpatialPattern = Blink_Selection(X, good_eeg, good_blinks, alpha)
		best_alpha = alpha
		
	return X_purged, best_alpha, Bmask, Bc, BlinkSpatialPattern



# Authors: Ludvik Alkhoury <Ludvik.alkhoury@gmail.com>
# License: BSD-3-Clause

from math import nan
import numpy as np
import copy
import scipy
from mne.io import BaseRaw
from mne.utils import logger, verbose, _check_preload, ProgressBar
from mne import pick_channels, pick_types

__version__ = '2.0.0'  # @VERSION_INFO@


class ARMBR:
	
	def __init__(	self, 
					ch_name	=	None,
					alpha		=	-1.0):
		
		self.ch_name	= ch_name or []
		self.alpha 		= alpha
		self.is_fitted	= False  
		
		
	@verbose
	def fit(	self, 
				raw, 
				picks	=	"eeg", 
				start	=	None, 
				stop	=	None, 
				verbose	=	None
			):
		"""Fit ARMBR model using selected data from raw instance.

		Parameters
		----------
		raw : instance of mne.io.BaseRaw
			The raw EEG data.
		picks : str | list | slice | None
			Channels to include. Defaults to 'eeg'.
		start : int | None
			Sample index to start from (if manually specifying segment).
		stop : int | None
			Sample index to stop at (if manually specifying segment).
		verbose : bool | str | int | None
			Control verbosity of the logging output.
		"""		

		if start is not None and stop is not None:
			# User provided manual segment (in samples)
			data = raw.get_data(picks=picks, start=start, stop=stop)
			t_start = start / raw.info['sfreq']
			t_stop = (stop - 1) / raw.info['sfreq']
			logger.info(f"Using manual segment from {t_start:.2f}s to {t_stop:.2f}s.")
		else:
			n_samples = raw.n_times
			mask = np.ones(n_samples, dtype=bool)

			# Step 1: Drop BAD_ segments
			for annot in raw.annotations:
				if annot['description'].startswith("BAD_"):
					onset, duration = annot['onset'], annot['duration']
					bad_start, bad_stop = raw.time_as_index([onset, onset + duration])
					mask[bad_start:bad_stop] = False
					logger.info(f"Dropped {annot['description']} segment: {onset:.2f}s to {onset + duration:.2f}s")

			# Step 2: Include only armbr_fit segments
			armbr_annots = [
				annot for annot in raw.annotations 
				if "armbr_fit" in annot['description'].lower()
			]

			if not armbr_annots:
				# No armbr_fit found, use all non-BAD data
				data = raw.get_data(picks=picks)[:, mask]
				total_secs = mask.sum() / raw.info['sfreq']
				logger.info(f"No 'armbr_fit' found. Using {total_secs:.2f} seconds of non-BAD data.")
			else:
				segments = []
				for annot in armbr_annots:
					onset, duration = annot['onset'], annot['duration']
					seg_start, seg_stop = raw.time_as_index([onset, onset + duration])
					segment_mask = mask[seg_start:seg_stop]
					if np.any(segment_mask):
						segment_data = raw.get_data(picks=picks, start=seg_start, stop=seg_stop)
						segments.append(segment_data[:, segment_mask])
						start_sec = seg_start / raw.info['sfreq']
						stop_sec = (seg_stop - 1) / raw.info['sfreq']
						duration_sec = segment_mask.sum() / raw.info['sfreq']
						logger.info(f"Included armbr_fit: {start_sec:.2f}s to {stop_sec:.2f}s ({duration_sec:.2f}s used)")
				data = np.concatenate(segments, axis=1) if segments else np.empty((len(picks), 0))

		# Save output to class variables
		self._eeg_data			= _rotate_arr(data)
		self.sfreq  			= raw.info['sfreq']
		self.ch_names	 		= raw.ch_names
		self._channel_indices	= [i for i, ch_type  in enumerate(raw.get_channel_types()) if ch_type == 'eeg']
		self._eeg_indices		= pick_types(raw.info, eeg=True)
		
		self._run_armbr(self.ch_name)
		self.is_fitted = True  
		
		logger.info("ARMBR model fitting complete.")
		
		return self
		
	
	@verbose
	def apply(self, raw, picks="eeg", verbose=None):
		"""Apply ARMBR blink removal to raw EEG data.

		Parameters
		----------
		raw : instance of mne.io.BaseRaw
			The raw data to clean.
		picks : str | list | None
			Channel picks to apply ARMBR to. Defaults to 'eeg'.
		verbose : bool | str | int | None
			Control verbosity of the logging output.

		Returns
		-------
		self : instance of ARMBR
			Returns the current instance with ARMBR applied.
		"""
		
		if not getattr(self, "is_fitted", False):
			raise RuntimeError("You must call .fit() before .apply().")
	
		
		# Check if raw is preloaded (required to modify data)
		_check_preload(raw, 'apply')
	
		eeg_raw		= _rotate_arr( raw.get_data(picks=picks) )
		eeg_clean	= eeg_raw .dot(self.blink_spatial_pattern)
		
		# Apply cleaned data back to Raw object
		raw.apply_function(lambda x: eeg_clean.T, picks=picks, channel_wise=False)
		
		logger.info("ARMBR blink suppression applied to raw data.")
		
		return self
		
		
	def plot(self, show=True, verbose=None):
		"""Plot EEG signals before and after ARMBR cleaning.

		Parameters
		----------
		show : bool
			Whether to display the figure immediately (default: True).
		verbose : bool | str | int | None
			Control verbosity of the logging output.
			
		Returns
		-------
		fig : matplotlib.figure.Figure
			The matplotlib figure containing the plots.
		"""

		if not getattr(self, "is_fitted", False):
			raise RuntimeError("You must call .fit() before .plot().")
		
		import matplotlib.pyplot as plt
		
		# Prepare data
		raw_eeg = _rotate_arr(self._eeg_data)
		cleaned = _rotate_arr(self.cleaned_eeg)

		n_channels = raw_eeg.shape[1]
		offset = np.max(np.std(raw_eeg)) * 10

		fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

		# Plot original EEG
		for idx in range(n_channels):
			axes[0].plot(raw_eeg[:, idx] - offset * idx, color='r')
		axes[0].set_title("Before ARMBR")
		axes[0].set_xlabel("Time (samples)")
		axes[0].set_yticks([])

		# Plot cleaned EEG
		for idx in range(n_channels):
			axes[1].plot(cleaned[:, idx] - offset * idx, color='k')
		axes[1].set_title("After ARMBR")
		axes[1].set_xlabel("Time (samples)")
		axes[1].set_yticks([])

		fig.suptitle("ARMBR Cleaning Results", fontsize=14)

		if show:
			plt.tight_layout()
			plt.show()
		
		logger.info("Plotted before/after ARMBR EEG.")
		return self, fig
		
		

	
	def copy(self):
		"""Create a deep copy of the ARMBR instance.

		Returns
		-------
		inst : instance of ARMBR
			A deep copy of the current object.
		"""
		inst = copy.deepcopy(self)
		logger.info("ARMBR object copied.")
		return inst
			




	def _prep_blink_channels(self, blink_chs):
		"""Resolve blink channel names or indices to internal format.

		Parameters
		----------
		blink_chs : list of str | list of int
			List of blink channels as names or indices.
		"""
		is_all_int = all(isinstance(ch, int) or (isinstance(ch, str) and ch.isdigit()) for ch in blink_chs)
		is_all_str = all(isinstance(ch, str) for ch in blink_chs)

		if is_all_int:
			self.ch_name_inx = [int(ch) for ch in blink_chs]
			logger.info(f"Blink channels (indices): {self.ch_name_inx}")

		elif is_all_str:
			self.ch_name = blink_chs  # Save user-specified names
			lower_all_ch_names = [name.lower() for name in self.ch_names]

			ch_indices = []
			valid_names = []
			for ch in blink_chs:
				ch_lower = ch.lower()
				if ch_lower in lower_all_ch_names:
					idx = lower_all_ch_names.index(ch_lower)
					ch_indices.append(idx)
					valid_names.append(self.ch_names[idx])

			self.ch_name = valid_names
			self.ch_name_inx = ch_indices
			logger.info(f"Blink channels (names): {self.ch_name}")

		else:
			raise ValueError("Blink channel list must contain only channel names or only indices.")

					
			

		
	def _run_armbr(self, blink_chs):
		"""Run the ARMBR blink removal algorithm.

		Parameters
		----------
		blink_chs : list of str | list of int
			Blink channel names or indices.
		"""
		# Resolve channel names or indices
		self._prep_blink_channels(blink_chs)

		if len(self.ch_name_inx) > 0:
			# Apply ARMBR
			x_purged, best_alpha, blink_mask, blink_comp, spatial_pattern = _armbr(
				self._eeg_data, self.ch_name_inx, self.sfreq, self.alpha
			)

			# Store outputs
			self.cleaned_eeg = _rotate_arr(x_purged)
			self.best_alpha = best_alpha
			self.blink_mask = blink_mask
			self.blink_comp = blink_comp
			self.blink_spatial_pattern = spatial_pattern

		else:
			raise RuntimeError("No blink channels were identified. ARMBR was not performed.")
			
		return self
	
		
		
# ============================================================
# Internal utility functions (not intended for end-user access)
# ============================================================

def _rotate_arr(X):
	"""Ensure EEG array has shape (n_samples, n_channels).

	Parameters
	----------
	X : ndarray
		Input EEG array. Can be 1D, (n_channels, n_samples), or already (n_samples, n_channels).

	Returns
	-------
	X_out : ndarray
		EEG array in shape (n_samples, n_channels).
	"""
	X = np.asarray(X)

	if X.ndim == 1:
		X = X[:, np.newaxis]  # Convert to column vector

	if X.shape[0] < X.shape[1]:
		return X.T  # Transpose to make samples on rows
	return X


def _max_amp(data, sfreq, window_size=15, shift_size=15):
	"""Compute maximum absolute amplitude over sliding windows.

	Parameters
	----------
	data : array-like, shape (n_samples,)
		1D time-series data.
	sfreq : float
		Sampling frequency in Hz.
	window_size : float
		Length of each window in seconds. Default is 15.
	shift_size : float
		Step size between windows in seconds. Default is 15.

	Returns
	-------
	max_values : list of float
		Maximum absolute amplitudes for each window.
	"""
	window_pts = int(window_size * sfreq)
	shift_pts = int(shift_size * sfreq)

	max_values = []
	for start in range(0, len(data), shift_pts + 1):
		stop = min(start + window_pts, len(data))
		window = data[start:stop]
		max_values.append(np.max(np.abs(window)))

	return max_values


	
def _segment(data, sfreq, window_size=15, shift_size=15):
	"""Segment time-series data into overlapping windows.

	Parameters
	----------
	data : array-like, shape (n_samples,) or (n_samples, ...)
		Time-series data to segment.
	sfreq : float
		Sampling frequency in Hz.
	window_size : float
		Length of each segment in seconds. Default is 15.
	shift_size : float
		Step size between segments in seconds. Default is 15.

	Returns
	-------
	segments : list of ndarray
		List of segmented arrays of shape (window_pts, ...)
	"""
	window_pts = int(window_size * sfreq)
	shift_pts = int(shift_size * sfreq)

	segments = []
	for start in range(0, len(data), shift_pts + 1):
		stop = min(start + window_pts, len(data))
		segments.append(data[start:stop])

	return segments

	
	
def _data_select(data, init_size=3, std_threshold=5.0):
	"""Filter outliers from a 1D signal based on standard deviation threshold.

	Parameters
	----------
	data : array-like, shape (n_samples,)
		Input data vector to filter.
	init_size : int
		Number of initial points used to estimate baseline statistics. Default is 3.
	std_threshold : float
		Standard deviation threshold for excluding outliers. Default is 5.0.

	Returns
	-------
	filtered_data : list of float
		Values retained after outlier removal.
	filtered_indices : list of int
		Indices of the retained values.
	excluded_values : list of float
		Values excluded as outliers.
	excluded_indices : list of int
		Indices of the excluded outliers.
	"""
	if len(data) == 0:
		raise ValueError("The input data vector must not be empty.")

	data = np.array(data)
	filtered_data = []
	filtered_indices = []
	excluded_values = []
	excluded_indices = []

	# Initialize with first few points
	for j in range(min(init_size, len(data))):
		filtered_data.append(data[j])
		filtered_indices.append(j)

	# Iterate through remaining points
	for i in range(init_size, len(data)):
		mean_prev = np.mean(data[:i])
		std_prev = np.std(data[:i])

		if abs(data[i] - mean_prev) <= std_threshold * std_prev:
			filtered_data.append(data[i])
			filtered_indices.append(i)
		else:
			data[i] = mean_prev  # Replace in-place (if needed downstream)
			excluded_values.append(data[i])
			excluded_indices.append(i)

	return filtered_data, filtered_indices, excluded_values, excluded_indices

	
	
	
def _data_prep(eeg, sfreq, blink_indices):
	"""Prepare EEG data by extracting segments with clean blink signals.

	Parameters
	----------
	eeg : ndarray, shape (n_samples, n_channels) or (n_channels, n_samples)
		Raw EEG data.
	sfreq : float
		Sampling frequency in Hz.
	blink_indices : list of int
		Indices of channels most affected by blinks.

	Returns
	-------
	good_eeg : ndarray
		Filtered EEG data segments with good blink content.
	orig_eeg : ndarray
		Original EEG data (possibly transposed).
	good_blinks : ndarray
		Blink reference signal (1D) from clean segments.
	"""
	# Ensure EEG is (n_samples, n_channels)
	if eeg.shape[0] < eeg.shape[1]:
		eeg = eeg.T
	orig_eeg = eeg.copy()

	# Construct average blink reference
	blink_signal = np.mean(eeg[:, blink_indices], axis=1)

	# Invert if median > mean
	if np.median(blink_signal) > np.mean(blink_signal):
		blink_signal = -blink_signal

	# Get blink-related metrics and segments
	blink_amp = _max_amp(np.diff(blink_signal), sfreq)
	blink_epochs = _segment(blink_signal, sfreq)
	eeg_epochs = _segment(eeg, sfreq)

	# Select segments with acceptable blink amplitude
	_, good_indices, _, _ = _data_select(blink_amp)

	good_blinks = []
	good_eeg = []

	for i in good_indices:
		good_blinks.append(blink_epochs[i])
		good_eeg.append(eeg_epochs[i])

	good_blinks = np.concatenate(good_blinks)
	good_eeg = np.concatenate(good_eeg, axis=0)

	return good_eeg, orig_eeg, good_blinks

	

def _projectout(X, X_reduced, blink_mask, mask_in=None):
	"""Project out blink components from multichannel EEG data.

	Parameters
	----------
	X : ndarray, shape (n_samples, n_channels)
		Original EEG time-series.
	X_reduced : ndarray, shape (n_samples, n_channels)
		Subset of EEG data used for estimating covariance.
	blink_mask : ndarray, shape (n_samples, n_refs)
		Binary mask identifying blink occurrences.
	mask_in : ndarray of bool | list of int | None
		Mask indicating which channels to use in the projection.
		If None, all channels are included.

	Returns
	-------
	M_purge : ndarray, shape (n_channels, n_channels)
		Projection matrix to suppress blink components.
	w : ndarray, shape (n_channels, n_refs)
		Projection weights (spatial filters).
	a : ndarray, shape (n_channels, n_refs)
		Spatial patterns of blink artifacts.
	sigma : ndarray, shape (n_channels, n_channels)
		Covariance matrix used in projection.
	blink_comp : ndarray, shape (n_samples, n_refs)
		Estimated blink components.
	x_purged : ndarray, shape (n_samples, n_channels)
		Blink-suppressed EEG.
	"""
	# Ensure correct shapes
	X = _rotate_arr(X)
	X_reduced = _rotate_arr(X_reduced)
	blink_mask = _rotate_arr(blink_mask)
	blink_mask = blink_mask.astype(float, copy=False)  # <- restored here

	n_samples, n_channels = X_reduced.shape
	n_refs = blink_mask.shape[1]

	# Handle mask_in logic
	if mask_in is None:
		mask_in = np.ones(n_channels, dtype=bool)
	else:
		mask_in = np.asarray(mask_in)
		if mask_in.dtype != bool:
			if mask_in.min() > 0 and (mask_in.max() > 1 or mask_in.size != n_channels):
				indices = mask_in
				mask_in = np.zeros(n_channels, dtype=bool)
				mask_in[indices] = True
		mask_in = mask_in.astype(bool)

	# Input validation
	if mask_in.size != n_channels:
		raise ValueError(f"mask_in size {mask_in.size} does not match number of channels {n_channels}.")
	if blink_mask.shape[0] != n_samples:
		raise ValueError(f"blink_mask sample count {blink_mask.shape[0]} does not match X sample count {n_samples}.")

	mask_out = ~mask_in
	eye = np.eye(n_channels)
	sigma = np.cov(X_reduced, rowvar=False)

	# Replace non-included rows and columns in covariance matrix
	sigma[:, mask_out] = eye[:, mask_out]
	sigma[mask_out, :] = eye[mask_out, :]

	# Solve regression
	X_in = np.hstack([X_reduced[:, mask_in], np.ones((n_samples, 1))])
	solution = np.linalg.lstsq(X_in, blink_mask, rcond=None)[0]

	bias = solution[-1]
	w = np.zeros((n_channels, n_refs))
	w[mask_in, :] = solution[:-1]

	# Normalize
	rescale = np.sum((sigma @ w) * w, axis=0)
	rescale = np.diag(rescale ** -0.5)
	w = w @ rescale

	a = sigma @ w
	M_est = w @ a.T
	M_purge = eye - M_est
	blink_comp = X @ w
	x_purged = X @ M_purge

	return M_purge, w, a, sigma, blink_comp, x_purged



def _blink_selection(eeg_orig, eeg_filt, blink_filt, alpha, mask_in=None):
	"""Select and suppress blink artifacts from EEG data.

	Parameters
	----------
	eeg_orig : ndarray, shape (n_samples, n_channels)
		Original EEG data including all time points.
	eeg_filt : ndarray, shape (n_samples, n_channels)
		Subset of EEG data to use for blink suppression.
	blink_filt : ndarray, shape (n_samples,)
		Reference blink signal (e.g., averaged frontal signal).
	alpha : float
		Threshold scaling factor for blink detection.
	mask_in : ndarray of bool | list of int | None
		Boolean mask or list of channel indices to include. If None, all channels used.

	Returns
	-------
	eeg_clean : ndarray, shape (n_samples, n_channels)
		EEG after blink suppression.
	blink_artifact : ndarray, shape (n_samples,)
		Time series of blink artifact estimated and removed.
	ref_mask : ndarray, shape (n_samples,)
		Binary mask of blink locations in the reference signal.
	blink_pattern : ndarray, shape (n_channels, 1)
		Spatial pattern of blink artifact.
	"""
	n_channels = eeg_orig.shape[1]

	if mask_in is None:
		mask_in = np.ones(n_channels, dtype=int)

	# Compute inter-quartile statistics
	Qa = np.quantile(blink_filt, 0.159)
	Qb = np.quantile(blink_filt, 0.841)
	Q2 = np.quantile(blink_filt, 0.5)
	std_iqr = (Qb - Qa) / 2
	T0 = Q2 + alpha * std_iqr

	# Build reference mask (binary vector of blink positions)
	reduced_eeg = eeg_filt[blink_filt > Qa, :]
	reduced_blink = blink_filt[blink_filt > Qa]
	ref_mask = reduced_blink > T0

	# Project out blink if ref_mask contains any positive sample
	if np.sum(ref_mask) != 0:
		blink_pattern, _, _, _, blink_artifact, eeg_clean = _projectout(
			eeg_orig, reduced_eeg, ref_mask, mask_in
		)
	else:
		eeg_clean = np.array([])
		blink_artifact = np.array([])
		blink_pattern = np.array([])

	return eeg_clean, blink_artifact, ref_mask, blink_pattern



def _armbr(X, blink_ch_idx, sfreq, alpha=-1.0):
	"""Run ARMBR blink removal on multichannel EEG data.

	Parameters
	----------
	X : ndarray, shape (n_samples, n_channels)
		Input EEG data.
	blink_ch_idx : list of int
		Indices of blink-related channels to use for reference.
	sfreq : float
		Sampling frequency in Hz.
	alpha : float
		Blink detection threshold scaling factor. If -1, automatically optimized.

	Returns
	-------
	x_clean : ndarray, shape (n_samples, n_channels)
		EEG with blink artifacts removed.
	best_alpha : float or None
		Optimal alpha value found (or input alpha if not optimized).
	ref_mask : ndarray
		Binary mask indicating blink locations.
	blink_comp : ndarray
		Extracted blink component.
	blink_pattern : ndarray
		Spatial pattern of the blink.
	"""
	X = _rotate_arr(X)
	good_eeg, _, good_blinks = _data_prep(X, sfreq, blink_ch_idx)

	if alpha == -1:
		alpha_range = np.arange(0.01, 10, 0.1)
		energy_ratios = []

		with ProgressBar(alpha_range, mesg='Running ARMBR') as pb:
			for test_alpha in pb:
				x_tmp, blink_tmp, _, _ = _blink_selection(X, good_eeg, good_blinks, test_alpha)

				if blink_tmp.size > 0 and not np.isnan(np.sum(blink_tmp)):
					bpf = scipy.signal.firwin(10, [1, 8], pass_zero=False, fs=sfreq)
					blink_filt = scipy.signal.filtfilt(bpf, 1, blink_tmp.T).T
					ratio = np.sum(blink_filt ** 2) / np.sum((blink_tmp - blink_filt) ** 2)
					energy_ratios.append(ratio)
				else:
					break

		energy_ratios = np.array(energy_ratios)
		alpha_range = alpha_range[:len(energy_ratios)]

		if energy_ratios.size > 0:
			best_alpha = alpha_range[np.argmax(energy_ratios)]
			x_clean, blink_comp, ref_mask, blink_pattern = _blink_selection(X, good_eeg, good_blinks, best_alpha)
		else:
			x_clean = X
			blink_comp = np.array([])
			ref_mask = np.array([])
			blink_pattern = np.array([])
			best_alpha = None

	else:
		x_clean, blink_comp, ref_mask, blink_pattern = _blink_selection(X, good_eeg, good_blinks, alpha)
		best_alpha = alpha

	return x_clean, best_alpha, ref_mask, blink_comp, blink_pattern


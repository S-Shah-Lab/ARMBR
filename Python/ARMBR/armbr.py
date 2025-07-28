# Authors:	Ludvik Alkhoury <Ludvik.alkhoury@gmail.com>
#			Giacomo Scanavini <scanavini.giacomo@gmail.com>
#			N. Jeremy Hill <jezhill@gmail.com>
# License: BSD-3-Clause

"""
Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements

ARMBR is a lightweight and easy-to-use method for blink artifact removal from EEG signals using multivariate backward regression. 
The algorithm detects the times at which eye blinks occur and then estimates their linear scalp projection by regressing a simplified, 
time-locked reference signal against the multichannel EEG. This projection is used to suppress blink-related components while preserving 
underlying brain signals. ARMBR requires minimal training data, does not depend on dedicated EOG channels, and operates robustly in both 
offline and real-time (online) settings, including BCI applications.

This module implements the ARMBR algorithm, described in:

Alkhoury L, Scanavini G, Louviot S, Radanovic A, Shah SA, Hill NJ. (2025).
"Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements."
Journal of Neural Engineering, 22(3), 036048.
https://doi.org/10.1088/1741-2552/ade566

(see ARMBR.bibtex for the BibTeX entry)


The core algorithm supports both standalone and MNE integration via the `ARMBR` class:
from armbr import run_armbr   # core non-MNE-dependent code (just needs numpy)
from armbr import ARMBR       # MNE-compatible wrapper class
"""



import copy

import numpy as np

try: import mne
except:
	VERBOSE = lambda x: x
	class LOGGER:
		def info(x): print(x)
else:
	import mne.utils; from mne.utils import verbose as VERBOSE, logger as LOGGER
	import mne.filter

__version__ = '2.0.7'  # @VERSION_INFO@


class ARMBR:
	"""
	MNE-compatible wrapper for ARMBR, a blink artifact removal algorithm for EEG.

	ARMBR identifies and suppresses blink artifacts using multivariate backward
	regression. It leverages a binarized blink reference signal to estimate and
	project out blink-related spatial components. The method requires only a small
	amount of training data and works without requiring ICA or dedicated EOG
	channels. It is especially suitable for real-time and online BCI scenarios.

	Parameters
	----------
	ch_name : list of str | list of int | None
		Names or indices of EEG channels highly affected by blinks (e.g. Fp1, Fp2).
		These are used to construct the blink reference signal. If None, defaults to [].
	alpha : float | 'auto'
		Blink detection threshold factor. If 'auto', ARMBR selects the threshold
		automatically using method described in Section 4. Threshold selection (see Alkhoury et al., 2025).

	References
	----------
	L. Alkhoury et al. (2025), Journal of Neural Engineering, 22(3), 036048.
	https://doi.org/10.1088/1741-2552/ade566

	"""
	
	bibtex = """
		@article{alkhoury2025_armbr,
			title={Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements},
			author={Alkhoury, Ludvik and Scanavini, G and Louviot, S and Radanovic, A and Shah, SA and Hill, NJ},
			journal={Journal of Neural Engineering},
			volume={22},
			number={3},
			pages={036048},
			year={2025},
			doi={10.1088/1741-2552/ade566}
		}
	"""
	
	def __init__(	self, 
					ch_name =	None,
					alpha   =	'auto',
					):
		"""
		Initialize ARMBR with optional blink channel specification and threshold strategy.

		Parameters
		----------
		ch_name : list of str | list of int | None
			Names or indices of EEG channels expected to show prominent blink activity.
			Common choices include Fp1, Fp2, or frontal channels near the eyes.
		alpha : float | 'auto'
			Blink detection threshold multiplier. If 'auto', the optimal threshold
			is selected automatically by maximizing the low-frequency-to-high-frequency
			energy ratio in the extracted blink component.

		Notes
		-----
		The user-specified channels should capture prominent blink deflections.
		No EOG channels are required.
		"""
		self.ch_name	= ch_name or []
		self.alpha 		= alpha
		self.is_fitted	= False  
		
		
	@VERBOSE
	def fit(	self, 
				raw, 
				picks	=	"eeg", 
				start	=	None, 
				stop	=	None, 
				verbose	=	None
			):
		"""
		Fit the ARMBR model using raw EEG data.

		This method prepares the data for blink artifact removal by identifying
		clean EEG segments, extracting the relevant samples, and computing the
		ARMBR spatial projection. Training can use either manually specified 
		time ranges, annotated segments labeled "armbr_fit", or all non-rejected
		data (excluding 'BAD_' annotations).

		Parameters
		----------
		raw : instance of mne.io.BaseRaw
			Continuous raw EEG recording.
		picks : str | list | slice | None
			Channels to include. Defaults to 'eeg'. Can be a string like 'eeg',
			a list of channel names, or indices.
		start : float | None
			Start time in seconds for fitting. If provided, overrides annotations
			and uses this exact time range.
		stop : float | None
			Stop time in seconds for fitting. If provided with `start`, overrides
			annotations and uses this exact time range.
		verbose : bool | str | int | None
			Controls verbosity of the log output.

		Returns
		-------
		self : instance of ARMBR
			The fitted instance, with spatial weights and state stored.

		Notes
		-----
		If 'armbr_fit' annotations are present, those segments are prioritized.
		Otherwise, all data not marked by 'BAD_' annotations is used. If both
		`start` and `stop` are provided, these override annotations entirely.
		The model must be fitted before calling `.apply()` or `.plot()`.
		"""

		
		if start is not None and stop is not None:
			# User provided manual segment (in samples)
			data = raw.get_data(picks=picks, start=int(start*raw.info['sfreq']), stop=int(stop*raw.info['sfreq'])) 
			LOGGER.info("Using manual segment from {:.2f}s to {:.2f}s.".format(start, stop))
		else:
			n_samples = raw.n_times
			mask = np.ones(n_samples, dtype=bool)

			# Step 1: Drop BAD_ segments
			for annot in raw.annotations:
				if annot['description'].startswith("BAD_"):
					onset, duration = annot['onset'], annot['duration']
					print(onset)
					print(raw.first_time)

					bad_start, bad_stop = raw.time_as_index([onset - raw.first_time, onset + duration - raw.first_time])
					mask[bad_start:bad_stop] = False
					LOGGER.info("Dropped {} segment: {:.2f}s to {:.2f}s".format(
									annot['description'],
									bad_start / raw.info['sfreq'],
									bad_stop / raw.info['sfreq']
								))


			# Step 2: Include only armbr_fit segments
			armbr_annots = [
				annot for annot in raw.annotations 
				if annot['description'].lower() == "armbr_fit"
			]

			if not armbr_annots:
				# No armbr_fit found, use all non-BAD data
				data = raw.get_data(picks=picks)[:, mask]
				total_secs = mask.sum() / raw.info['sfreq']
				LOGGER.info("No 'armbr_fit' found. Using {:.2f} seconds of non-BAD data.".format(total_secs))
			else:
				segments = []
				for annot in armbr_annots:
					onset, duration = annot['onset'], annot['duration']
					seg_start, seg_stop = raw.time_as_index([onset- raw.first_time, onset + duration- raw.first_time])
					segment_mask = mask[seg_start:seg_stop]
					if np.any(segment_mask):
						segment_data = raw.get_data(picks=picks, start=seg_start, stop=seg_stop)
						segments.append(segment_data[:, segment_mask])
						start_sec = (seg_start / raw.info['sfreq']) 
						stop_sec = ((seg_stop - 1) / raw.info['sfreq'])  

						duration_sec = segment_mask.sum() / raw.info['sfreq']
						LOGGER.info("Included armbr_fit: {:.2f}s to {:.2f}s ({:.2f}s used)".format(start_sec, stop_sec, duration_sec))

				data = np.concatenate(segments, axis=1) if segments else np.empty((len(picks), 0))

		# Save output to class variables
		self._eeg_data			= _rotate_arr(data)
		self.sfreq  			= raw.info['sfreq']
		self.ch_names	 		= raw.ch_names
		self._channel_indices	= [i for i, ch_type  in enumerate(raw.get_channel_types()) if ch_type == 'eeg']
		self._eeg_indices		= [raw.ch_names.index(ch) for ch in raw.copy().pick('eeg').ch_names]
		self._raw_info			= raw.info
		
		self._run_armbr(self.ch_name)
		self.is_fitted = True  
		
		LOGGER.info("ARMBR model fitting complete.")
		
		return self
		
	
	@VERBOSE
	def apply(self, raw, picks="eeg", verbose=None):
		"""
		Apply ARMBR blink removal to raw EEG data.
		
		This method removes blink artifacts from the specified channels using
		the spatial projection matrix estimated during `.fit()`. The projection
		is applied directly to the data via MNE's `apply_function`.

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

		Raises
		------
		RuntimeError
			If `.fit()` has not been called before this method.
		ValueError
			If the raw object is not preloaded (since in-place editing is required).

		Notes
		-----
		This modifies the input `raw` object in-place. To preserve the original,
		use `.copy()` before calling `.apply()`.
		"""

		import mne.utils
		
		if not getattr(self, "is_fitted", False):
			raise RuntimeError("You must call .fit() before .apply().")
	
		
		# Check if raw is preloaded (required to modify data)
		mne.utils._check_preload(raw, 'apply')
	
		eeg_raw		= _rotate_arr( raw.get_data(picks=picks) )
		eeg_clean	= eeg_raw.dot(self.blink_removal_matrix)
		
		# Apply cleaned data back to Raw object
		raw.apply_function(lambda x: eeg_clean.T, picks=picks, channel_wise=False)
		
		LOGGER.info("ARMBR blink suppression applied to raw data.")
		
		return self
		
		
	@VERBOSE
	def plot(self, show=True, verbose=None):
		"""
		Plot EEG signals before and after ARMBR cleaning.
		
		This diagnostic plot shows side-by-side traces of the original and
		ARMBR-cleaned EEG data from the training set. It helps visually assess
		the effectiveness of the blink suppression.

		Parameters
		----------
		show : bool
			Whether to display the figure immediately (default: True).
		verbose : bool | str | int | None
			Control verbosity of the logging output.
			
		Returns
		-------
		self : instance of ARMBR
			Returns the current instance for chaining.
		fig : matplotlib.figure.Figure
			Matplotlib figure containing the before/after subplots.

		Raises
		------
		RuntimeError
			If `.fit()` has not been called before this method.

		Notes
		-----
		Red traces indicate raw EEG. Black traces show cleaned data.
		Each channel is vertically offset for readability.
		This plot only shows data used during training (`fit()`).
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
		time_samples = np.arange(len(raw_eeg[:, 0]))/self.sfreq
		
		for idx in range(n_channels):
			axes[0].plot(time_samples, raw_eeg[:, idx] - offset * idx, color='r')
		axes[0].set_title("Before ARMBR")
		axes[0].set_xlabel("Time (samples)")
		axes[0].set_yticks([])
		axes[0].set_xlim([time_samples[0], time_samples[-1]])
		
		# Plot cleaned EEG
		for idx in range(n_channels):
			axes[1].plot(time_samples, cleaned[:, idx] - offset * idx, color='k')
		axes[1].set_title("After ARMBR")
		axes[1].set_xlabel("Time (samples)")
		axes[1].set_yticks([])
		axes[1].set_xlim([time_samples[0], time_samples[-1]])

		fig.suptitle("ARMBR Cleaning Results", fontsize=14)

		if show:
			plt.tight_layout()
			plt.show()
		
		LOGGER.info("Plotted before/after ARMBR EEG.")
		return self, fig
		
	

	def plot_blink_patterns(self, show=True):
		
		"""
		Visualize the spatial blink components estimated by ARMBR.

		This method generates topographic plots of each component in the
		blink spatial pattern matrix, which represent how blinks manifest
		across the EEG scalp sensors.

		Parameters
		----------
		show : bool
			If True, the plot is displayed immediately. Default is True.

		Returns
		-------
		self : instance of ARMBR
			Returns the current instance for chaining.
		plt : module
			The matplotlib.pyplot module with the figure and axes created.

		Notes
		-----
		This function uses `mne.viz.plot_topomap` to display scalp maps for
		each spatial component. The colormap is centered at zero and scaled
		symmetrically based on the maximum absolute pattern value.
		"""

		import matplotlib.pyplot as plt
		from mpl_toolkits.axes_grid1 import make_axes_locatable


		n_components = self.blink_spatial_pattern.shape[1]
		abs_max = np.max(np.abs(self.blink_spatial_pattern))
		order = int(np.floor(np.log10(abs_max))) if abs_max > 0 else 0
		precision = 10 ** -(order - 1)   

		clim_val = np.ceil(abs_max * precision) / precision  # symmetric rounded clim
		vlim = (-clim_val, clim_val)

		n_cols = 1
		n_rows = n_components

		fig, axes = plt.subplots(n_rows, n_cols, figsize=(4, 2.5 * n_rows))
		if n_components == 1:
			axes = np.array([axes])
		axes = axes.flatten()

		# Create placeholder for colorbar axis
		divider = make_axes_locatable(axes[-1])
		cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

		for i in range(n_components):
			pattern = self.blink_spatial_pattern[:, i].squeeze()

			im, cm = mne.viz.plot_topomap(
				pattern,
				self._raw_info,
				ch_type='eeg',
				sensors=True,
				contours=False,
				cmap='RdBu_r',
				axes=axes[i],
				show=False,
				vlim=vlim
			)
			axes[i].set_title('Component {}'.format(i + 1), fontsize=10)


		# Shared colorbar
		clim = dict(kind='value', lims=[-clim_val, 0, clim_val])
		cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
		cbar.ax.set_ylabel("Pattern value", rotation=270, labelpad=15)

		plt.tight_layout()
		if show:
			plt.show()

		return self, plt






	
	def copy(self):
		"""
		Create a deep copy of the ARMBR instance.
		
		This is useful for preserving the current model state before applying
		changes such as refitting or reapplying to different data.

		Returns
		-------
		inst : instance of ARMBR
			A deep copy of the current object, including all internal state.

		Notes
		-----
		The copied instance is independent of the original and can be modified
		or reapplied without affecting the original.
		"""
		inst = copy.deepcopy(self)
		LOGGER.info("ARMBR object copied.")
		return inst
			




	def _prep_blink_channels(self, blink_chs):
		"""
		Standardize user-specified blink channels to internal index format.

		This method resolves the `blink_chs` argument passed to ARMBR into
		canonical numeric indices based on `self.ch_names`. Accepts either
		a list of integers (indices) or a list of strings (channel names).

		Parameters
		----------
		blink_chs : list of str | list of int
			Blink-affected EEG channels. Can be specified by name or index.

		Raises
		------
		ValueError
			If the input is a mix of names and indices, or contains invalid types.

		Notes
		-----
		This method sets `self.ch_name` and `self.ch_name_inx` attributes
		based on the resolved input.
		"""

		is_all_int = all(isinstance(ch, int) or (isinstance(ch, str) and ch.isdigit()) for ch in blink_chs)
		is_all_str = all(isinstance(ch, str) for ch in blink_chs)

		if is_all_int:
			self.ch_name_inx = [int(ch) for ch in blink_chs]
			LOGGER.info("Blink channels (indices): {}".format(self.ch_name_inx))

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
			LOGGER.info("Blink channels (names): {}".format(self.ch_name))

		else:
			raise ValueError("Blink channel list must contain only channel names or only indices.")





	def _run_armbr(self, blink_chs):
		"""
		Internal method to execute the ARMBR algorithm on training data.

		This function estimates blink components by performing multivariate
		backward regression using a simplified binary reference signal constructed
		from the specified blink channels. The result includes a projection matrix
		that can be used to suppress blink artifacts in new data.

		Parameters
		----------
		blink_chs : list of str | list of int
			Names or indices of channels used to construct the blink reference.

		Returns
		-------
		self : instance of ARMBR
			Returns the instance with fitted model attributes stored.

		Raises
		------
		RuntimeError
			If no valid blink channels could be resolved.

		Notes
		-----
		Stores internal variables including:
		- `cleaned_eeg`: blink-suppressed training data
		- `best_alpha`: threshold used (if auto)
		- `blink_mask`: binary vector marking blink segments
		- `blink_comp`: blink time course (latent variable)
		- `blink_spatial_pattern`: scalp patterns
		- `blink_removal_matrix`: projection operator for cleaning
		"""

		# Resolve channel names or indices
		self._prep_blink_channels(blink_chs)
		
		if self.alpha.lower() == 'auto':
			alpha = -1
		else:
			alpha = int(alpha)

		if len(self.ch_name_inx) > 0:
			# Apply ARMBR
			x_purged, best_alpha, blink_mask, blink_comp, blink_spatial_pattern, blink_removal_matrix = run_armbr(
				self._eeg_data, self.ch_name_inx, self.sfreq, alpha
			)

			# Store outputs
			self.cleaned_eeg = _rotate_arr(x_purged)
			self.best_alpha = best_alpha
			self.blink_mask = blink_mask
			self.blink_comp = blink_comp
			self.blink_spatial_pattern = blink_spatial_pattern
			self.blink_removal_matrix = blink_removal_matrix

		else:
			raise RuntimeError("No blink channels were identified. ARMBR was not performed.")
			
		return self
	



def run_armbr(X, blink_ch_idx, sfreq, alpha=-1.0):
	"""
	Run ARMBR blink artifact removal on multichannel EEG data.

	This function implements the full ARMBR algorithm pipeline on continuous EEG
	data. It identifies blink artifacts from frontal EEG channels, extracts a
	low-rank blink component, estimates its spatial distribution, and subtracts
	it from the signal using multivariate backward regression.

	Parameters
	----------
	X : ndarray, shape (n_samples, n_channels)
		Input EEG data. Each row is a time sample; each column is a channel.
	blink_ch_idx : list of int
		Indices of channels strongly affected by blinks (e.g. Fp1, Fp2).
	sfreq : float
		Sampling frequency in Hz.
	alpha : float
		Threshold multiplier for blink detection. If set to -1, the algorithm
		performs automatic optimization based on the energy ratio of low- to
		high-frequency components.

	Returns
	-------
	x_clean : ndarray, shape (n_samples, n_channels)
		EEG data with blink artifacts removed.
	best_alpha : float or None
		The optimal alpha value used. If alpha was set manually, this matches
		the input; if optimized, it is the best value found.
	ref_mask : ndarray, shape (n_samples,)
		Binary mask marking detected blink events.
	blink_comp : ndarray, shape (n_samples,)
		Time course of the extracted blink component.
	blink_pattern : ndarray, shape (n_channels,)
		Estimated spatial topography of the blink.
	blink_removal_matrix : ndarray, shape (n_channels, n_channels)
		Spatial projection matrix used to remove blink activity.

	Notes
	-----
	When `alpha=-1`, the algorithm sweeps a range of values to maximize the
	low-frequency to high-frequency energy ratio of the blink component, as
	described in Alkhoury et al. (2025). The returned `blink_removal_matrix`
	can be applied to other EEG data for online or batch blink suppression.
	"""
	
	try:
		import mne.utils, mne.filter
		USE_MNE = True
	except ImportError:
		import scipy.signal as signal
		USE_MNE = False
	

	
	X = _rotate_arr(X)
	good_eeg, _, good_blinks = _data_prep(X, sfreq, blink_ch_idx)

	if alpha == -1:
		alpha_range = np.arange(0.01, 10, 0.1)
		energy_ratios = []
		
		iterator = mne.utils.ProgressBar(alpha_range, mesg='Running ARMBR') if USE_MNE else alpha_range
			

		for test_alpha in iterator:
			x_tmp, blink_tmp, _, _, _ = _blink_selection(X, good_eeg, good_blinks, test_alpha)

			if blink_tmp.size > 0 and not np.isnan(np.sum(blink_tmp)):
				if USE_MNE:
					blink_filt = mne.filter.filter_data(
						blink_tmp.T, sfreq=sfreq, l_freq=1, h_freq=8,
						method='iir', iir_params=dict(order=4, ftype='butter'),
						verbose=False
					).T
				else:
					# Use FIR filter with scipy if MNE is unavailable
					#bpf = scipy.signal.firwin(101, [1, 8], pass_zero=False, fs=sfreq)
					#blink_filt = scipy.signal.filtfilt(bpf, 1, blink_tmp.T).T
					
					sos1 = signal.butter(N=4, Wn=[40], btype='lowpass', fs=sfreq, output='sos')
					sos2 = signal.butter(N=4, Wn=[1], btype='highpass', fs=sfreq, output='sos')

					filt_blink_tmp = signal.sosfiltfilt(sos1, blink_tmp.T)
					blink_filt = signal.sosfiltfilt(sos2, filt_blink_tmp).T
			
	

				ratio = np.sum(blink_filt ** 2) / np.sum((blink_tmp - blink_filt) ** 2)
				energy_ratios.append(ratio)
			else:
				break

		energy_ratios = np.array(energy_ratios)
		alpha_range = alpha_range[:len(energy_ratios)]

		if energy_ratios.size > 0:
			best_alpha = alpha_range[np.argmax(energy_ratios)]
			x_clean, blink_comp, ref_mask, blink_pattern, blink_removal_matrix = _blink_selection(X, good_eeg, good_blinks, best_alpha)
		else:
			x_clean = X
			blink_comp = np.array([])
			ref_mask = np.array([])
			blink_pattern = np.array([])
			best_alpha = None

	else:
		x_clean, blink_comp, ref_mask, blink_pattern, blink_removal_matrix = _blink_selection(X, good_eeg, good_blinks, alpha)
		best_alpha = alpha

	return x_clean, best_alpha, ref_mask, blink_comp, blink_pattern, blink_removal_matrix

	
# ============================================================
# Internal utility functions (not intended for end-user access)
# ============================================================

def _rotate_arr(X):
	"""
	Ensure EEG array is in (n_samples, n_channels) shape.

	This helper function standardizes the orientation of EEG data,
	so that time samples are along the first axis and channels along
	the second. It handles 1D arrays, transposes if necessary, and
	returns the adjusted array.

	Parameters
	----------
	X : ndarray
		Input EEG data. Can be 1D, (n_channels, n_samples), or (n_samples, n_channels).

	Returns
	-------
	X_out : ndarray
		Output array in shape (n_samples, n_channels).

	Notes
	-----
	This is useful when loading or manipulating EEG matrices, since
	EEG libraries often vary in dimension conventions.
	"""

	X = np.asarray(X)

	if X.ndim == 1:
		X = X[:, np.newaxis]  # Convert to column vector

	if X.shape[0] < X.shape[1]:
		return X.T  # Transpose to make samples on rows
	return X


def _max_amp(data, sfreq, window_size=15, shift_size=15):
	"""
	Compute maximum absolute amplitude over sliding windows.

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

	Notes
	-----
	This function is used to assess blink signal amplitude variability
	for segment selection in ARMBR training.
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
	"""
	Segment time-series data into overlapping windows.

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
		List of windows (each of shape (window_pts, ...)) extracted from input data.
	"""
	window_pts = int(window_size * sfreq)
	shift_pts = int(shift_size * sfreq)

	segments = []
	for start in range(0, len(data), shift_pts + 1):
		stop = min(start + window_pts, len(data))
		segments.append(data[start:stop])

	return segments

	
	
def _data_select(data, init_size=3, std_threshold=5.0):
	"""
	Filter outliers from a 1D signal based on standard deviation threshold.

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
	"""
	Prepare EEG data by extracting segments with clean blink signals.

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
	
	Notes
	-----
	This function handles preprocessing for the blink training pipeline, including
	signal polarity alignment, segmentation, and amplitude-based filtering.
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
	"""
	Project out blink components from multichannel EEG data.

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

	Notes
	-----
	This is the core projection engine of ARMBR. It uses spatial filtering
	to identify and remove blink topographies while preserving neural activity.
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
		raise ValueError("mask_in size {} does not match number of channels {}.".format(mask_in.size, n_channels))
	if blink_mask.shape[0] != n_samples:
		raise ValueError("blink_mask sample count {} does not match X sample count {}.".format(blink_mask.shape[0], n_samples))

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
	eeg_clean : ndarray
		Blink-suppressed EEG.
	blink_artifact : ndarray
		Extracted blink artifact waveform.
	ref_mask : ndarray
		Binary mask marking blink time points.
	blink_pattern : ndarray
		Spatial blink topography.
	blink_removal_matrix : ndarray
		Projection matrix used to remove blink activity.

	Notes
	-----
	Uses IQR-based thresholding on the blink signal to detect blink events,
	then estimates and removes corresponding spatial blink components.
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
		blink_removal_matrix, _, blink_pattern, _, blink_artifact, eeg_clean = _projectout(
			eeg_orig, reduced_eeg, ref_mask, mask_in
		)
	else:
		eeg_clean = np.array([])
		blink_artifact = np.array([])
		blink_pattern = np.array([])
		blink_removal_matrix = np.array([])

	return eeg_clean, blink_artifact, ref_mask, blink_pattern, blink_removal_matrix



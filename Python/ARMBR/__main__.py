"""
This is the Python implementation of the blink removal method from EEG signals; ARMBR. 
The EEG data types that the code supposed are: .fif, .edf, and .dat files.

Before you run the code, make sure you are in the Python directory of the ARMBR repository.

If you want to use the indices of the blink reference channels then use below, where -c "79,92" represents indices 79 and 92:
python -m ARMBR -f "../SemiSyntheticData/Sub1/Sub1_Synthetic_Blink_Contaminated_EEG.fif" -c "79,92" --plot

If you want to use the name of the blink reference channels then use below, where -c "C16,C29" represents channel name C16 and C29:
python -m ARMBR -f "../SemiSyntheticData/Sub1/Sub1_Synthetic_Blink_Contaminated_EEG.fif" -c "C16,C29" --plot

python -m ARMBR requires the mne package.
The ARMBR module alone needs either mne or scipy.

Code by Ludvik Alkhoury, Giacomo Scanavini, and Jeremy hill
June 25, 2024--

"""
import os
import sys
import time
import shutil
import argparse
import warnings

parser1 = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter, prog='python -m ARMBR')
parser1.add_argument( "-f", "--fit",      			default='', type=str,	help='Full path of the EEG data to train ARMBR on. Supported formats are .fif, .edf, and .dat. You can also specify a .txt file containing spatial-filter weights previously saved by --save-weights.')
parser1.add_argument( "-a", "--apply",    			default='', type=str,	help='Full path of the EEG data to apply ARMBR to. Supported formats are .fif, .edf, and .dat.')
parser1.add_argument( "-c", "--blink-channels", 	default='', type=str,	help='Names or indices of blink reference channel(s).')
parser1.add_argument( "-e", "--exclude-channels", 	default='', type=str,	help='Names or indices of channel(s) to exclude from spatial filter calculation. The spatial filter will pass these through unchanged.')
parser1.add_argument( "--save-eeg", 				default='', type=str,	help='Use to save the EEG data after blink removal as a .fif mne raw object.')
parser1.add_argument( "--save-weights", 			default='', type=str,	help='Use to save the spatial filter weights computed from ARMBR.')
parser1.add_argument( "--BCI2000", 					default='', type=str,	help='Go into BCI2000-support GUI mode, targeting the specified BCI2000 distribution root dir.')
parser1.add_argument( "--plot", 					action = 'store_true',	help='Use to plot the cleaned EEG signals of --apply.')
parser1.add_argument( "--version", 					action = 'store_true',	help='Print the package version and exit.')
parser1.add_argument( "--install-bci2000-demo",		action = 'store_true',	help='Install `ARMBR_Fit.bat` and `ARMBR_Apply.bat` into the BCI2000 distribution root directory.')


OPTS1 = parser1.parse_args()

from ARMBR import ARMBR, __version__
from ARMBR import load_bci2000_weights, save_bci2000_weights


def load_data(filename):
	
	file_extension = os.path.splitext(filename)[1]
	
	if file_extension == '.fif':
		raw = mne.io.read_raw_fif(filename, preload=True)
		return raw
		
	elif file_extension == '.edf':
		raw = mne.io.read_raw_edf(filename, preload=True)
		return raw
		
	elif file_extension == '.dat':
		reader = None
		try:
			from BCI2000Tools.FileReader import bcistream
		except:
			from BCI2kReader import BCI2kReader as b2k
			reader	      = b2k.BCI2kReader(filename)
			eeg_data      = reader.signals
			sampling_rate = reader.samplingrate
			ch_names      = reader.parameters['ChannelNames']
		else:
			reader           = bcistream(filename)
			eeg_data, states = reader.decode()
			sampling_rate    = reader.samplingfreq_hz
			ch_names         = reader.params.ChannelNames
		finally:
			if reader is not None: reader.close()
		info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types='eeg')
		raw = mne.io.RawArray(np.asarray(eeg_data)*1e-6, info)
		return raw
	
	elif file_extension == '.txt':
		blink_removal_matrix = np.loadtxt(filename)
		return blink_removal_matrix
		
	elif file_extension == '.prm':
		try: blink_removal_matrix, channel_names = load_bci2000_weights(filename)
		except ValueError as err: raise SystemExit(err)
		return blink_removal_matrix
	else:
		raise SystemExit('At the moment this code supports files of type .fif, .edf, .dat and .txt.')


#===============================================================================


if OPTS1.version:
	print( 'ARMBR %s' % __version__ )
	sys.exit( 0 )

if OPTS1.install_bci2000_demo:
	from ARMBR.BCI2000GUI import install_demo
	try: sys.exit(install_demo(OPTS1.BCI2000, show=True))
	except Exception as err: raise SystemExit(err)

if OPTS1.BCI2000:
	from ARMBR.BCI2000GUI import run_gui
	for opt in 'save_eeg save_weights plot'.split():
		if getattr( OPTS1, opt ): raise SystemExit( "The --%s option is not supported in --BCI2000 GUI mode." % opt.replace( '_', '-' ) )
	sys.exit( run_gui( bci2000root=OPTS1.BCI2000, data_file_path=OPTS1.fit, blink_channels=OPTS1.blink_channels ) )

if not OPTS1.fit:
	raise SystemExit('no training data or weights file specified')

import numpy as np
try: import mne 
except: raise SystemExit('For this mode of operation, you will first need to:  python -m pip install mne')

fit_data = load_data( OPTS1.fit )
print(' ')
blink_channels		= OPTS1.blink_channels.replace(',',' ').split()
exclude_channels	= OPTS1.exclude_channels.replace(',',' ').split()

if not OPTS1.apply:
	OPTS1.apply = OPTS1.fit

loaded_weights_directly = isinstance(fit_data, np.ndarray)
if not blink_channels and not loaded_weights_directly:
	raise SystemExit('no blink channels specified')
	
if isinstance(fit_data, mne.io.BaseRaw):
	myARMBR = ARMBR(blink_channels)
	myARMBR.fit(fit_data, exclude=exclude_channels)
	raw_apply	= load_data( OPTS1.apply )
	if OPTS1.plot: before = raw_apply.copy()
	myARMBR.apply(raw_apply)
	
elif loaded_weights_directly:
	spatial_filters_as_rows = fit_data.T
	raw_apply	= load_data( OPTS1.apply )
	if OPTS1.plot: before = raw_apply.copy()
	mne.utils._check_preload(raw_apply, 'apply') # Check if raw is preloaded (required to modify data)
	eeg_data	= raw_apply.get_data(picks='eeg')
	eeg_clean	= spatial_filters_as_rows.dot(eeg_data)
	raw_apply.apply_function(lambda x: eeg_clean, picks='eeg', channel_wise=False) # Apply cleaned data back to Raw object
	
else:
	raise ValueError("Wrong extension.")




# Save weights
if OPTS1.save_weights: 
	print( 'saving weights to ' + OPTS1.save_weights )
	if OPTS1.save_weights.lower().endswith( '.prm' ):
		if loaded_weights_directly:
			print( r'/!\ cannot save weights to .prm format without channel information from the original data file' )
		else:
			save_bci2000_weights(myARMBR.blink_removal_matrix, fit_data.info['ch_names'], OPTS1.save_weights,
								 training_file_name=OPTS1.fit, blink_channels=blink_channels, exclude_channels=exclude_channels)
	else:
		np.savetxt(OPTS1.save_weights, myARMBR.blink_removal_matrix, fmt="%.10f")


if OPTS1.save_eeg:
	raw_apply.save(OPTS1.save_eeg)
else:
	print('Data not saved.')
			
# Plot EEG before and after blink removal
if OPTS1.plot:
	try: import matplotlib.pyplot as plt
	except: plt = None # if we got this far, then I guess mne must be using a different plotting backend?
	# except: raise SystemExit( 'To support the --plot option, you need to do:   python -m pip install matplotlib' )
	if plt and 'IPython' in sys.modules: plt.ion()
	for title, dataset in [ ( 'before', before ), ( 'after', raw_apply ) ]:
		dataset.copy().filter( l_freq=1, h_freq=40, method='iir', 
							   iir_params=dict(order=4, ftype='butter'), 
							   verbose=False).plot( title=title, scalings=50e-6 )
	if plt and 'IPython' not in sys.modules: plt.show()


	
	
	


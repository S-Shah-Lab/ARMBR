"""
This is the Python implementation of the blink removal method from EEG signals; ARMBR. 
The EEG data types that the code supposed are: .fif, .edf, and .dat files.

Before you run the code, make sure you are in the Python directory of the ARMBR repository.

If you want to use the indices of the blink reference channels then use below, where -c "79,92" represents indices 79 and 92:
python -m ARMBR -p "..\SemiSyntheticData\Sub1\Sub1_Synthetic_Blink_Contaminated_EEG.fif" -c "79,92" --plot

If you want to use the name of the blink reference channels then use below, where -c "C16,C29" represents channel name C16 and C29:
python -m ARMBR -p "..\SemiSyntheticData\Sub1\Sub1_Synthetic_Blink_Contaminated_EEG.fif" -c "C16,C29" --plot

python -m ARMBR requires the mne package.
The ARMBR module alone needs either mne or scipy.

Code by Ludvik Alkhoury, Giacomo Scanavini, and Jeremy hill
June 25, 2024

"""
import argparse
import warnings
import os
import sys


parser1 = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter, prog='python -m ARMBR')
# TODO: maybe break --data-path into --fit and --apply paths (and one supported format for --fit might be a file that only contains spatial-filter weights, previously saved by --saved-weights)
parser1.add_argument( "-p", "--data-path",      default='', type=str, help='Full path of the EEG data. At this point, the code supports .fif, .edf, and .dat data.')
parser1.add_argument( "-c", "--blink-channels", default='', type=str, help='Names or indices or blink reference channel(s).')
#parser1.add_argument( '-f', '--filter-band', metavar='l_freq_hz, h_freq_hz', default='1,40', type=str, help="String of cutoff for lower and upper frequency limits of the EEG (and the acoustic envelope, if used). Pass `None,None` to turn off filtering." )
parser1.add_argument( "--save", action = 'store_true', help='Use to save the EEG data after blink removal as a .fif mne raw object.')
parser1.add_argument( "--save-path", default='', type=str, help='Directory where data is saved.')
parser1.add_argument( "--plot", action = 'store_true', help='Use to plot the cleaned EEG signals.')
parser1.add_argument( "--version", action = 'store_true', help='Print the package version and exit.')
parser1.add_argument( "--BCI2000", default='', type=str, help='Go into BCI2000-support GUI mode, targeting the specified BCI2000 distribution root dir.')

OPTS1 = parser1.parse_args()


from ARMBR import ARMBR, __version__

if OPTS1.version:
	print( 'ARMBR %s' % __version__ )
	sys.exit( 0 )

if OPTS1.BCI2000:
	from ARMBR.BCI2000GUI import RunGUI
	for opt in 'save save_path plot'.split():
		if getattr( OPTS1, opt ): raise SystemExit( "The --%s option is not supported in --BCI2000 GUI mode." % opt.replace( '_', '-' ) )
	sys.exit( RunGUI( bci2000root=OPTS1.BCI2000, data_file_path=OPTS1.data_path, blink_channels=OPTS1.blink_channels ) )


import numpy as np
import mne 

filter_band		= [float(f) for f in OPTS1.filter_band.replace(',',' ').split()] # TODO: remove
blink_channels	= OPTS1.blink_channels.replace(',',' ').split()

if len(OPTS1.data_path) > 0:
	if len(blink_channels) > 0:
		
		file_extension = os.path.splitext(OPTS1.data_path)[1]
		
		if file_extension == '.fif':
			raw = mne.io.read_raw_fif(OPTS1.data_path, preload=True)
		
		elif file_extension == '.edf':
			raw = mne.io.read_raw_edf(OPTS1.data_path, preload=True)
		
		elif file_extension == '.dat':
			
			from BCI2kReader import BCI2kReader as b2k
			
			reader	      = b2k.BCI2kReader(OPTS1.data_path)
			eeg_data      = reader.signals
			sampling_rate = reader.samplingrate
			ch_names      = reader.parameters['ChannelNames']
			
			info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types='eeg')
			raw = mne.io.RawArray(eeg_data*1e-6, info)
			
		raw.filter(l_freq=filter_band[0], h_freq=filter_band[1], method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)
		# TODO: don't do filtering here, although...


		myARMBR = ARMBR(blink_channels)
		myARMBR.fit(raw)
		myARMBR.apply(raw)
		
		if OPTS1.plot:
			import matplotlib.pyplot as plt
			raw.plot()  # TODO: ...purely for graphical purposes, we might want filtering here
			plt.show()
		
		if OPTS1.save and len(OPTS1.save_path)>0:
			raw.save(OPTS1.save_path)
		else:
			print('Data not saved.')
			
	else: 
		print('No blink channels found.')

	
else: # no data
	print('No data directory.')
	
	


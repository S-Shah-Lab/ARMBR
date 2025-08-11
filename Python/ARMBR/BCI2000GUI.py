# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog
import os
import sys
import time
import functools

import scipy.signal
import numpy as np

from ARMBR import run_armbr, save_bci2000_weights

def wrap_text(text, width=70):
	return '\n'.join(text[i:i+width] for i in range(0, len(text), width))


def timeit(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print("[⏱️] {} took {:.2f} seconds.".format(func.__name__, end - start ))
		return result
	return wrapper


class BCI2000GUI(tk.Tk):
	def __init__(self, bci2000root, data_file_path=None, blink_channels=None, exclude_channels=None): 
		
		self.bci2000root = os.path.abspath( os.path.expanduser( bci2000root ) )
		self.default_params_path = os.path.join(self.bci2000root, "parms")
		self.default_param_name = "ARMBR_BlinkRemovalMatrix.prm"
		
		
		# @@@  We assume that the BCI2000Tools package is
		#      version-controlled or released as part of the
		#      BCI2000 distro. Therefore, before anything can
		#      be imported from BCI2000Tools, we need to
		#      configure the Python path:		
		bci_tools_path = os.path.join(self.bci2000root, "tools", "python")
		if bci_tools_path not in sys.path:
			sys.path.insert(0, bci_tools_path)
			
		super().__init__()

		self.title("ARMBR Training GUI.")
		self.geometry("600x250") # mac needs more width than windows
		
		# --------------------LINE 1----------------
		# Label for .dat File selection
		self.data_path_label = tk.Label(self, text="Select `.dat` File:")
		self.data_path_label.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="e")

		# Button to browse for a .dat file
		self.browse_button = tk.Button(self, text="Browse", command=self.select_dat_file)
		self.browse_button.grid(row=0, column=1, pady=5, sticky="w")

		# Label to show the selected .dat file name
		self.selected_file_label = tk.Label(self, text="No file selected", anchor="center", fg='#2288FF', width=60)
		self.selected_file_label.grid(row=1, column=0, columnspan=3, padx=25, pady=5, sticky="w")

		
		# --------------------LINE 2----------------
		self.data_blink_chan = tk.Label(self, text="Select blink channels:")
		self.data_blink_chan.grid(row=2, column=0, sticky="e", padx=(0, 5), pady=5)
		
		# Entry field for data path
		self.blink_channels_var = tk.StringVar()
		self.blink_channels_entry = tk.Entry(self, textvariable=self.blink_channels_var, width=30)
		self.blink_channels_entry.grid(row=2, column=1,sticky="w", pady=5)
		
		self.check_blink_channels_button = tk.Button(self, text="Show channels", command=self.show_available_channels)
		self.check_blink_channels_button.grid(row=2, column=2, padx=0, pady=5, sticky="w")
		
		# --------------------LINE 3----------------
		self.data_exclude_chan = tk.Label(self, text="Select channels to exclude:")
		self.data_exclude_chan.grid(row=3, column=0, sticky="e", padx=(0, 5), pady=5)
		
		# Entry field for data path
		self.exclude_channels_var = tk.StringVar()
		self.exclude_channels_entry = tk.Entry(self, textvariable=self.exclude_channels_var, width=30)
		self.exclude_channels_entry.grid(row=3, column=1,sticky="w", pady=5)
		
		self.check_exclude_channels_button = tk.Button(self, text="Show channels", command=lambda: self.show_available_channels(blinks_or_exclude_chans=0))
		self.check_exclude_channels_button.grid(row=3, column=2, padx=0, pady=5, sticky="w")
		
		
		
		# --------------------LINE 4----------------
		self.run_ARMBR_button = tk.Button(self, text="Run ARMBR", command=self.run_armbr_)
		self.run_ARMBR_button.grid(row=5, column=1, padx=0, pady=10)
		
		# --------------------LINE 5----------------
		# Create a frame to hold both the Text widget and the Scrollbar
		message_frame = tk.Frame(self)
		message_frame.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

		# Create the Text widget
		self.message_display = tk.Text(
			message_frame, height=4, width=60, wrap="word",
			bg=self.cget("bg"), borderwidth=0
		)
		self.message_display.pack(side="left", fill="both", expand=True)

		# Add a vertical Scrollbar
		scrollbar = tk.Scrollbar(message_frame, command=self.message_display.yview)
		scrollbar.pack(side="right", fill="y")

		# Link the Text widget to the scrollbar
		self.message_display.config(yscrollcommand=scrollbar.set)
		self.message_display.config(state="disabled")

		if data_file_path: self.select_dat_file( data_file_path )
		if isinstance( blink_channels, str ): blink_channels = blink_channels.replace( ',', ' ' ).split()
		if blink_channels: self.blink_channels_var.set( ','.join( blink_channels ) )
		
		if isinstance( exclude_channels, str ): exclude_channels = exclude_channels.replace( ',', ' ' ).split()
		if exclude_channels: self.exclude_channels_var.set( ','.join( exclude_channels ) )
		
	
	def update_message(self, text, color="red", state="normal", do_wrap=True):  
		self.message_display.config(state=state)
		self.message_display.delete("1.0", tk.END)
		
		if do_wrap:
			text = wrap_text(text)
		
		self.message_display.insert(tk.END, text)
		
		self.message_display.tag_add("color", "1.0", tk.END)
		self.message_display.tag_config("color", foreground=color)
		
		self.message_display.tag_config("center", justify="center")
		self.message_display.tag_add("center", "1.0", tk.END)
	
		self.message_display.config(state="disabled")


	def select_dat_file(self, file_path=None):
		if not file_path:
			data_dir = os.path.join(self.bci2000root, 'data')
			if not os.path.isdir(data_dir): data_dir = self.bci2000root
			if not os.path.isdir(data_dir): data_dir = os.getcwd()
			file_path = filedialog.askopenfilename(
				filetypes=[("DAT files", "*.dat")],
				initialdir=data_dir,
			)
		if file_path:
			self.data_file_path = os.path.abspath( os.path.expanduser( file_path ) )  # Store full path internally
			self.data_file_name = os.path.basename(file_path)
			self.selected_file_label.config(text=wrap_text('Selected file: '+self.data_file_name))


	def show_available_channels(self, blinks_or_exclude_chans=1):
		# if blinks_or_exclude_chans == 1 --> then we are picking blinks channels
		# if blinks_or_exclude_chans == 0 --> then we are picking channels to remove from analysis
		
		# === LOAD DATA ===
		
		if not hasattr(self, 'data_file_name'):
			raise ValueError("No data available! Please load data first.")
			
		self.update_message(text='Loading: ' + wrap_text(self.data_file_name))

		from BCI2000Tools.FileReader import bcistream # see @@@
		b = bcistream(self.data_file_path)
		eeg, States = b.decode()
		eeg = np.array(eeg).astype('float64')
		FsOrig = b.samplingrate()

		self.eeg = eeg
		self.fs = FsOrig
		self.channel_names = b.params['ChannelNames']
		b.close()
		
		
		# === POPUP WINDOW ===
		top = tk.Toplevel(self)
		top.title("Select blink channels")

		# === SCROLLABLE FRAME ===
		canvas = tk.Canvas(top, width=300, height=600)
		scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
		scrollable_frame = tk.Frame(canvas)

		scrollable_frame.bind(
			"<Configure>",
			lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
		)

		canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
		canvas.configure(yscrollcommand=scrollbar.set)
		
				# Enable mousewheel scrolling
		def _on_mousewheel(event):
			canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

		def _on_linux_scroll(event):
			if event.num == 4:
				canvas.yview_scroll(-1, "units")
			elif event.num == 5:
				canvas.yview_scroll(1, "units")

		# Optional: Only scroll when mouse is over the canvas
		canvas.bind("<Enter>", lambda e: canvas.bind("<MouseWheel>", _on_mousewheel))
		canvas.bind("<Leave>", lambda e: canvas.unbind("<MouseWheel>"))
		
		# Bind Linux scroll buttons (always active)
		canvas.bind_all("<Button-4>", _on_linux_scroll)
		canvas.bind_all("<Button-5>", _on_linux_scroll)



		canvas.grid(row=0, column=0, columnspan=2)
		scrollbar.grid(row=0, column=2, sticky="ns")

		# === CHECKBOXES FOR CHANNELS ===
		self.channel_vars = []
		for chan in self.channel_names:
			var = tk.BooleanVar()
			cb = tk.Checkbutton(scrollable_frame, text=chan, variable=var)
			cb.pack(anchor='w')
			self.channel_vars.append((chan, var))

		# === BUTTON TO ADD SELECTED CHANNELS ===
		def apply_selected_channels():
			selected = [chan for chan, var in self.channel_vars if var.get()]
			if blinks_or_exclude_chans:
				self.blink_channels_var.set(",".join(selected))
			else:
				self.exclude_channels_var.set(",".join(selected))
			top.destroy()

		add_btn = tk.Button(top, text="Add Channels", command=apply_selected_channels)
		add_btn.grid(row=1, column=0, pady=10, columnspan=2)



	def run_armbr_(self):
		
		# === LOAD DATA ===
		if not hasattr(self, 'eeg'):
			self.update_message(text='Loading: ' + self.data_file_name)
			
			from BCI2000Tools.FileReader import bcistream # see @@@
			b = bcistream(self.data_file_path)
			eeg, States = b.decode()
			eeg = np.array(eeg).astype('float64')
			FsOrig = b.samplingrate()
			
			self.eeg = eeg
			self.fs = FsOrig
			self.channel_names = b.params['ChannelNames']
			b.close()
			
			# === FILTER DATA (1-40 Hz bandpass)===
			sos1 = scipy.signal.butter(N=4, Wn=[40], btype='lowpass', fs=self.fs, output='sos')
			sos2 = scipy.signal.butter(N=4, Wn=[1], btype='highpass', fs=self.fs, output='sos')

			if np.size(self.eeg, axis=0) > np.size(self.eeg, axis=1):
				self.eeg = self.eeg.T

			self.eeg = scipy.signal.sosfiltfilt(sos1, self.eeg)
			self.eeg = scipy.signal.sosfiltfilt(sos2, self.eeg).T
		
		
		# === PARSE BLINK CHANNELS ===
		self.blink_chan = self.blink_channels_entry.get().replace(',', ' ').split()
		self.blink_chan_ix = [self.channel_names.index(blk_chn) for blk_chn in self.blink_chan]
		# TODO: deliver appropriate error if an illegal channel was specified
		
		self.exclude_chan = self.exclude_channels_entry.get().replace(',', ' ').split()
		self.exclude_chan_ix = [self.channel_names.index(blk_chn) for blk_chn in self.exclude_chan if blk_chn in self.channel_names]
		
		
		# Create ARMBR object (using filtered EEG)
		self.update_message(text='Running ARMBR. This can take a while...')
		self.message_display.update()  # Force update
		
		_, _, _, _, _, blink_removal_matrix = run_armbr(self.eeg, self.blink_chan_ix, self.exclude_chan_ix, int(self.fs), -1)

		self.update_message(text='ARMBR done. Weights not saved.')
		self.message_display.update()  # Force update

		self.m = blink_removal_matrix

		weights_file_name = os.path.join(self.default_params_path, self.default_param_name)
		save_bci2000_weights( self.m, channel_names=self.channel_names, filename=weights_file_name,
		                      training_file_name=self.data_file_path, blink_channels=self.blink_chan, exclude_channels=self.exclude_chan)
		self.update_message(text='Weights saved at: '+ weights_file_name, do_wrap=False)
		
		return self
		
	def display_info_window(self):
		"""Displays a new window with information and a Continue button."""
		info_window = tk.Toplevel()
		info_window.title("Would you like to continue?")
		info_window.geometry("600x300")

		text_to_display = (
				"Loading file: {datapath}\n"
				"Blink channels: {blinkchan}\n"
				"Saving parameter file: {parampath}\n\n"
				"Would you like to continue?"
			).format(datapath=self.data_file_path, blinkchan=self.blink_chan, parampath=os.path.join( self.default_params_path, self.default_param_name))

							
		# Add information text
		from tkinter import ttk
		info_label = ttk.Label(
			info_window,
			text=text_to_display,
			wraplength=550,
			justify="center",
			font=("Arial", 12)
		)
		info_label.pack(pady=20)

		# Add a Continue button
		continue_button = ttk.Button(info_window, text="Continue", command=lambda: self._close_info_and_run(info_window))
		continue_button.pack(pady=20)

	def _close_info_and_run(self, window):
		"""Closes the information window and continues with ARMBR processing."""
		window.destroy()

	def set_params_name(self):

		self.params_name = self.params_var.get()
		# Check if the filename has an extension
		name, ext = os.path.splitext(self.params_name)
		# If the file has an extension, remove it and append .prm
		if ext:
			self.params_name = name + '.prm'
		else:
			# If no extension, just add .prm
			self.params_name = self.params_name + '.prm'
			
		# Update the text later
		self.update_message(text='Parameter name: ' + str(self.params_name))


	def load_blink_channels(self):
		self.blink_chan = [chan for chan in self.blink_channels_entry.get().replace(" ","").replace(" ","").split(",")]
		
		self.blink_chan_ix = [self.channel_names.index(blk_chn) for blk_chn in self.blink_chan]
		
		self.update_message(text='Blink channels: ' + str(self.blink_chan))
		
def run_gui( *pargs, **kwargs ):	
	app = BCI2000GUI( *pargs, **kwargs )
	app.mainloop()


def install_demo( bci2000root=None, show=False ):

	def create_file( content, directory, filename ):
		filepath = os.path.join(directory, filename)
		verb = 'Overwrote' if os.path.isfile(filepath) else 'Created'
		with open(filepath, "w") as file:
			file.write(content)
		print("✅ {verb} {filepath}".format(verb=verb, filepath=filepath))
	
	if bci2000root: # Case 1: Directory provided
		dirs = [ bci2000root ]
	else: # Case 2: Ask user for directories
		#dirs_input = input("Enter one or more BCI2000 distribution root directory (comma-separated): ")
		#dirs = [d.strip() for d in dirs_input.split(",") if d.strip()]
		import tkinter.filedialog; root = tkinter.Tk(); root.withdraw();
		try:    dirs = [ tkinter.filedialog.askdirectory(message="Select the root directory of your BCI2000 distribution", initialdir=os.getcwd()) ]
		except: dirs = [ tkinter.filedialog.askdirectory(  title="Select the root directory of your BCI2000 distribution", initialdir=os.getcwd()) ]
		if dirs == [ '' ]: raise RuntimeError('user cancelled')
		
	exit_status = 0
	for directory in dirs:
		directory = os.path.abspath(os.path.expanduser(directory))
		if os.path.exists(directory):
			create_file(ARMBR_FIT_CONTENT,   os.path.join(directory, 'batch'), 'ARMBR_1_Fit.bat')
			create_file(ARMBR_APPLY_CONTENT, os.path.join(directory, 'batch'), 'ARMBR_2_Apply.bat')
		else:
			print("⚠️  Directory '{directory}' does not exist.".format(directory=directory))
			exit_status = 1
	if show and len(dirs) == 1 and exit_status == 0:
		if sys.platform.lower().startswith('win'): os.system('start "" "{}"'.format(os.path.join(dirs[0], 'batch')))
	return exit_status

##########################################################################
################# .bat file contents #####################################
##########################################################################


ARMBR_FIT_CONTENT = r"""

:<<@GOTO:EOF

:: Windows cmd.exe #################################################

@echo off
python -V || (
	echo.
	echo You need to install Python
	echo.
	pause
	exit /b 1
)
python -m ARMBR --version || (
	echo.
	echo The ARMBR package was not found in this Python distribution.
	echo To fix this, run:  python -m pip install ARMBR
	echo.
	pause
	exit /b 1
)
set OLDDIR="%CD%"
cd "%~dp0.."
set "BCI2000ROOT=%CD%"
cd "%OLDDIR%"
python -m ARMBR "--BCI2000=%BCI2000ROOT%" %* || pause

@GOTO:EOF

## Bash ##########################################################

OLDDIR=$(pwd)
cd $(dirname "${BASH_SOURCE[0]}")/..
BCI2000ROOT=$(pwd)
cd "$OLDDIR"

which python3 >/dev/null 2>&1 && SNAKE=python3 || SNAKE=python
$SNAKE -V && GOT_PYTHON=1 || GOT_PYTHON=
test -z "$GOT_PYTHON" && echo -e '\nCould not find a python executable\n' && exit 1
$SNAKE -m ARMBR --version && GOT_ARMBR=1 || GOT_ARMBR=
test -z "$GOT_ARMBR" && echo -e '\nThe ARMBR package was not found in this Python distribution.\nTo fix this, run:  python -m pip install ARMBR\n' && exit 1
$SNAKE -m ARMBR --BCI2000=$BCI2000ROOT "$@" 
""".lstrip()

ARMBR_APPLY_CONTENT = r"""

#! ../prog/BCI2000Shell
@cls & ..\prog\BCI2000Shell %0 %* #! && exit /b 0 || exit /b 1


change directory $BCI2000LAUNCHDIR
show window
set title ${extract file base $0}
reset system
startup system localhost

set environment DATFILE  $1 
if [ $DATFILE == "" ]
	set environment DATFILE ${choose file of type .dat from ../data/samplefiles with prompt "Select a data file to play back:"}
end
if [ $DATFILE == "" ]
	set environment DATFILE ${real path "../data/samplefiles/eeg1_2.dat"}
	warn playing default file $DATFILE
end
set title ${extract file base $DATFILE}.dat

start executable FilePlayback             --local --FileFormat=null --PlaybackFileName=$DATFILE
start executable SpectralSignalProcessing --local
start executable DummyApplication         --local

wait for connected

set environment WEIGHTS ${real path ../parms/ARMBR_BlinkRemovalMatrix.prm}
if [ ${exists file $WEIGHTS} ]
	load parameterfile $WEIGHTS
else
	warn ARMBR will not be applied - found no $WEIGHTS
end

set parameter VisualizeTiming                      0
set parameter VisualizeSource                      0
set parameter VisualizeTransmissionFilter          1
set parameter VisualizeSpatialFilter               1
set parameter VisualizeSpectralEstimator           0

set parameter Filtering matrix Classifier=  1 4    1 1 1 0
set parameter Filtering matrix Expressions= 1 1    0
set parameter Filtering list   Adaptation=    1    0

set parameter WindowLength                         1s
set parameter FirstBinCenter                       1Hz
set parameter LastBinCenter                       80Hz
set parameter BinWidth                             1Hz
set parameter SpectralEstimator                    2    # FFT

setconfig
set state Running 1
""".lstrip()


	
if __name__ == "__main__":
	run_gui()

import tkinter as tk
from tkinter import filedialog
import os
import sys
import time

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ARMBR import run_armbr

def wrap_text(text, width=70):
	return '\n'.join(text[i:i+width] for i in range(0, len(text), width))


def timeit(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print(f"[⏱️] {func.__name__} took {end - start:.2f} seconds.")
		return result
	return wrapper


class BCI2000GUI(tk.Tk):
	def __init__(self, bci2000root):
		
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
		self.geometry("500x220")
		
		# --------------------LINE 1----------------
		# Label for .dat Directory
		
		self.data_path_label = tk.Label(self, text="Select `.dat` Directory:")
		self.data_path_label.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="e")

		# Entry field for data path
		self.data_path_var = tk.StringVar()
		self.data_path_entry = tk.Entry(self, textvariable=self.data_path_var, width=30)
		self.data_path_entry.grid(row=0, column=1, padx=0, pady=5, sticky="w")
		
		# Button to open file explorer to select a directory
		self.browse_button = tk.Button(self, text="Browse", command=self.select_data_path)
		self.browse_button.grid(row=0, column=2, padx=0, pady=5)
		

		# --------------------LINE 2----------------
		# Label for .dat Files
		self.data_file_label = tk.Label(self, text="Select `.dat` File:")
		self.data_file_label.grid(row=1, column=0, sticky="e", padx=(0, 5), pady=5)

		# Dropdown for selecting .dat file
		self.data_file_var = tk.StringVar()
		self.data_file_menu = tk.OptionMenu(self, self.data_file_var, "")
		self.data_file_menu.config(width=30)  # Adjust width as needed
		self.data_file_menu.grid(row=1, column=1, columnspan=1, sticky="w", pady=5)
		

		# --------------------LINE 3----------------
		self.data_blink_chan = tk.Label(self, text="Select Blink Channels:")
		self.data_blink_chan.grid(row=2, column=0, sticky="e", padx=(0, 5), pady=5)
		
		# Entry field for data path
		self.blink_channels_var = tk.StringVar()
		self.blink_channels_entry = tk.Entry(self, textvariable=self.blink_channels_var, width=30)
		self.blink_channels_entry.grid(row=2, column=1,sticky="w", pady=5)
		
		self.check_channels_button = tk.Button(self, text="Show Channels", command=self.show_available_channels)
		self.check_channels_button.grid(row=2, column=2, padx=0, pady=5, sticky="w")

		# --------------------LINE 6----------------
		self.run_ARMBR_button = tk.Button(self, text="Run ARMBR", command=self.run_armbr_)
		self.run_ARMBR_button.grid(row=5, column=1, padx=0, pady=10)
		
		# --------------------LINE 7----------------
		self.message_display = tk.Label(self, text=" ", width=40)
		self.message_display.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="we")
		


		
	def show_available_channels(self):
		# === LOAD DATA ===
		self.DatFileDir = self.data_path_entry.get() + '/' + self.data_file_var.get()
		
		self.message_display.config(text='Loading: ' + wrap_text(self.data_file_var.get()), fg="red")

		from BCI2000Tools.FileReader import bcistream # see @@@
		b = bcistream(self.DatFileDir)
		eeg, States = b.decode()
		eeg = np.array(eeg)
		FsOrig = b.samplingrate()

		self.eeg = eeg
		self.fs = FsOrig
		self.channel_names = b.params['ChannelNames']
		
		
		# === FILTER DATA (1-40 Hz bandpass)===
		sos1 = signal.butter(N=4, Wn=[40], btype='lowpass', fs=self.fs, output='sos')
		sos2 = signal.butter(N=4, Wn=[1], btype='highpass', fs=self.fs, output='sos')

		if np.size(self.eeg, axis=0) > np.size(self.eeg, axis=1):
			self.eeg = self.eeg.T

		self.eeg = signal.sosfiltfilt(sos1, self.eeg)
		self.eeg = signal.sosfiltfilt(sos2, self.eeg).T
		
		
		# === POPUP WINDOW ===
		top = tk.Toplevel(self)
		top.title("Select Blink Channels")

		# === SCROLLABLE FRAME ===
		canvas = tk.Canvas(top, width=300, height=200)
		scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
		scrollable_frame = tk.Frame(canvas)

		scrollable_frame.bind(
			"<Configure>",
			lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
		)

		canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
		canvas.configure(yscrollcommand=scrollbar.set)

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
			self.blink_channels_var.set(",".join(selected))
			top.destroy()

		add_btn = tk.Button(top, text="Add Channels", command=apply_selected_channels)
		add_btn.grid(row=1, column=0, pady=10, columnspan=2)



	def run_armbr_(self):
		
		# === LOAD DATA ===
		if not hasattr(self, 'eeg'):
			self.DatFileDir = self.data_path_entry.get() + '/' + self.data_file_var.get()
			self.message_display.config(text='Loading: ' + wrap_text(self.data_file_var.get()), fg="red")
			
			from BCI2000Tools.FileReader import bcistream # see @@@
			b = bcistream(self.DatFileDir)
			eeg, States = b.decode()
			eeg = np.array(eeg)
			FsOrig = b.samplingrate()
			
			self.eeg = eeg
			self.fs = FsOrig
			self.channel_names = b.params['ChannelNames']
			
			# === FILTER DATA (1-40 Hz bandpass)===
			sos1 = signal.butter(N=4, Wn=[40], btype='lowpass', fs=self.fs, output='sos')
			sos2 = signal.butter(N=4, Wn=[1], btype='highpass', fs=self.fs, output='sos')

			if np.size(self.eeg, axis=0) > np.size(self.eeg, axis=1):
				self.eeg = self.eeg.T

			self.eeg = signal.sosfiltfilt(sos1, self.eeg)
			self.eeg = signal.sosfiltfilt(sos2, self.eeg).T
		
		

		# === PARSE BLINK CHANNELS ===
		self.blink_chan = [chan.strip() for chan in self.blink_channels_entry.get().split(",")]
		self.blink_chan_ix = [self.channel_names.index(blk_chn) for blk_chn in self.blink_chan]

		# Create ARMBR object (using filtered EEG)
		self.message_display.config(text=wrap_text('Running ARMBR. This can take a while...'), fg="red")
		self.message_display.update()  # Force update
		
		_, _, _, _, _, blink_removal_matrix = run_armbr(self.eeg, self.blink_chan_ix, int(self.fs), -1)

		self.message_display.config(text='ARMBR done. Weights not saved.', fg="red")
		self.message_display.update()  # Force update

		from BCI2000Tools.Electrodes import ChannelSet # see @@@
		c = ChannelSet(" ".join(self.channel_names))
		m = blink_removal_matrix

		self.m = m
		blink_free_eeg = m @ c

		
		c.BCI2000SpatialFilterParameters(
			m, 
			outputFileName=self.default_params_path + '/' + self.default_param_name, 
			fullFormat=True
		)
		
		
		# Construct the TransmitChanList line using self.blink_chan
		transmit_line = "Source list TransmitChList= %5d" % len(self.channel_names) + " " + " ".join(self.channel_names)

		# Append it to the .prm file
		with open(os.path.join(self.default_params_path, self.default_param_name), "a") as f:
			f.write(transmit_line + "\n")

		weights_saved_at = 'Weights save at:\n '+ self.default_params_path + '/' + self.default_param_name
		
		wrapped_text = wrap_text(weights_saved_at)
		self.message_display.config(text=wrapped_text, fg="red")
		
		return self
		
	def display_info_window(self):
		"""Displays a new window with information and a Continue button."""
		info_window = tk.Toplevel()
		info_window.title("Would you like to continue?")
		info_window.geometry("600x300")

		text_to_display = (
				f"Loading file: {self.DatFileDir}\n"
				f"Blink channels: {self.blink_chan}\n"
				f"Saving parameter file: {self.params_path_var.get()}/{self.params_name}\n\n"
				"Would you like to continue?"
			)

							
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
		self.message_display.config(text='Parameter name: ' + str(self.params_name), fg="red")


	def load_blink_channels(self):
		self.blink_chan = [chan for chan in self.blink_channels_entry.get().replace(" ","").replace(" ","").split(",")]
		
		self.blink_chan_ix = [self.channel_names.index(blk_chn) for blk_chn in self.blink_chan]
		
		self.message_display.config(text='Blink channels: ' + str(self.blink_chan), fg="red")
		
		
	def select_data_path(self):
		# Open folder dialog to select the directory
		initialdir = os.path.join( self.bci2000root, 'data' )
		if not os.path.isdir( initialdir ): initialdir = self.bci2000root
		folder_path = filedialog.askdirectory(title="Select Folder with .dat Files", initialdir=initialdir)
		if folder_path:
			# Update the entry field with the selected folder path
			self.data_path_var.set(folder_path)
			# Update the dropdown with .dat files in the selected folder
			self.update_dat_dropdown(folder_path)
		
			
	def update_dat_dropdown(self, folder_path):
		# List all .dat files in the selected directory
		dat_files = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
		# If there are .dat files, update the dropdown
		if dat_files:
			# Set the default value for the dropdown
			self.data_file_var.set(dat_files[0])
			# Update the dropdown options
			menu = self.data_file_menu["menu"]
			menu.delete(0, "end")  # Remove all existing entries
			for file in dat_files:
				menu.add_command(label=file, command=tk._setit(self.data_file_var, file))
			
			# Ensure that the dropdown menu is clickable again after updating
			self.data_file_menu.grid()  # Refresh the grid configuration

		else:
			# If no .dat files are found, clear the dropdown
			self.data_file_var.set("")
			self.data_file_menu["menu"].delete(0, "end")

def RunGUI( bci2000root ):	
	app = BCI2000GUI( bci2000root=bci2000root )
	app.mainloop()

if __name__ == "__main__":
	RunGUI()

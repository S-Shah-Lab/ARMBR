import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from ARMBR import ARMBR 


def _load_mne_raw(with_annotation=True):
    # Load MNE sample EEG data
    data_path = sample.data_path()
    raw_path = str(data_path) + "/MEG/sample/sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    raw.pick_types(meg=False, eeg=True)
    raw.filter(l_freq=0.5, h_freq=40)
    
    if with_annotation:
        raw.set_annotations(mne.Annotations(onset=[0.0], duration=[20.0], description=['armbr_fit']))
    return raw

def test_armbr_basic_fit_and_apply():
    raw = _load_mne_raw()
    armbr = ARMBR(ch_name=["EEG 001"])
    armbr.fit(raw, verbose=True, start=0, stop=18) # ARMBR is trained on data from 0 to 18 seconds
    original = raw.copy().get_data()
    armbr.apply(raw)
    updated = raw.get_data()
    assert hasattr(armbr, 'best_alpha')
    assert hasattr(armbr, 'blink_spatial_pattern')
    assert updated.shape == original.shape
    assert not np.allclose(original, updated)

def test_armbr_fit_and_apply_with_annotation():
    raw = _load_mne_raw(with_annotation=True) # Add armbr_fit annotation to raw 
    armbr = ARMBR(ch_name=["EEG 001"])
    armbr.fit(raw, verbose=True)
    original = raw.copy().get_data()
    armbr.apply(raw)
    updated = raw.get_data()
    assert hasattr(armbr, 'best_alpha')
    assert hasattr(armbr, 'blink_spatial_pattern')
    assert updated.shape == original.shape
    assert not np.allclose(original, updated)

def test_armbr_multiple_blink_channels():
    raw = _load_mne_raw()
    armbr = ARMBR(ch_name=["EEG 001", "EEG 002"])
    armbr.fit(raw)
    armbr.apply(raw)
    assert hasattr(armbr, 'blink_comp')
    
def test_armbr_missing_annotation_uses_full_data():
    raw = _load_mne_raw(with_annotation=False)
    armbr = ARMBR(ch_name=[0])  # using index
    armbr.fit(raw)
    assert armbr._eeg_data.shape[0] == raw.n_times
    assert armbr.blink_comp.shape[0] == raw.n_times

def test_armbr_bad_segment_exclusion():
    raw = _load_mne_raw(with_annotation=False)
    bad_annot = mne.Annotations(onset=[raw.first_samp/raw.info['sfreq'] + 10], duration=[200.0], description=["BAD_segment"], orig_time=raw.annotations.orig_time)
    raw.set_annotations(raw.annotations + bad_annot)
    armbr = ARMBR(ch_name=["EEG 001"])
    armbr.fit(raw, verbose=True)
    # Confirm exclusion from annotation
    used_samples = armbr._eeg_data.shape[0]
    assert used_samples < raw.n_times

def test_armbr_raises_on_mixed_types():
    raw = _load_mne_raw()
    armbr = ARMBR(ch_name=["EEG 001", 2])
    armbr.fit(raw)
    
def test_armbr_plot_eeg_before_after_with_blinkspatialpattern():
    # Load MNE sample EEG data
    raw         = _load_mne_raw()
    raw_before  = raw.copy()
    raw_after   = raw.copy()

    armbr = ARMBR(ch_name=["EEG 001"])
    armbr.fit(raw_before, start=0, stop=20)
    armbr.apply(raw_after)

    # Plot raw before and after ARMBR
    raw_before.plot(title="Before ARMBR", start=9, duration=4, n_channels=10, scalings='auto')
    raw_after.plot(title="After ARMBR", start=9, duration=4, n_channels=10, scalings='auto')
    
    # Plot blink spatial pattern 
    armbr.plot_blink_patterns()
    
    

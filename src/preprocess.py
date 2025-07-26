import numpy as np
import mne

def preprocess_eeg_trials(eeg_trials, sfreq, l_freq=1.0, h_freq=40.0, downsample=None):
    """
    Preprocess a list of EEG trial arrays.

    Args:
        eeg_trials (list of np.ndarray): EEG trials, each shape=(samples, channels)
        sfreq (float): Original sampling frequency (Hz)
        l_freq (float): Low frequency cut for bandpass filter
        h_freq (float): High frequency cut for bandpass filter
        downsample (int or None): New sampling rate after downsampling. No downsampling if None.

    Returns:
        processed_trials (list of np.ndarray): Preprocessed EEG trials with shape=(channels, samples)
        new_sfreq (float): Sampling frequency after optional downsampling
    """
    processed_trials = []

    for trial_data in eeg_trials:
        # trial_data shape: (samples, channels) -> transpose to (channels, samples) for MNE use
        trial_data = trial_data.T

        # Create MNE RawArray from numpy data for filtering
        info = mne.create_info(ch_names=trial_data.shape[0], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(trial_data, info)

        # Apply bandpass filter
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)

        # Downsample if needed
        if downsample is not None and downsample < sfreq:
            raw.resample(downsample, npad='auto', verbose=False)
            new_sfreq = downsample
        else:
            new_sfreq = sfreq

        # Get data back as numpy array (channels x samples)
        processed_trial = raw.get_data()
        processed_trials.append(processed_trial)

    return processed_trials, new_sfreq

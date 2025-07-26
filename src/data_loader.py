import scipy.io as sio
import numpy as np


def load_s1_mat(file_path):
    """
    Load S1.mat EEG data from KU Leuven dataset.

    Returns:
        eeg_trials: list of np.ndarray, each (samples, channels)
        attended_labels: list of str, 'L' or 'R'
        fs: int, sampling frequency in Hz
    """
    # Load .mat file with struct_as_record disabled for easier attribute access
    mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

    # Extract trials (1-D array of trial objects)
    trials = mat_data['trials'].flatten()

    eeg_trials = []
    attended_labels = []
    fs = None

    for trial in trials:
        eeg_data = trial.RawData.EegData  # (samples, channels)
        eeg_trials.append(eeg_data)

        # Extract sampling frequency once (assumed same for all trials)
        if fs is None:
            fs = int(trial.FileHeader.SampleRate)

        # Attended ear label ('L' or 'R')
        attended_labels.append(trial.attended_ear)

    return eeg_trials, attended_labels, fs

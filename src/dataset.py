import torch
from torch.utils.data import Dataset


def segment_trials(eeg_trials, segment_length, overlap=0):
    """
    Segment each trial into fixed-length windows with optional overlap.

    Args:
        eeg_trials (list of np.ndarray): Each trial shape (channels, samples)
        segment_length (int): Window size in samples (e.g., 256 for 2 seconds at 128 Hz)
        overlap (int): Overlap size in samples (default=0 for no overlap)

    Returns:
        segmented_trials (list of np.ndarray): List of fixed-length segments (channels, segment_length)
    """
    segmented = []
    for trial in eeg_trials:
        n_channels, n_samples = trial.shape
        step = segment_length - overlap
        for start in range(0, n_samples - segment_length + 1, step):
            segment = trial[:, start : start + segment_length]
            segmented.append(segment)
    return segmented


class EEGAttentionDataset(Dataset):
    """
    PyTorch Dataset for EEG trials and attention labels.

    Expects:
    - eeg_trials: list of np.ndarray with shape (channels, samples)
    - labels: list of str ('L' or 'R')

    Outputs:
    - tensor EEG data (channels, samples) as float32
    - label as int (0 for 'L', 1 for 'R')
    """

    def __init__(self, eeg_trials, labels):
        self.eeg_trials = eeg_trials
        self.labels = labels
        self.label_map = {'L': 0, 'R': 1}

        # Check that number of samples and labels match
        assert len(self.eeg_trials) == len(self.labels), \
            f"Mismatch between trials ({len(self.eeg_trials)}) and labels ({len(self.labels)})"

    def __len__(self):
        return len(self.eeg_trials)

    def __getitem__(self, idx):
        eeg = self.eeg_trials[idx]
        label_str = self.labels[idx]
        label = self.label_map[label_str]

        # Convert to torch tensor, float32
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return eeg_tensor, label_tensor

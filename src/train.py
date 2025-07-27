import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.data_loader import load_s1_mat
from src.preprocess import preprocess_eeg_trials
from src.dataset import segment_trials, EEGAttentionDataset
from src.model import CNNLSTMAttentionDecoder


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    file_path,
    batch_size=16,
    num_epochs=20,
    learning_rate=1e-3,
    segment_length=256,
    val_ratio=0.2,
    device=None,
    save_path="best_model.pth"
):
    """
    Load data, preprocess, and train the CNN-LSTM model.

    Args:
        file_path (str): Path to the S1.mat file.
        batch_size (int): Batch size for DataLoader.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        segment_length (int): Length of EEG segments (samples).
        val_ratio (float): Fraction of data for validation.
        device (torch.device or None): Device to use. If None, auto-select.
        save_path (str): Path to save the best model.

    Returns:
        model: Trained PyTorch model (best validation accuracy).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    eeg_trials, attended_labels, fs = load_s1_mat(file_path)
    print(f"Loaded {len(eeg_trials)} trials at {fs} Hz")

    print("Preprocessing EEG...")
    processed_trials, new_fs = preprocess_eeg_trials(eeg_trials, sfreq=fs, l_freq=1, h_freq=40, downsample=128)
    print(f"Resampled to {new_fs} Hz")

    print("Segmenting trials...")
    segmented_trials = segment_trials(processed_trials, segment_length, overlap=0)

    # Repeat labels per segment
    segmented_labels = []
    for label, trial in zip(attended_labels, processed_trials):
        n_segments = (trial.shape[1] - segment_length) // segment_length + 1
        segmented_labels.extend([label] * n_segments)

    dataset = EEGAttentionDataset(segmented_trials, segmented_labels)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model, criterion, optimizer
    model = CNNLSTMAttentionDecoder(n_channels=64, segment_length=segment_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Acc: {best_val_acc:.4f}")

    print(f"Training completed. Best Val Acc: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EEG Attention Decoder CNN-LSTM")
    parser.add_argument("--file_path", type=str, default="../data/raw/S1.mat", help="Path to S1.mat")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_model(file_path=args.file_path, num_epochs=args.epochs, batch_size=args.batch_size)

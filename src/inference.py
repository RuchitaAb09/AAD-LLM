import torch
from src.model import CNNLSTMAttentionDecoder

def load_trained_model(model_path, n_channels=64, segment_length=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTMAttentionDecoder(n_channels=n_channels, segment_length=segment_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_segments(model, eeg_segments, device=None):
    """
    Args:
        model: Trained EEG model
        eeg_segments: list of np.ndarray (channels, segment_length)
        device: torch.device
    Returns:
        predictions: list of int labels (0='L', 1='R')
        probabilities: list of [P(left), P(right)] scores for each segment
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for seg in eeg_segments:
            tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)  # (1, C, S)
            logits = model(tensor)  # (1, 2)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = prob.argmax()
            preds.append(pred)
            probs.append(prob)
    return preds, probs

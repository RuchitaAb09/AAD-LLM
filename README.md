# Simplified AAD-LLM: Neural Attention-Driven Auditory Scene Understanding

This project implements a minimal, open-source version of neural attention-driven auditory scene understanding using the KU Leuven Auditory Attention Dataset. The goal is to decode auditory attention (which speaker a listener focuses on) from EEG, and demonstrate selective audio processing.

## Features

- Loads EEG data from BioSemi .bdf files.
- Preprocesses EEG: filtering, epoching, artifact removal.
- Trains a simple neural network (CNN/LSTM) for auditory attention decoding.
- Extracts attended speaker's audio based on decoder predictions.
- Modular Python repo suitable for research and portfolio.

## Project Structure

See [below](#directory-structure) for full directory layout.

## Data

- **Dataset:** [KU Leuven Auditory Attention Dataset](https://zenodo.org/records/4004271)
- Place raw `.bdf` EEG and audio files in `data/raw/`.

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Run notebooks for each stage, or use scripts in `src/`.
3. See `notebooks/` for demonstration.

## Directory Structure
aad-llm-simplified/
├── data/
│ ├── raw/
│ ├── processed/
├── notebooks/
├── src/
├── tests/
├── requirements.txt
├── README.md
└── LICENSE


## References

- [MNE-Python: EEG/MEG analysis toolbox](https://mne.tools/)
- [KU Leuven AAD Dataset info](https://homes.esat.kuleuven.be/~abertran/datasets.html)

---

MIT License. Created July 2025.

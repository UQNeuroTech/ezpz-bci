# Data Directory

This directory contains both EEG data files and configuration files used by the ezpz-bci application:

## EEG Data Files
- `eeg_samples.json`: Contains the raw EEG samples collected from the device
- `eeg_markers.json`: Contains markers indicating the different actions performed during data collection

## Configuration Files
- `categories.json`: Stores user-defined categories for BCI training, along with configuration parameters like:
  - Training categories (e.g., left_hand, right_hand)
  - Cycle count and duration for data collection
  - Epoch count and learning rate for model training
- `config.json`: Stores keyboard/hotkey mappings for controlling applications

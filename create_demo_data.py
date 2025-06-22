#!/usr/bin/env python3
"""
Demo Data Generator for ezpz-BCI
Creates synthetic EEG data to test the complete pipeline without real hardware
"""

import numpy as np
import json
import os
from pathlib import Path
import mne
from brainflow.board_shim import BoardShim, BoardIds

def generate_synthetic_eeg_data(n_trials=30, trial_duration=3, sampling_rate=250):
    """
    Generate synthetic EEG data that mimics real BCI data
    
    Parameters:
    - n_trials: Number of trials per class (Rest, Left Fist, Right Fist)
    - trial_duration: Duration of each trial in seconds
    - sampling_rate: EEG sampling rate in Hz
    """
    
    print("🧠 Generating synthetic EEG data...")
    
    # EEG parameters
    n_channels = 8  # OpenBCI Cyton channels
    n_samples_per_trial = trial_duration * sampling_rate
    
    # Create class labels (0=Nothing, 1=Rest, 2=Left Fist, 3=Right Fist)
    classes = [1, 2, 3]  # Rest, Left Fist, Right Fist
    class_names = ["Rest", "Left Fist", "Right Fist"]
    
    eeg_samples = []
    eeg_markers = []
    
    for class_idx, class_label in enumerate(classes):
        print(f"  Generating {n_trials} trials for {class_names[class_idx]}...")
        
        for trial in range(n_trials):
            # Generate synthetic EEG data
            # Base EEG: mix of alpha (8-12 Hz) and beta (13-30 Hz) rhythms
            time = np.linspace(0, trial_duration, n_samples_per_trial)
            
            trial_data = []
            for channel in range(n_channels):
                # Create realistic EEG signal
                alpha_freq = 10 + np.random.normal(0, 1)  # 8-12 Hz alpha
                beta_freq = 20 + np.random.normal(0, 3)   # 13-30 Hz beta
                
                # Base signal
                signal = (
                    0.5 * np.sin(2 * np.pi * alpha_freq * time) +
                    0.3 * np.sin(2 * np.pi * beta_freq * time) +
                    0.1 * np.random.normal(0, 1, len(time))  # noise
                )
                
                # Add class-specific patterns
                if class_label == 2:  # Left Fist - stronger activity in right hemisphere
                    if channel in [1, 3, 5, 7]:  # Right hemisphere channels
                        signal += 0.2 * np.sin(2 * np.pi * 15 * time)  # Motor rhythm
                elif class_label == 3:  # Right Fist - stronger activity in left hemisphere
                    if channel in [0, 2, 4, 6]:  # Left hemisphere channels
                        signal += 0.2 * np.sin(2 * np.pi * 15 * time)  # Motor rhythm
                
                # Convert to microvolts (BrainFlow format)
                signal = signal * 50  # Scale to realistic EEG amplitude
                trial_data.append(signal.tolist())
            
            eeg_samples.append(trial_data)
            eeg_markers.append(class_label)
    
    return eeg_samples, eeg_markers

def save_demo_data(eeg_samples, eeg_markers, data_dir="demo_data"):
    """Save synthetic data in the format expected by the pipeline"""
    
    # Create data directory
    Path(data_dir).mkdir(exist_ok=True)
    
    # Save raw JSON data (format from collect_data_openbci.py)
    samples_path = os.path.join(data_dir, "eeg_samples.json")
    markers_path = os.path.join(data_dir, "eeg_markers.json")
    
    print(f"💾 Saving raw data to {data_dir}/")
    with open(samples_path, 'w') as f:
        json.dump(eeg_samples, f)
    
    with open(markers_path, 'w') as f:
        json.dump(eeg_markers, f)
    
    print(f"  ✅ Saved {len(eeg_samples)} trials")
    print(f"  ✅ Classes: {set(eeg_markers)}")
    
    return samples_path, markers_path

def convert_to_mne_format(eeg_samples, eeg_markers, save_path="demo_data/demo_eeg_data.fif"):
    """Convert synthetic data to MNE format for training"""
    
    print("🔄 Converting to MNE format...")
    
    # EEG parameters
    sampling_rate = 250
    n_channels = 8
    
    # Flatten the data structure
    all_data = []
    all_markers = []
    
    for trial_idx, (trial_data, marker) in enumerate(zip(eeg_samples, eeg_markers)):
        # trial_data is [n_channels, n_samples]
        trial_array = np.array(trial_data)
        
        # Add to continuous data
        if len(all_data) == 0:
            all_data = trial_array
        else:
            all_data = np.concatenate([all_data, trial_array], axis=1)
        
        # Create marker events (mark the start of each trial)
        trial_length = trial_array.shape[1]
        trial_markers = [0] * trial_length
        trial_markers[0] = marker  # Mark beginning of trial
        all_markers.extend(trial_markers)
    
    # Convert to microvolts to volts for MNE
    all_data = all_data / 1000000
    
    # Create MNE info
    ch_names = [f'EEG{i+1}' for i in range(n_channels)] + ['STI']
    ch_types = ['eeg'] * n_channels + ['stim']
    
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
    
    # Add stimulus channel
    stim_data = np.array(all_markers).reshape(1, -1)
    full_data = np.vstack([all_data, stim_data])
    
    # Create Raw object
    raw = mne.io.RawArray(full_data, info)
    raw.set_montage("standard_1020")
    
    # Filter data
    raw = raw.filter(l_freq=0.5, h_freq=40)
    
    # Find events
    events = mne.find_events(raw, stim_channel='STI')
    
    # Create epochs
    event_dict = {
        "Rest": 1,
        "Left Fist": 2,
        "Right Fist": 3
    }
    
    tmin, tmax = 0, 2.5  # Epoch from 0 to 2.5 seconds after event
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, preload=True)
    
    # Save epochs
    Path(save_path).parent.mkdir(exist_ok=True)
    epochs.save(save_path, overwrite=True)
    
    print(f"  ✅ Saved MNE epochs: {save_path}")
    print(f"  ✅ Epochs shape: {epochs.get_data().shape}")
    print(f"  ✅ Event counts: {dict(zip(epochs.event_id.keys(), [len(epochs[k]) for k in epochs.event_id.keys()]))}")
    
    return save_path

def main():
    """Generate complete demo dataset"""
    
    print("🚀 Creating ezpz-BCI Demo Dataset")
    print("=" * 50)
    
    # Generate synthetic EEG data
    eeg_samples, eeg_markers = generate_synthetic_eeg_data(
        n_trials=20,  # 20 trials per class
        trial_duration=3,
        sampling_rate=250
    )
    
    # Save raw JSON data
    samples_path, markers_path = save_demo_data(eeg_samples, eeg_markers)
    
    # Convert to MNE training format
    mne_path = convert_to_mne_format(eeg_samples, eeg_markers)
    
    print("\n🎉 Demo dataset created successfully!")
    print("\nFiles created:")
    print(f"  📁 Raw JSON data: demo_data/eeg_samples.json, eeg_markers.json")
    print(f"  📁 Training data: {mne_path}")
    
    print("\n🧪 Ready to test the complete pipeline:")
    print("  1. Use the GUI Data Collection tab (simulation mode)")
    print("  2. Use the GUI Training tab with the .fif file")
    print("  3. Use the GUI Classification tab with the trained model")
    
    return mne_path

if __name__ == "__main__":
    main() 
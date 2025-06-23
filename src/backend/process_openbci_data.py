import json
import numpy as np
import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import matplotlib.pyplot as plt

def load_openbci_data(jsons_path, verbose=False, show_ui = True):
    
    # Convert relative paths to absolute for clearer error messages
    import os
    abs_jsons_path = os.path.abspath(jsons_path)
    samples_path = os.path.join(abs_jsons_path, "eeg_samples.json")
    markers_path = os.path.join(abs_jsons_path, "eeg_markers.json")

    # Check if directory exists
    if not os.path.isdir(abs_jsons_path):
        raise FileNotFoundError(f"Data directory not found: {abs_jsons_path}. Please ensure the data directory exists.")

    # Check if files exist before trying to open them
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"EEG samples file not found at: {samples_path}. Make sure data collection has been run.")
    if not os.path.exists(markers_path):
        raise FileNotFoundError(f"EEG markers file not found at: {markers_path}. Make sure data collection has been run.")

    with open(samples_path, 'r') as json_file1:
        eeg_samples = json.load(json_file1)

    with open(markers_path, 'r') as json_file2:
        eeg_markers = json.load(json_file2)

    buffered_sample_number = len(eeg_samples)
    channel_number = len(eeg_samples[0])

    if verbose:
        print("OpenBCI Dataset Loaded")
        print("Dataset contains", buffered_sample_number, "buffered samples, and", channel_number, "channels")
        print("----------Printing Size of Each Channel Buffer---------")
        for i in range(buffered_sample_number):
            print()
            for j in range(channel_number):
                print(len(eeg_samples[i][j]), end=" ")
            print("| marker =", eeg_markers[i], end="")
        print("\n-----------------------------------------------------")

    eeg_samples_unbuffered = []
    eeg_markers_unbuffered = []

    for channel in range(channel_number + 1):
        eeg_samples_unbuffered.append([])

    prev_marker = 0
    for i, buffer in enumerate(eeg_samples):
        if i <= 1:
            continue
        new_marker = eeg_markers[i]
        buffer_size = len(buffer[0])
        for b in range(buffer_size):
            # sample = []
            for channel in range(channel_number):
                # sample.append(buffer[channel][b])
                scaled_data_point = buffer[channel][b] / 1000000 # BrainFlow returns uV, convert to V for MNE
                eeg_samples_unbuffered[channel].append(scaled_data_point)

            eeg_markers_unbuffered.append(new_marker)
            if (b==0) and new_marker != prev_marker:
                eeg_samples_unbuffered[channel_number].append(new_marker)
                prev_marker = new_marker
            else:
                eeg_samples_unbuffered[channel_number].append(0)
    
    # print(len(eeg_samples_unbuffered), len(eeg_samples_unbuffered[0]))
    # print(eeg_samples_unbuffered)

    return eeg_samples_unbuffered, eeg_markers_unbuffered

def convert_to_mne(name, board_id, save_name, save_path, samples, markers, save=True, show_ui=True):
    # adapted from: https://brainflow.readthedocs.io/en/stable/notebooks/brainflow_mne.html 

    params = BrainFlowInputParams()

    eeg_channels = BoardShim.get_eeg_channels(board_id.value)
    eeg_data = np.array(samples)#[eeg_channels, :]
    # eeg_data = eeg_data / 1000000 # BrainFlow returns uV, convert to V for MNE

    print("eeg_channels", eeg_channels)
    print("len(eeg_data)", len(eeg_data))
    print("len(eeg_data[0])", len(eeg_data[0]))
    print("eeg_data[0]", eeg_data[0])
    # print("eeg_data", eeg_data)

    # Creating MNE objects from brainflow data arrays
    ch_types = ['eeg'] * len(eeg_channels)
    ch_types.append('stim')
    ch_names = BoardShim.get_eeg_names(board_id.value)
    ch_names.append('STI 014')

    print("ch_types", ch_types)
    print("ch_names", ch_names)
    
    sfreq = BoardShim.get_sampling_rate(board_id.value)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(eeg_data, info)
    print("----------events--------")
    events = mne.find_events(raw)#, stim_channel="STI 014")
    print(events[:5])  # show the first 5 events
    raw.set_montage("standard_1005")
    raw = raw.filter(l_freq=0.2, h_freq=30, method='iir')  # bandpass filter


    # plot data
    if show_ui:
        raw.plot()
        raw.plot_psd(average=False)
        plt.show()

    # Create EPOCHS
    event_dict = {
        "Rest": 1,
        "Left Fist": 2,
        "Right Fist": 3
    }
    tmin, tmax = -0.2, 2  # define epochs around events (in s)
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, preload=True)

    print(epochs)
    print(epochs.event_id)

    # epochs["Left Hand"].average().plot()
    # epochs["Right Hand"].average().plot()
    # epochs["Rest"].average().plot()

    # # Plot topomaps
    # times = np.arange(0, 1, 0.1)
    # epochs.average().plot_topomap(times, ch_type='eeg')
    # epochs["Left Hand"].average().plot_topomap(times, ch_type='eeg')
    # epochs["Right Hand"].average().plot_topomap(times, ch_type='eeg')
    # epochs["Rest"].average().plot_topomap(times, ch_type='eeg')

    # Plot epochs
    if show_ui:
        epochs.plot(scalings='auto', events=True)

        plt.show()

    if save:
        # Save Epochs
        epochs.save(save_path + "/" + save_name + '-epo.fif', overwrite=True)

    return epochs

if __name__ == "__main__":
    import os
    import sys
    # Add the project root directory to Python path
    project_root = os.path.dirname("../../")
    sys.path.insert(0, project_root)

    name = "ezpz-test"
    samples, markers = load_openbci_data("../../data/", verbose=True)

    board_id = BoardIds.SYNTHETIC_BOARD
    # board_id = BoardIds.CYTON_BOARD
    convert_to_mne(board_id, name, name, "../../data", samples, markers, save=True)
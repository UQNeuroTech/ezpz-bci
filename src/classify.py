import argparse
import time
from pprint import pprint

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from backend.key_actuate import press_key

import json
import random

import torch
import numpy as np
import eegnet  # Make sure eegnet.py is accessible/importable
MODEL_PATH = "../models/reuben-openbci/data-reuben-2122-2205-3-classes/data-reuben-2122-2205-3-classes-MM.pth"  # e.g. "models/my_model.pth"
CHANS = 8  # Update as per your setup
TIME_POINTS = 801  # Update as per your setup

# These values should match your training data.
# Ideally, load from a file. For demo:
TRAIN_MEAN = -3.913006129388261e-06
TRAIN_STD = 4.6312790254451314e-05

# Load model once and set to eval
model = eegnet.EEGNetModel(chans=CHANS, time_points=TIME_POINTS)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

marker_dict = {
    0: "Nothing",
    1: "Rest", 
    2: "Left Fist", 
    3: "Right Fist"
}


def main():
    board_id = BoardIds.CYTON_BOARD
    
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = "/dev/ttyUSB0"

    pprint(BoardShim.get_board_descr(board_id))

    board = BoardShim(BoardIds.CYTON_BOARD, params)
    board.prepare_session()
    board.start_stream()

    iter = 0

    done = False

    eeg_samples = []

    eeg_markers = []

    start_time = time.time()
    prev_time = time.time()

    while not done:
        iter = iter + 1
        time.sleep(0.1)

        cur_time = time.time()

        if cur_time - prev_time > 2: 
            prev_time = cur_time

        data = board.get_board_data()  # get all data and remove it from internal buffer

        channels = board.get_eeg_channels(board_id)

        if iter == 1:
            print("Number of Channels:", len(data))

        eeg_sample = [data[i].tolist() for i in channels]

        prediction = classify_eeg_sample(eeg_sample)
        marker = marker_dict[prediction]
        press_key(prediction)
        print(f"Iteration: {iter}, Prediction: {marker} ({prediction})")
        

        # eeg_samples.append(eeg_sample)
        # eeg_markers.append(prompt_order[prompt_iter])

    board.stop_stream()
    board.release_session()

    samples_path = "eeg_samples.json"
    markers_path = "eeg_markers.json"

    with open(samples_path, 'w') as json_file1:
        json.dump(eeg_samples, json_file1)

    with open(markers_path, 'w') as json_file2:
        json.dump(eeg_markers, json_file2)


def classify_eeg_sample(eeg_sample):
    """
    eeg_sample: np.ndarray or list, shape [CHANS, TIME_POINTS]
    """
    # Convert to numpy if needed
    eeg_np = np.array(eeg_sample)

    # Z-score normalization (adapt as needed)
    eeg_np = (eeg_np - TRAIN_MEAN) / TRAIN_STD

    # Reshape to (batch, 1, chans, time_points)
    eeg_tensor = torch.tensor(eeg_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, chans, time_points]

    # Model prediction
    with torch.no_grad():
        logits = model(eeg_tensor)
        pred_class = torch.argmax(logits, dim=1).item()

    return pred_class



if __name__ == "__main__":
    main()

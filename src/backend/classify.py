
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from src.backend.connect import initalize_board
from src.backend import eegnet
from process_openbci_data import convert_to_mne
from src.backend.key_actuate import press_key

import json

import torch
import numpy as np

import mne

MODEL_PATH = "./data/ezpz-model.pth"  # e.g. "models/my_model.pth"
CHANS = 8
TIME_POINTS = 801

with open("./data/ezpz-model.json", 'r') as json_file1:
    train_metas = json.load(json_file1)

# These values should match your training data.
# Ideally, load from a file. For demo:
TRAIN_MEAN = train_metas["eeg_data_mean"]
TRAIN_STD = train_metas["eeg_data_std"]

# Load model once and set to eval
model = eegnet.EEGNetModel(chans=train_metas["chans"], time_points=train_metas["time_points"])
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

marker_dict = {
    0: "Rest", 
    1: "Left Fist", 
    2: "Right Fist"
}

def main():
    # board_id = BoardIds.SYNTHETIC_BOARD
    board_id = BoardIds.CYTON_BOARD
    board = initalize_board(board_id, "/dev/ttyUSB0")  # Adjust port as needed

    iter = 0

    done = False

    eeg_samples = []

    eeg_markers = []

    start_time = time.time()
    prev_time = time.time()

    while not done:
        time.sleep(0.1)

        cur_time = time.time()

        if cur_time - prev_time > 2.2: 
            iter = iter + 1
            prev_time = cur_time

            data = board.get_board_data()  # get all data and remove it from internal buffer

            channels = board.get_eeg_channels(board_id)
            channels = channels[:8]  # Limit to first 8 channels ---------------------------------------------------- TODO: ADJUST IN FUTURE

            if iter == 1:
                print("Number of Channels:", len(data))

            eeg_sample = [data[i].tolist() for i in channels]

            try:
                prediction, logits = classify_eeg_sample(eeg_sample)
                marker = marker_dict[prediction]
                print("logits:", logits)
                print(f"Iteration: {iter}, Prediction: {marker} ({prediction})")
                press_key(prediction)
            except Exception as e:
                print(f"Error classifying EEG sample: {e}")
                continue

            # eeg_samples.append(eeg_sample)
            # eeg_markers.append(prompt_order[prompt_iter])

    board.stop_stream()
    board.release_session()



def classify_eeg_sample(eeg_sample):
    """
    eeg_sample: np.ndarray or list, shape [CHANS, TIME_POINTS]
    """

    print("eeg_sample type/shape:", type(eeg_sample), np.array(eeg_sample).shape)

    # board_id = BoardIds.SYNTHETIC_BOARD
    board_id = BoardIds.CYTON_BOARD
    raw = convert_to_mne(board_id, " ", " ", "data", eeg_sample, None, save=False, classify=True, show_ui=False)

    # print("raw:", raw, " | data: ", raw.get_data().shape)
    # print("raw.info:", raw.info)
    # print("data:", raw.get_data())

    # Z-score normalization (adapt as needed)

    scaled = raw.get_data() / 1000000  # BrainFlow returns uV, convert to V for MNE
    normed = (scaled - TRAIN_MEAN) / TRAIN_STD

    # Convert to numpy
    eeg_np = np.array(normed)

    # Reshape to (batch, 1, chans, time_points)
    eeg_tensor = torch.tensor(eeg_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, chans, time_points]

    print("eeg_tensor.shape:", eeg_tensor.shape)

    # Model prediction
    with torch.no_grad():
        logits = model(eeg_tensor)
        pred_class = torch.argmax(logits, dim=1).item()

    return pred_class, logits

# def classify_eeg_sample(eeg_sample):
#     """
#     eeg_sample: np.ndarray or list, shape [CHANS, TIME_POINTS]
#     """

#     print("eeg_sample type/shape:", type(eeg_sample), np.array(eeg_sample).shape)

#     # Convert to numpy if needed
#     eeg_np = np.array(eeg_sample)



#     # Z-score normalization (adapt as needed)
#     eeg_np = (eeg_np - TRAIN_MEAN) / TRAIN_STD

#     # Reshape to (batch, 1, chans, time_points)
#     eeg_tensor = torch.tensor(eeg_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, chans, time_points]

#     print("eeg_tensor.shape:", eeg_tensor.shape)

#     # Model prediction
#     with torch.no_grad():
#         logits = model(eeg_tensor)
#         pred_class = torch.argmax(logits, dim=1).item()

#     return pred_class



if __name__ == "__main__":
    # import os
    # import sys  
    # # Add the project root directory to Python path
    # project_root = os.path.dirname("../../")
    # sys.path.insert(0, project_root)
    main()

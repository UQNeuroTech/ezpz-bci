import argparse
import time
import os
from pprint import pprint

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import json
import random

marker_dict = {
    0: "Nothing",
    1: "Rest", 
    2: "Left Fist", 
    3: "Right Fist"
}

def generate_prompt_order(n_pairs):
    """
    Generate a prompt order list with equal numbers of Left and Right Fist prompts,
    inserting a Rest (1) between each, and starting with [Nothing, Rest] and ending with Rest.

    Parameters:
        n_pairs (int): Number of Left and Right Fist prompts each (total 2 * n_pairs active prompts)

    Returns:
        List[int]: Prompt order list
    """
    # Create and shuffle active prompts
    active_prompts = [2] * n_pairs + [3] * n_pairs
    random.shuffle(active_prompts)

    # Start with Nothing and Rest
    prompt_order = [0, 1]

    # Interleave Rest (3) between each active prompt
    for prompt in active_prompts:
        prompt_order.append(prompt)
        prompt_order.append(1)

    return prompt_order


def add_nothing_prompts(lst):
    result = []
    for i, val in enumerate(lst):
        result.append(val)
        if i != len(lst) - 1:
            result.append(0)
    return result


def initalize_board(board_id, port):
    """
    Initialize the OpenBCI board and return the board object.
    """
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = port

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    return board


def main(ui_callback, is_running):
    batch_size = 5

    # Read cycle_count from categories.json
    cycle_count = 2
    try:
        json_file_path = "./src/gui/db/categories.json"
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                cycle_count = data.get("cycle_count", cycle_count)
    except Exception as e:
        print(f"Error reading cycle_count from categories.json: {e}")

    print(f"Will run for {cycle_count} cycles")

    prompt_order = generate_prompt_order(batch_size)
    print(prompt_order)
    prompt_order = add_nothing_prompts(prompt_order)
    print(prompt_order)

    board_id = BoardIds.SYNTHETIC_BOARD
    # board_id = BoardIds.CYTON_BOARD
    
    # BoardShim.enable_dev_board_logger()

    # params = BrainFlowInputParams()
    # params.serial_port = "/dev/ttyUSB0"

    # pprint(BoardShim.get_board_descr(board_id))

    # board = BoardShim(BoardIds.CYTON_BOARD, params)
    # board.prepare_session()
    # board.start_stream()

    board = initalize_board(board_id, "/dev/ttyUSB0")

    iter = 0
    prompt_iter = 0
    cycles_completed = 0

    done = False

    eeg_samples = []

    eeg_markers = []

    start_time = time.time()
    prev_time = time.time()

    while not done:
        iter = iter + 1
        time.sleep(0.1)

        # Check if we should stop the thread
        if not is_running():
            print("Data collection stopped by user")
            done = True
            continue

        cur_time = time.time()

        if cur_time - prev_time > 2: 
            current_marker = marker_dict[prompt_order[prompt_iter]]
            print(prompt_iter, '|', current_marker)

            # Call UI callback if provided
            if ui_callback is not None:
                ui_callback(current_marker)

            prompt_iter += 1
            prev_time = cur_time

            if prompt_iter >= len(prompt_order):
                cycles_completed += 1
                print(f"Completed cycle {cycles_completed} of {cycle_count}")

                # Check if we've reached the target number of cycles
                if cycles_completed >= cycle_count:
                    print(f"Reached target of {cycle_count} cycles. Stopping.")
                    done = True
                    continue

                # Generate new prompts for next cycle
                prompt_order = generate_prompt_order(batch_size)
                print(prompt_order)
                prompt_order = add_nothing_prompts(prompt_order)
                print(prompt_order)
                prompt_iter = 0
                data_discard = board.get_board_data()
                print("Data-discarded:", len(data_discard))
                print(marker_dict[prompt_order[prompt_iter]])
                continue

        data = board.get_board_data()  # get all data and remove it from internal buffer

        channels = board.get_eeg_channels(board_id)

        if iter == 1:
            print("Number of Channels:", len(data))

        eeg_sample = [data[i].tolist() for i in channels]

        eeg_samples.append(eeg_sample)
        eeg_markers.append(prompt_order[prompt_iter])

    board.stop_stream()
    board.release_session()

    samples_path = "./data/eeg_samples.json"
    markers_path = "./data/eeg_markers.json"

    with open(samples_path, 'w') as json_file1:
        json.dump(eeg_samples, json_file1)

    with open(markers_path, 'w') as json_file2:
        json.dump(eeg_markers, json_file2)

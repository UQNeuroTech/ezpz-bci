from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def initalize_board(board_id, port, is_synthetic):
    """
    Initialize the OpenBCI board and return the board object.
    """
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    if is_synthetic:
        #    Set the number of EEG channels using the 'other_info' parameter.
        #    To create an 8-channel device, we provide a string with 8 comma-separated values.
        #    The actual values do not matter for this purpose, only the count.
        params.other_info = '0,0,0,0,0,0,0,0'
        board_id = BoardIds.SYNTHETIC_BOARD

    elif board_id == BoardIds.CYTON_BOARD:
        params.serial_port = port

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    return board
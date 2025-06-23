from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def initalize_board(board_id, port):
    """
    Initialize the OpenBCI board and return the board object.
    """
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    if board_id == BoardIds.CYTON_BOARD:
        params.serial_port = port

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    return board
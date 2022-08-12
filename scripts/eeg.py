from typing import List, Tuple, Optional
import mne
import numpy as np
import pandas as pd
import serial.tools.list_ports
from mne_features.feature_extraction import extract_features
from nptyping import NDArray
try:
    from brainflow import BrainFlowInputParams, BoardShim, BoardIds
except ImportError:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


class EEG:
    """
    A class used to wrap all the communication with the OpenBCI EEG
    ...

    Attributes
    ----------
    board_id : int
        id of the OpenBCI board
    ip_port : int
        port for the board
    serial_port : str
        serial port for the board
    headset : str
        the headset name we use, will be presented in the metadata
    """
    def __init__(self, board_id: int = BoardIds.CYTON_DAISY_BOARD.value, ip_port: int = 6677,
                 serial_port: Optional[str] = None, headset: str = "72"):

        # Board Id and Headset Name
        self.board_id = board_id
        self.headset: str = headset

        # BrainFlowInputParams
        self.params = BrainFlowInputParams()
        self.params.ip_port = ip_port
        self.params.serial_port = serial_port if serial_port is not None else self.find_serial_port()
        self.params.headset = headset
        self.params.board_id = board_id
        self.board = BoardShim(board_id, self.params)

        # Other Params
        self.sfreq = self.board.get_sampling_rate(board_id)
        self.marker_row = self.board.get_marker_channel(self.board_id)
        self.eeg_names = self.get_board_names()

        self.configurations = 'x1030110Xx2030110Xx3030110Xx4030110Xx5030110Xx6030110Xx7030110Xx8030110Xx9030110Xx10030110Xx11030110Xx12030110Xx13131000Xx14131000Xx15131000Xx16131000X'

    def extract_trials(self, data: NDArray, num_stims, num_trials) -> [List[Tuple], List[List]]:
        """
        The method get ndarray and extract the labels and durations from the data.
        :param data: the data from the board.
        :return: duration, labels
        """

        # Init params
        durations, labels = [], []

        # Get marker indices
        markers_idx = np.where(data[self.marker_row, :] != 0)[0]

        # For each marker
        for idx in markers_idx:

            # Decode the marker
            status, label, _ = self.decode_marker(data[self.marker_row, idx])

            if status == 'start':

                labels.append(label)
                durations.append((idx,))

            elif status == 'stop':

                durations[-1] += (idx,)

        labels = [labels[x:x+num_stims] for x in range(0, num_stims*num_trials, num_stims)]
        return durations, labels

    def on(self):
        """Turn EEG On"""
        self.board.prepare_session()
        self.board.config_board(self.configurations)
        self.board.start_stream()

    def off(self):
        """Turn EEG Off"""
        self.board.stop_stream()
        self.board.release_session()

    def insert_marker(self, status: str, label: int, index: int):
        """Insert an encoded marker into EEG data"""

        marker = self.encode_marker(status, label, index)  # encode marker
        self.board.insert_marker(marker)  # insert the marker to the stream

    def get_board_data(self) -> NDArray:
        """The method returns the data from board and remove it"""
        return self.board.get_board_data()

    def get_board_names(self) -> List[str]:
        """The method returns the board's channels"""
        if self.headset == "72":
            return ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'O1', 'O2']
        else:
            return self.board.get_eeg_names(self.board_id)

    def get_board_channels(self, alternative=True) -> List[int]:
        """Get list with the channels locations as list of int"""
        if alternative:
            return self.board.get_eeg_channels(self.board_id)[:-3]
        else:
            return self.board.get_eeg_channels(self.board_id)

    def find_serial_port(self) -> str:
        """
        Return the string of the serial port to which the FTDI dongle is connected.
        If running in Synthetic mode, return ""
        Example: return "COM5"
        """
        if self.board_id == BoardIds.SYNTHETIC_BOARD:
            return ""
        else:
            plist = serial.tools.list_ports.comports()
            FTDIlist = [comport for comport in plist if comport.manufacturer == 'FTDI']
            if len(FTDIlist) > 1:
                raise LookupError(
                    "More than one FTDI-manufactured device is connected. Please enter serial_port manually.")
            if len(FTDIlist) < 1:
                raise LookupError("FTDI-manufactured device not found. Please check the dongle is connected")
            return FTDIlist[0].name

    @staticmethod
    def encode_marker(status: str, label: int, index: int):
        """
        Encode a marker for the EEG data.
        :param status: status of the stim (start/end)
        :param label: the label of the stim (right -> 0, left -> 1, idle -> 2, tongue -> 3, legs -> 4)
        :param index: index of the current label
        :return:
        """
        markerValue = 0
        if status == "start":
            markerValue += 1
        elif status == "stop":
            markerValue += 2
        else:
            raise ValueError("incorrect status value")

        markerValue += 10 * label

        markerValue += 100 * index

        return markerValue

    @staticmethod
    def decode_marker(marker_value: int) -> (str, int, int):
        """
        Decode the marker and return a tuple with the status, label and index.
        Look for the encoder docs for explanation for each argument in the marker.
        :param marker_value:
        :return:
        """
        if marker_value % 10 == 1:
            status = "start"
            marker_value -= 1
        elif marker_value % 10 == 2:
            status = "stop"
            marker_value -= 2
        else:
            raise ValueError("incorrect status value. Use start or stop.")

        label = ((marker_value % 100) - (marker_value % 10)) / 10

        index = (marker_value - (marker_value % 100)) / 100

        return status, int(label), int(index)

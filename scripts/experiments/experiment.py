import os
import random
import sys
from datetime import datetime
from tkinter import messagebox
from tkinter.filedialog import askdirectory
from scripts.eeg import EEG
from psychopy import event
try:
    from brainflow import BoardIds
except ImportError:
    from brainflow.board_shim import BoardIds


class Experiment:
    def __init__(self, eeg, num_trials, classes, num_stims=100):
        self.num_trials: int = num_trials
        self.num_stims: int = num_stims
        self.eeg: EEG = eeg

        if self.eeg.board_id == BoardIds.SYNTHETIC_BOARD:
            messagebox.showwarning(title="bci4als WARNING", message="You are running a synthetic board!")
            self.debug = True
        else:
            self.debug = False
        # override in subclass
        self.cue_length = None
        self.stim_length = None
        self.session_directory = None
        self.enum_image = {0: 'target_1', 1: 'target_2', 2: 'idle'}
        self.experiment_type = None
        # self.skip_after = None

        # labels
        self.labels = []
        self.targets = []
        self._init_labels(keys=classes)
        self._init_targets()

        if num_stims < 10:
            raise IOError("num_stims must be greater then 10")

    def run(self):
        pass

    def write_metadata(self):
        # The path of the metadata file
        path = os.path.join(self.session_directory, 'metadata.txt')

        with open(path, 'w') as file:
            # Datetime
            file.write(f'Experiment datetime: {datetime.now()}\n\n')

            # Channels
            file.write('EEG Channels:\n')
            file.write('*************\n')
            for index, ch in enumerate(self.eeg.get_board_names()):
                file.write(f'Channel {index + 1}: {ch}\n')

            # Experiment data
            file.write('\nExperiment Data\n')
            file.write('***************\n')
            file.write(f'Experiment Type: {self.experiment_type}\n')
            file.write(f'Num of trials: {self.num_trials}\n')
            file.write(f'Num of stimulus per trail: {self.num_stims}\n')
            file.write(f'Single trial length: {self.stim_length*self.num_stims}\n')
            file.write(f'Cue length: {self.cue_length}\n')
            file.write(f'Stim length: {self.stim_length}\n')
            file.write(f'Labels Enum: {self.enum_image}\n')
            # file.write(f'Skip After: {self.skip_after}\n')

    def _ask_subject_directory(self):
        """
        init the current subject directory
        :return: the subject directory
        """

        # get the CurrentStudy recording directory
        if not messagebox.askokcancel(title='bci4als',
                                      message="Welcome to the motor imagery EEG recorder."
                                              "\n\nNumber of trials: {}\n\n"
                                              "Please select the subject directory:".format(self.num_trials)):
            sys.exit(-1)

        # show an "Open" dialog box and return the path to the selected file
        init_dir = os.path.join(os.path.split(os.path.abspath(''))[0], 'recordings')
        subject_folder = askdirectory(initialdir=init_dir)
        if not subject_folder:
            sys.exit(-1)
        return subject_folder

    @staticmethod
    def get_keypress():
        """
        Get keypress of the user
        :return: string of the key
        """
        keys = event.getKeys()
        if keys:
            return keys[0]
        else:
            return None

    @staticmethod
    def create_session_folder(subject_folder: str) -> str:
        """
        The method create new folder for the current session. The folder will be at the given subject
        folder.
        The method also creating a metadata file and locate it inside the session folder
        :param subject_folder: path to the subject folder
        :return: session folder path
        """

        current_sessions = []
        for f in os.listdir(subject_folder):

            # try to convert the current sessions folder to int
            # and except if one of the sessions folder is not integer
            try:
                current_sessions.append(int(f))

            except ValueError:
                continue

        # Create the new session folder
        session = (max(current_sessions) + 1) if len(current_sessions) > 0 else 1
        session_folder = os.path.join(subject_folder, str(session))
        os.mkdir(session_folder)

        return session_folder

    def _init_labels(self, keys):
        """
        This method creates list containing a stimulus vector
        :return: the stimulus in each trial (list)
        """

        # Create the balance label vector
        # 10% for target_1, 10% for target_2 and 80% for idle
        idle = keys[2]
        target_1 = keys[0]
        target_2 = keys[1]
        for j in range(self.num_trials):
            # idle mode
            temp = [idle] * self.num_stims
            # target modes
            percent = int(self.num_stims * 0.1)
            temp[0:percent] = [target_1] * percent
            temp[percent:2*percent] = [target_2] * percent

            random.shuffle(temp)

            self.labels += [temp]

    def _init_targets(self):
        """
        this method creates a list of targets - 0 or 1
        :return: list of targets
        """
        # randomly choose the target for the trail

        targets = [random.randint(0, 1) for _ in range(self.num_trials)]
        self.targets = targets

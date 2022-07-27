from scripts.bci4als_code.eeg import EEG
from scripts.bci4als_code.experiments.offline import OfflineExperiment
from psychopy import visual, event
import sys


class OnlineExperiment(OfflineExperiment):
    """
    Class for running an online MI experiment.

    Attributes:
    ----------

        num_trials (int):
            Amount of trials in the experiment.

        buffer_time (float):
            Time in seconds for collecting EEG data before model's prediction.

        threshold (int):
            The amount the times the model need to be correct (predict = stim) before moving to the next stim.

    """

    def __init__(self, eeg: EEG, num_trials: int, num_stims=100, stim_length=0.8):
        super().__init__(eeg, num_trials, stim_length, num_stims=num_stims)
        # experiment params
        self.experiment_type = "Online"
        self.win = None

    def validate_labels(self, labels):
        pass

    def check_prediction(self):
        """
        target_1 = red_square
        target_2 = purple_triangle
        :return the corresponding label for the chosen target
        """
        # Init psychopy and screen params
        self._init_window()

        win = self.window_params['main_window']
        mouse_pos = event.Mouse()

        # init visuals
        target_1 = visual.ImageStim(win, self.images_path[self.enum_image[0]], pos=(180, -75),
                                    size=(235, 200))
        target_2 = visual.ImageStim(win, self.images_path[self.enum_image[1]], pos=(-180, -75),
                                    size=(210, 210))
        message = visual.TextStim(win, 'Please choose the target you had concentrated on:',
                                  color='white', height=35, pos=(0, 150))
        # show options
        target_1.draw()
        target_2.draw()
        message.draw()
        win.flip()

        while not mouse_pos.isPressedIn(target_1) or not mouse_pos.isPressedIn(target_2):
            if mouse_pos.isPressedIn(target_1):
                win.close()
                return 1
            if mouse_pos.isPressedIn(target_2):
                win.close()
                return 2
            # Halt if escape was pressed
            if 'escape' == self.get_keypress():
                sys.exit(-1)

    # def run(self, use_eeg: bool = True, full_screen: bool = False):
    #     # Init the current experiment folder
    #     self.subject_directory = self._ask_subject_directory()
    #     self.session_directory = self.create_session_folder(self.subject_directory)
    #
    #     # Create experiment's metadata
    #     self.write_metadata()
    #
    #     # Init psychopy and screen params
    #     self._init_window()
    #
    #     # Start stream
    #     # initialize headset
    #     print("Turning EEG connection ON")
    #     self.eeg.on()
    #
    #     print(f"Running {self.num_trials} trials")
    #     # Run trials
    #     # Messages for user
    #     for s in range(self.num_stims):
    #         # Show stim on window
    #         self._show_stimulus(0, s)
    #     self.window_params['main_window'].close()
    #
    #     # Export and return the data
    #     trials, data, durations = self._extract_trials()
    #
    #     print("Turning EEG connection OFF")
    #     self.eeg.off()
    #
    #     # Dump files to pickle
    #     # self._export_files(trials, data, durations)
    #
    #     return trials, data, durations, self.labels

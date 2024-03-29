import os
import pickle
import sys
import time
from tkinter import messagebox
from typing import Dict, Any
from scripts.experiments.experiment import Experiment
from scripts.eeg import EEG
from psychopy import visual


class OfflineExperiment(Experiment):

    def __init__(self, eeg: EEG, num_trials: int, stim_length: float=1,
                 next_length: float = 2, cue_length: float = 0.25, ready_length: float = 1,
                 full_screen: bool = True, classes: tuple = (0, 1, 2),
                 num_stims=100):

        super().__init__(eeg, num_trials, classes, num_stims)
        self.experiment_type = "Offline"
        self.window_params: Dict[str, Any] = {}
        self.full_screen: bool = full_screen

        # trial times
        self.cue_length: float = cue_length
        self.next_length: float = next_length
        self.ready_length: float = ready_length
        self.stim_length: float = stim_length

        # paths
        self.subject_directory: str = ''
        self.session_directory: str = ''
        self.images_path: Dict[str, str] = {
            'target_1': os.path.join(os.path.dirname(__file__), 'images', 'red_square.jpg'),
            'target_2': os.path.join(os.path.dirname(__file__), 'images', 'purple_triangle.png'),
            'idle': os.path.join(os.path.dirname(__file__), 'images', 'blue_circle.png'), }
        self.visual_params: Dict[str, Any] = {'text_color': 'white', 'text_height': 48}

    def _init_window(self):
        """
        init the psychopy window
        :return: dictionary with the window, red_square, purple_triangle and blue_circle (idle).
        """

        # Create the main window
        main_window = visual.Window(monitor='testMonitor', units='pix',
                                    color='black', fullscr=self.full_screen)

        # Create stimulus
        target_1_stim = visual.ImageStim(main_window, image=self.images_path['target_1'])
        target_2_stim = visual.ImageStim(main_window, image=self.images_path['target_2'])
        idle_stim = visual.ImageStim(main_window, image=self.images_path['idle'])

        self.window_params = {'main_window': main_window, 'target_1': target_1_stim,
                              'target_2': target_2_stim, 'idle': idle_stim}

    def _user_messages(self, target):
        """
        Show for the user messages in the following order:
            1. Next message
            2. Cue for the trial condition
            3. Ready & state message
        :param target: the current target of the trail
        """

        color = self.visual_params['text_color']
        height = self.visual_params['text_height']
        trial_image = self.enum_image[target]
        win = self.window_params['main_window']

        # Show 'next' message
        next_message = visual.TextStim(win, 'The target is...', color=color, height=height)
        next_message.draw()
        win.flip()

        # sleep between clues
        time.sleep(self.next_length)

        # Show cue & play sound
        cue = visual.ImageStim(win, self.images_path[trial_image])
        cue.draw()
        win.flip()
        time.sleep(self.cue_length)

        # Show ready message
        ready_message = visual.TextStim(win, 'Ready...', pos=[0, 0], color=color, height=height)
        ready_message.draw()
        win.flip()
        time.sleep(self.ready_length)

    def _show_stimulus(self, trial_index, stim_index):
        """
        Show the current stimuli on screen and wait.
        Additionally response to shutdown key.
        :param trial_index: the current trial index
        :param stim_index: the current stim in index
        """

        # Params
        win = self.window_params['main_window']
        trial_img = self.enum_image[self.labels[trial_index][stim_index]]

        # Draw and push marker
        self.eeg.insert_marker(status='start', label=self.labels[trial_index][stim_index], index=trial_index)
        self.window_params[trial_img].draw()
        win.flip()

        # Wait
        time.sleep(self.stim_length)
        self.eeg.insert_marker(status='stop', label=self.labels[trial_index][stim_index], index=trial_index)

        # make blink between stimulus
        win = self.window_params['main_window']
        win.flip()
        time.sleep(0.25)

        # Halt if escape was pressed
        if 'escape' == self.get_keypress():
            sys.exit(-1)

    def _extract_trials(self):
        """
        The method extract from the offline experiment collected EEG data and
        split it into trials.
        The method export a pickle file to the subject directory with a list
        with all the trials.
        :return: list of trials where each trial is a List DataFrame.
        """

        # Wait to get the last marker
        time.sleep(0.5)

        # Extract the data
        trials = []
        data = self.eeg.get_board_data()
        ch_channels = self.eeg.get_board_channels()
        durations, labels = self.eeg.extract_trials(data, self.num_stims, self.num_trials)
        # Assert the labels
        self.validate_labels(labels)

        durations = [durations[x:x + self.num_stims] for x in range(0, len(durations), self.num_stims)]
        # Append each
        for duration in durations:
            trial = []
            for start, end in duration:
                stim = data[ch_channels, start:end]
                trial.append(stim)
            trials.append(trial)

        return trials, data, durations

    def validate_labels(self, labels):
        """
        Check if the labels are OK
        """
        assert self.labels == labels, 'The labels are not equals to the extracted labels'

    def _export_files(self, trials, data=None, durations=None):
        """
        Export the experiment files (trials & labels)
        """

        # Dump to pickle
        trials_path = os.path.join(self.session_directory, 'trials.pickle')
        print(f"Dumping extracted trials recordings to {trials_path}")
        pickle.dump(trials, open(trials_path, 'wb'))

        # Save the labels as pickle file
        labels_path = os.path.join(self.session_directory, 'labels.pickle')
        print(f"Saving labels to {labels_path}")
        pickle.dump(self.labels, open(labels_path, 'wb'))

        # Save the labels as pickle file
        targets_path = os.path.join(self.session_directory, 'targets.pickle')
        print(f"Saving targets to {targets_path}")
        pickle.dump(self.targets, open(targets_path, 'wb'))

        # Save the raw_data as pickle file
        data_path = os.path.join(self.session_directory, 'raw_data.pickle')
        print(f"Saving raw data to {data_path}")
        pickle.dump(data, open(data_path, 'wb'))

        # Save the durations as pickle file
        durations_path = os.path.join(self.session_directory, 'durations.pickle')
        print(f"Saving durations for data splitting in {durations_path}")
        pickle.dump(durations, open(durations_path, 'wb'))

    def run(self):
        # Init the current experiment folder
        self.subject_directory = self._ask_subject_directory()
        self.session_directory = self.create_session_folder(self.subject_directory)

        # Create experiment's metadata
        self.write_metadata()

        messagebox.showinfo(title='bci4als', message='Start running trials...')

        # Init psychopy and screen params
        self._init_window()

        # Start stream
        # initialize headset
        print("Turning EEG connection ON")
        self.eeg.on()

        print(f"Running {self.num_trials} trials")
        # Run trials
        for t in range(self.num_trials):
            # Messages for user
            target = self.targets[t]
            self._user_messages(target)
            for s in range(self.num_stims):
                # Show stim on window
                self._show_stimulus(t, s)
        self.window_params['main_window'].close()

        # Export and return the data
        trials, data, durations = self._extract_trials()

        print("Turning EEG connection OFF")
        self.eeg.off()

        # Dump files to pickle
        self._export_files(trials, data, durations)

        return trials, self.labels

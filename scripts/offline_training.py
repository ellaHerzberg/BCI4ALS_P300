from bci4als_code.eeg import EEG
from scripts.bci4als_code.experiments.offline import OfflineExperiment
from main import load_data
import numpy as np
from classifier import csp_lda
import pickle
import os
import time


def offline_experiment():
    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2

    eeg = EEG(board_id=SYNTHETIC_BOARD)
    exp = OfflineExperiment(eeg=eeg, num_trials=2, stim_length=0.8, cue_length=1,
                            full_screen=True, audio=False, num_stims=10)

    exp.run()
    time.sleep(1)

    session_directory = exp.session_directory
    down_sample_target, down_sample_idle = load_data(session_directory, plot=False)
    targets, idles = [], []
    targets.extend(down_sample_target)
    idles.extend(down_sample_idle)

    # Prepare data for the classifier:
    labels = np.concatenate([np.zeros(len(idles)), np.ones(len(targets))])
    trials = np.vstack([np.array(idles), np.array(targets)])

    # Classification
    classifier = csp_lda(trials, labels)

    # Dump the MLModel
    pickle.dump(classifier, open(os.path.join(session_directory, 'model.pickle'), 'wb'))


if __name__ == '__main__':
    offline_experiment()

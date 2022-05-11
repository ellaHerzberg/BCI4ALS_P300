import os
import pickle
from bci4als.eeg import EEG
from bci4als.ml_model import MLModel
from bci4als.experiments.offline import OfflineExperiment


def offline_experiment():

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2
    eeg = EEG(board_id=SYNTHETIC_BOARD)

    exp = OfflineExperiment(eeg=eeg, num_trials=3, trial_length=0.5, cue_length=1.5,
                            full_screen=True, audio=False)

    trials, labels = exp.run()

    # TODO: check if relevant
    # Classification
    model = MLModel(trials=trials, labels=labels)
    session_directory = exp.session_directory
    model.offline_training(eeg=eeg, model_type='csp_lda')

    # Dump the MLModel
    pickle.dump(model, open(os.path.join(session_directory, 'model.pickle'), 'wb'))


if __name__ == '__main__':

    offline_experiment()


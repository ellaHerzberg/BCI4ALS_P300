from bci4als_code.eeg import EEG
# from bci4als.ml_model import MLModel
from scripts.bci4als_code.experiments.offline import OfflineExperiment


def offline_experiment():
    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2
    eeg = EEG(board_id=2)

    exp = OfflineExperiment(eeg=eeg, num_trials=3, stim_length=0.5, cue_length=2,
                            full_screen=True, audio=False, num_stims=20)

    trials, labels = exp.run()

    # Classification
    # TODO: Implement on our own

    session_directory = exp.session_directory

    # model = MLModel(trials=trials, labels=labels)
    # model.offline_training(eeg=eeg, model_type='csp_lda')
    #
    # # Dump the MLModel
    # pickle.dump(model, open(os.path.join(session_directory, 'model.pickle'), 'wb'))


if __name__ == '__main__':
    offline_experiment()

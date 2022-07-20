import pickle
from scripts.bci4als_code.experiments.online import OnlineExperiment
from bci4als_code.eeg import EEG
from preprocess import *


def run_experiment(model_path: str):

    model = pickle.load(open(model_path, 'rb'))

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2
    eeg = EEG(board_id=SYNTHETIC_BOARD)

    exp = OnlineExperiment(eeg=eeg, num_trials=1, num_stims=10, stim_length = 0.8)

    trials_before, data, durations, labels = exp.run(use_eeg=True, full_screen=False)
    trials, labels, targets = general_preprocess(data, durations, labels)
    mean_target_1_data, mean_target_2_data, mean_idle_data = get_mean_of_stims(trials, labels, [0])
    down_sample_target_1 = down_sample(mean_target_1_data, sample_rate)
    down_sample_target_2 = down_sample(mean_target_2_data, sample_rate)
    down_sample_idle = down_sample(mean_idle_data, sample_rate)

    # Prepare data for the classifier:
    idle_prediction = model.predict(np.array(down_sample_idle))
    target_1_prediction = model.predict(np.array(down_sample_target_1))
    target_2_prediction = model.predict(np.array(down_sample_target_2))
    print("----------------------------------------success")


if __name__ == '__main__':

    model_path = f"..\\recordings\\4\\model.pickle"
    # model_path = None  # use if synthetic
    run_experiment(model_path=model_path)

# PAY ATTENTION!
# If synthetic - model Path should be none
# otherwise choose a model path

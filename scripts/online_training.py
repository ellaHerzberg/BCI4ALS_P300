from scripts.experiments.online import OnlineExperiment
from scripts.eeg import EEG
from preprocess import *
import numpy as np
from classifier import csp_lda
import os


def _run_experiment(exp, model):
    trials_before, data, durations, labels = exp.run(use_eeg=True, full_screen=True)

    # preprocess data
    trials, labels, targets = general_preprocess(data, durations, labels)
    mean_target_1_data, mean_target_2_data, mean_idle_data = get_mean_of_stims(trials, labels, [0])
    down_sample_target_1 = down_sample(mean_target_1_data, sample_rate)
    down_sample_target_2 = down_sample(mean_target_2_data, sample_rate)
    down_sample_idle = down_sample(mean_idle_data, sample_rate)

    # Prepare data for the classifier:
    idle_prediction = model.predict(np.array(down_sample_idle))
    target_1_prediction = model.predict(np.array(down_sample_target_1))
    target_2_prediction = model.predict(np.array(down_sample_target_2))

    # get participant response
    correct_label = exp.check_prediction()

    # checking prediction. idle and distractor should be 0.
    if idle_prediction != 0:
        print("idle is not 0")
    if target_1_prediction and target_2_prediction:
        print("distractor and target have same prediction")

    if correct_label == 1:
        if target_1_prediction:
            print("success for red square")
        else:
            print("failure for red square")
        return down_sample_idle, down_sample_target_1

    else:
        if target_2_prediction:
            print("success for purple triangle")
        else:
            print("failure for purple triangle")

        return down_sample_idle, down_sample_target_2


def run_experiment(model_path: str):
    model = pickle.load(open(model_path, 'rb'))

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2
    eeg = EEG(board_id=CYTON_DAISY)
    exp = OnlineExperiment(eeg=eeg, num_trials=1, num_stims=65, stim_length=0.8)

    idles, targets = [], []
    for i in range(1, 6):  # run twice for accuracy
        idle, target = _run_experiment(exp, model)

        idles.extend(idle)
        targets.extend(target)
        # every 2 trials update model
        if i % 2 == 0:
            # update classifier
            labels = np.concatenate([np.zeros(len(idles)), np.ones(len(targets))])
            trials = np.vstack([np.array(idles), np.array(targets)])
            print(trials.shape, labels.shape)
            model = csp_lda(trials, labels)

    pickle.dump(model, open(os.path.join(exp.session_directory, 'model.pickle'), 'wb'))


if __name__ == '__main__':
    model_path = f"..\\recordings\\model\\model.pickle"
    run_experiment(model_path=model_path)

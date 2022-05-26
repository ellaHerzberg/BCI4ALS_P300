from scipy.signal import lfilter, butter
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pandas import read_csv


fs = 256  # (Hz)
# elec = 13
# trials_amount = 15
# stimuli = 30
# labels_amount = 3
# trials = np.random.randint(0, 100, (trials_amount, elec, fs * stimuli))
# labels = np.random.randint(low=0, high=labels_amount, size=(trials_amount, stimuli))
# trials = np.reshape(trials, (trials_amount, elec, fs, stimuli))

lowcut = 1
highcut = 40

with open("..\\recordings\\23\\trials.pickle", "rb") as f:
    trials = pickle.load(f)

# data_labels = open("D:\\University\\Bachelor\\Brain and Cognition science\\Third year\\BCI4ALS\\BCI4ALS_P300\\recordings\\23\\labels.csv")
# labels = []
# for row in data_labels:
#     labels.append(row)
# labels = np.genfromtxt("D:\\University\\Bachelor\\Brain and Cognition science\\Third year\\BCI4ALS\\BCI4ALS_P300\\recordings\\23\\labels.csv"
#                        ,delimiter=",")
labels = np.loadtxt("..\\recordings\\23\\labels.csv",
                  delimiter=",")

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


times = np.arange(256)
# filtered = butter_bandpass_filter(trials, lowcut, highcut, fs, order=2)
plt.subplot(121)
plt.plot(times, trials[1, 1, 1, :])
plt.xlabel('before')
plt.subplot(122)
# plt.plot(times, filtered[1, 1, :, 1])
plt.xlabel('after')
plt.show()

mean_trails = np.zeros((trials_amount, elec, fs, labels_amount))
for i in range(trials_amount):
    for j in range(labels_amount):
        indx_trials = (labels[i, :] == 0)
        m_trials = trials[i, :, :, indx_trials]
        t_trials = np.mean(m_trials, 0)
        mean_trails[i, :, :, j] = t_trials

import matplotlib.pyplot as plt

"""
this file contains plotting functions to help us visualize the data
and choose the best electrodes to use for the ML
"""

def plot_electrodes(trial, elec):
    """
    plot single electrode from single trial
    """
    plt.plot(range(len(trial[elec])), trial[elec])
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


# --plot all electrodes in one figure
def plot_all_mean_data(mean_target_data, mean_distract_data, mean_idle_data):
    """
    plot means for every stimulus for every electrode
    """
    elec_len = 8

    fig, axs = plt.subplots(2, elec_len//2, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5)
    axs = axs.ravel()
    for i in range(elec_len):
        # plt.figure()
        axs[i].set_title('electorde #' + str(i+1))
        axs[i].plot(range(len(mean_target_data[0][i])), mean_target_data[0][i])
        axs[i].plot(range(len(mean_idle_data[0][i])), mean_idle_data[0][i])
        axs[i].plot(range(len(mean_distract_data[0][i])), mean_distract_data[0][i])
        axs[i].legend(['Target', 'Idle', 'Distract'], loc='lower right')
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Frequency")
    plt.show()


def plot_all_down_sample(down_sample_target, down_sample_idle):
    """
    plot down sampled for every stimulus for every electrode
    """
    elec_len = 8

    fig, axs = plt.subplots(2, elec_len//2, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5)
    axs = axs.ravel()
    for i in range(elec_len):
        axs[i].set_title('electorde #' + str(i+1))
        # fig.set_size_inches(12, 8)
        axs[i].plot(range(len(down_sample_target[0][i])), down_sample_target[0][i])
        axs[i].plot(range(len(down_sample_idle[0][i])), down_sample_idle[0][i])
        axs[i].legend(['Target', 'Idle'], loc='lower right')
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Frequency")
    plt.show()


# --plot every electrode in a single figure
def plot_mean_data(mean_target_data, mean_distract_data, mean_idle_data):
    """
    plot means for every stimulus for every electrode
    """
    elec_len = 8
    for i in range(elec_len):
        plt.figure()
        plt.suptitle('electorde #' + str(i + 1))
        plt.plot(range(len(mean_target_data[0][i])), mean_target_data[0][i])
        plt.plot(range(len(mean_idle_data[0][i])), mean_idle_data[0][i])
        plt.plot(range(len(mean_distract_data[0][i])), mean_distract_data[0][i])
        plt.legend(['Target', 'Idle', 'Distract'], loc='lower right')
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.show()


def plot_down_sample(down_sample_target, down_sample_idle):
    """
    plot down sampled for every stimulus for every electrode
    """
    elec_len = 8
    for i in range(elec_len):
        plt.figure()
        plt.suptitle('electorde #' + str(i + 1))
        # fig.set_size_inches(12, 8)
        plt.plot(range(len(down_sample_target[0][i])), down_sample_target[0][i])
        plt.plot(range(len(down_sample_idle[0][i])), down_sample_idle[0][i])
        plt.legend(['Target', 'Idle'], loc='lower right')
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.show()

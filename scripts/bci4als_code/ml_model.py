import os
import pickle
from typing import List
import mne
import pandas as pd
from scripts.bci4als_code.eeg import EEG
import numpy as np
from matplotlib.figure import Figure
from mne.channels import make_standard_montage
from mne.decoding import CSP
from mne_features.feature_extraction import FeatureExtractor

from nptyping import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler


class MLModel:
    """
    A class used to wrap all the ML model train, partial train and predictions

    ...

    Attributes
    ----------
    trials : list
        a formatted string to print out what the animal says
    """

    def __init__(self, trials: List[pd.DataFrame], labels: List[int]):

        self.trials: List[NDArray] = [t.to_numpy().T for t in trials]
        self.labels: List[int] = labels
        self.debug = True
        self.clf = None

    def offline_training(self, eeg: EEG, model_type: str = 'csp_lda'):

        if model_type.lower() == 'csp_lda':

            self._csp_lda(eeg)

        elif model_type.lower() == 'svm':
            self._svm(eeg)
        else:
            raise NotImplementedError(f'The model type `{model_type}` is not implemented yet')

    def create_epoch_array(self, eeg: EEG):
        # convert data to mne.Epochs
        ch_names = eeg.get_board_names()
        print(ch_names)
        ch_types = ['eeg'] * len(ch_names)
        sfreq: int = eeg.sfreq
        n_samples: int = min([t.shape[1] for t in self.trials])
        epochs_array: np.ndarray = np.stack([t[:, :n_samples] for t in self.trials])

        info = mne.create_info(ch_names, sfreq, ch_types)
        epochs = mne.EpochsArray(epochs_array, info)

        # set montage
        montage = make_standard_montage('standard_1020')
        epochs.set_montage(montage)

        # Apply band-pass filter
        epochs.filter(7., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)

        return epochs

    def _svm(self, eeg: EEG):

        print('Training PCA & SVM model')
        epochs = self.create_epoch_array(eeg)

        data = epochs.get_data()

        # for feature extraction
        selected_funcs = ['pow_freq_bands', 'variance']

        # Assemble a classifier
        # pca = PCA(n_components=24)
        # pca.fit(data)
        # data = pca.transform(data)


        self.clf = make_pipeline(FeatureExtractor(sfreq=eeg.sfreq,
                                         selected_funcs=selected_funcs),
                                 StandardScaler(),
                                 SVC(gamma='auto'))
        self.clf.fit(data, self.labels)

    def _csp_lda(self, eeg: EEG):

        print('Training CSP & LDA model')

        epochs = self.create_epoch_array(eeg)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)

        # Use scikit-learn Pipeline
        self.clf = Pipeline([('CSP', csp), ('LDA', lda)])

        # fit transformer and classifier to data
        self.clf.fit(epochs.get_data(), self.labels)

    def online_predict(self, data: NDArray, eeg: EEG):
        # Prepare the data to MNE functions
        data = data.astype(np.float64)

        # Filter the data
        data = mne.filter.filter_data(data, l_freq=7, h_freq=30, sfreq=eeg.sfreq, verbose=False)

        # Laplacian
        # data = eeg.laplacian(data, eeg.get_board_names())

        print(data.shape)
        # Predict
        prediction = self.clf.predict(data[np.newaxis])[0]


        return prediction

    def partial_fit(self, eeg, X: NDArray, y: int):

        # Append X to trials
        self.trials.append(X)

        # Append y to labels
        self.labels.append(y)

        # Fit with trials and labels
        self._csp_lda(eeg)

    def cross_validation(self, eeg: EEG, n_splits: int = 5):
        epochs = self.create_epoch_array(eeg)

        kf = KFold(n_splits=n_splits, shuffle=True)
        # scoring = ('r2', 'neg_mean_squared_error')

        cv_results = cross_validate(self.clf, epochs.get_data(), self.labels, cv=kf, scoring='accuracy', return_train_score=False)
        print(cv_results)

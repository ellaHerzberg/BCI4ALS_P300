from mne.decoding import CSP
from mne_features.feature_extraction import FeatureExtractor

from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score


def plot_decision_boundry(clf, X, y, stimuli):
    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    fig, ax = plt.subplots()
    # title for the plots
    title = ('Approximate decision surface of classifier')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    colors = ["navy", "turquoise", "red"]
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.2)
    # ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    for color, i, target_name in zip(colors, range(len(stimuli)), stimuli):
        # predicted = np.logical_and(correct,  labels == i)
        # ax.scatter(X[predicted, 0], X[predicted, 1], alpha=0.8, color='green', linewidths=4)
        ax.scatter(X[y == i, 0], X[y == i, 1], alpha=0.8, color=color, label=target_name)

    ax.set_ylabel('Feature 1')
    ax.set_xlabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()


def cross_validation(clf, trials, labels):

    n_splits = 4
    kf = KFold(n_splits=n_splits, shuffle=True)
    scoring = {'accuracy' : make_scorer(accuracy_score),
               'precision' : make_scorer(precision_score, average='macro'),
               'recall' : make_scorer(recall_score, average='macro')}

    cv_results = cross_validate(clf, trials, labels, cv=kf, scoring=scoring,
                                return_train_score=False)
    return cv_results

def print_scores(cv_results):
    print(f"\tAccuracy: {cv_results['test_accuracy'].mean():.2f}")
    print(f"\tPrecision: {cv_results['test_precision'].mean():.2f}")
    print(f"\tRecall: {cv_results['test_recall'].mean():.2f}")
    print(f"\tRunning time: {cv_results['fit_time'].mean():.2f}")


def csp_lda(trials, labels, stimuli=("idle", "target")):
    csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
    # Take for plotting only first two features
    csp_proj = csp.fit_transform(trials, labels)
    lda = LinearDiscriminantAnalysis()

    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    clf.fit(trials, labels)

    prediction = clf.predict(trials)
    correct = prediction == labels

    csp_lda_score = cross_validation(clf, trials, labels)
    print(f"CSP LDA Scores:")
    print_scores(csp_lda_score)

    lda_plot = LinearDiscriminantAnalysis()
    lda_plot.fit(csp_proj[:, :2], labels)
    plot_decision_boundry(lda_plot, csp_proj, labels, stimuli)
    # csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    return csp_lda_score


# trials = epochs.get_data()
# csp_lda_score = csp_lda(trials[1:], labels[1:])
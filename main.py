import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def draw_figure(X, y, feature_names):
    # plotting data in 2D with axes sampled # a) at random
    # b) from same electrode
    # c) from same feature type
    num_features = 9
    num_electrodes = 48
    # a) indices drawn at random
    i0, i1 = np.random.randint(0, X.shape[1], size=2)
    # b) same electrode, different feature (uncomment lines below)
    # f0, f1 = np.random.randint(0, num_features, size=2)
    # e = np.random.randint(0, num_electrodes)
    # i0, i1 = f0*num_electrodes + e, f1*num_electrodes + e
    # b) same feature, different electrode (uncomment lines below)
    f = np.random.randint(0, num_features)
    e0, e1 = np.random.randint(0, num_electrodes, size=2)
    i0, i1 = f * num_electrodes + e0, f * num_electrodes + e1
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    colors = ['blue', 'red']
    # select features i0, i1 and separate by class
    X00, X01 = X[y == 0][:, i0], X[y == 1][:, i0]
    X10, X11 = X[y == 0][:, i1], X[y == 1][:, i1]
    # plot cumulative distribution of feature i0 separate for each class
    axes[0].hist(X00, bins=20, label='y=0, ' + feature_names[i0], density=True, alpha=0.5)
    axes[0].hist(X01, bins=20, label='y=1, ' + feature_names[i0], density=True, alpha=0.5)
    axes[0].hist(X10, bins=20, label='y=0, ' + feature_names[i1], density=True, alpha=0.5)
    axes[0].hist(X11, bins=20, label='y=1, ' + feature_names[i1], density=True, alpha=0.5)
    axes[0].set_title('histograms')
    axes[0].legend()

    axes[1].plot(np.sort(X00), np.linspace(0, 1, X00.shape[0]), label='y=0, ' + feature_names[i0], alpha=0.5)
    axes[1].plot(np.sort(X01), np.linspace(0, 1, X01.shape[0]), label='y=1, ' + feature_names[i0], alpha=0.5)
    axes[1].plot(np.sort(X10), np.linspace(0, 1, X10.shape[0]), label='y=0, ' + feature_names[i1], alpha=0.5)
    axes[1].plot(np.sort(X11), np.linspace(0, 1, X11.shape[0]), label='y=1, ' + feature_names[i1], alpha=0.5)
    axes[1].set_title('empirical cumulative distribution functions')
    axes[1].legend()

    axes[2].scatter(X00, X10, label='y=0')
    axes[2].scatter(X01, X11, label='y=1')
    axes[2].set_xlabel(feature_names[i0])
    axes[2].set_ylabel(feature_names[i1])
    axes[2].set_title('scatter plot')
    axes[2].legend()

    print("0000")
    plt.show()


# load data
# rows in X are subject major order, i.e. rows 0-9 are all samples from subject 0, rows 10-19 all samples from subject 1, etc. # columns in X are in feature_type major order, i.e. columns 0-47 are alpha band power, eyes closed, electrodes 0-48
# feature identifiers for all columns in X are stored in feature_names.csv
X = np.loadtxt('data.csv', delimiter=',')
y = np.loadtxt('labels.csv', delimiter=',')
with open('feature_names.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')
    feature_names = [row for row in csv_reader][0]

#
# Feature Selection with variance
data_with_variance = VarianceThreshold(threshold=3).fit_transform(X)
# Wrapper
# data_with_wrapper = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(X, y)
draw_figure(X, y, feature_names)




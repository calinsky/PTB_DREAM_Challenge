#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebecca Calinsky
rdc2164
"""

import numpy as np

# %%
def ROC( Labels, Confidences, resolution = 101 ):
    # Labels: Ground Truth
    # Confidences: value in [0,1]; confidence that a given sample is a "Positive"

    Positives = Labels == 1
    num_pos = Positives.sum()

    Negatives = Labels == 0
    num_neg = Negatives.sum()

    Thr = np.linspace(0, 1, resolution)
    num_thrs = len(Thr)
    XY = np.zeros((num_thrs, 2))

    for i, thr in enumerate(Thr):
        # Positive guesses based on threshold
        H = Confidences > thr

        tpr = H[Positives].sum() / float(num_pos)
        fpr = H[Negatives].sum() / float(num_neg)
        
        XY[i] = (fpr, tpr)

    return XY


# %%
def PRC( Labels, Confidences, resolution = 101 ):
    # Labels: Ground Truth
    # Confidences: value in [0,1]; confidence that a given sample is a "Positive"

    Positives = Labels == 1
    num_pos = Positives.sum()

    Thr = np.linspace(0, 1, resolution)
    num_thrs = len(Thr)
    XY = np.zeros((num_thrs, 2))

    for i, thr in enumerate(Thr):
        # Positive guesses based on threshold
        H = Confidences > thr

        tp = H[Positives].sum()
        hp = H.sum()
        
        precis = tp / float(hp)  if hp > 0 else  1
        recall = tp / float(num_pos)
        
        XY[i] = (recall, precis)

    return XY


# %%
def plot( Labels, Confidences, plot_type, resolution = 101 ):
    if plot_type == "ROC":
        XY = ROC(Labels, Confidences, resolution)
        title  = "ROC"
        xlabel = "False Pos. Rate"
        ylabel = "True Pos. Rate"
        loc    = "lower right"
    else:
        XY = PRC(Labels, Confidences, resolution)
        title  = "PRC"
        xlabel = "Recall"
        ylabel = "Precision"
        loc    = "lower left"
    
    (X, Y) = (XY[:,0], XY[:,1])
    D =  X[:-1] - X[1:]
    A = (Y[:-1] + Y[1:])/2
    auc = (D * A).sum()

    import matplotlib.pyplot as plt
    plt.plot(X, Y, "r-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([-.01, 1.01])
    plt.ylim([-.01, 1.01])
    plt.legend([f'AUC: {auc:.2%}'], loc = loc)
#    plt.axis("equal")
    plt.show()

    return auc


# %% plots for fake dataset
#N = 100
#Labels = (np.random.rand(N) < 0.25).astype(float)
#Confidences = np.random.rand(N) *0.8 + Labels *0.2
#
#auc = plot(Labels, Confidences, "ROC")
#auc = plot(Labels, Confidences, "PRC")

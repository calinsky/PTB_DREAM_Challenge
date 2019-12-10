#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebecca Calinsky
rdc2164
"""

import numpy as np

# %% Load Data

data_file = "SC2_Data/DATA.pl"

import torch
DATA = torch.load(data_file)

get_variable = lambda var_name:  DATA[var_name].astype(float)
DATA_tr   = get_variable("DATA_tr")
DATA_ts   = get_variable("DATA_ts")
Labels_tr = get_variable("Labels_tr")

print(DATA_tr.shape, Labels_tr.shape, DATA_ts.shape, None)

Ids_tr = Ids[Is_Train]


# %% Split into Train/Validate

# ignore one type
# PPROM = 1 or sPTD = 2

# PPROM (ignore labels == 2)
Inds  = (Labels_tr != 2)
XX_p  =   DATA_tr[Inds]
Y_p   = Labels_tr[Inds]
Ids_p =    Ids_tr[Inds]

# sPTD (ignore labels == 1)
Inds  = (Labels_tr != 1)
XX_s  =   DATA_tr[Inds]
Y_s   = Labels_tr[Inds]
Ids_s =    Ids_tr[Inds]
# set all non-zero labels to 1 (otherwise some may be 2; no good!)
Y_s[Y_s != 0] = 1

# validation ratio
params = { "validation_ratio": 0.2, "same_prior": True*0 }

# Use RANDOM splits
#from split_data import source_split as split; params = {}
#from split_data import random_split as split
from split_data import random_split_id as split
((XX_p_tr, Y_p_tr), (XX_p_va, Y_p_va)) = split(XX_p, Y_p, Ids_p, **params)
((XX_s_tr, Y_s_tr), (XX_s_va, Y_s_va)) = split(XX_s, Y_s, Ids_s, **params)

print("PPROM shapes and priors")
print(XX_p_tr.shape, Y_p_tr.shape, XX_p_va.shape, Y_p_va.shape)
print(Y_p.mean(), Y_p_tr.mean(), Y_p_va.mean())

print("sPTD shapes and priors")
print(XX_s_tr.shape, Y_s_tr.shape, XX_s_va.shape, Y_s_va.shape)
print(Y_s.mean(), Y_s_tr.mean(), Y_s_va.mean())


# %%

def train( XX, Y, Ids, resolution = 21 ):
    # get random data split
    ((XX_tr, Y_tr), (XX_va, Y_va)) = split(XX, Y, Ids, **params)

    # set sample weights by source (1 for GSM, 2 for non-GSM)
    S = (XX_tr[:,-1] != 0).astype(float) +1

    # normalize weights by class
    pos_weights = S[Y_tr == 1].sum() / S.sum()
    W = np.array([pos_weights, 1-pos_weights])
    S *= W[Y_tr.astype(int)]

    # train random forest
    from sklearn.ensemble import RandomForestClassifier
    RFC = RandomForestClassifier(n_estimators = 2000, max_depth = 1)
    RFC.fit(XX_tr, Y_tr, sample_weight = S)

    def get_auc( Y, P ):
        def xy_to_auc( XY ):
            D =  XY[:-1,0] - XY[1:,0]
            A = (XY[:-1,1] + XY[1:,1])/2
            auc = (D * A).sum()
            return auc

        from curves import ROC, PRC
        XY = ROC(Y, P, resolution)
        auc_roc = xy_to_auc(XY)

        XY = PRC(Y, P, resolution)
        auc_prc = xy_to_auc(XY)

        auc = (auc_roc + auc_prc)/2
        return auc

    # get validation auc
    P = RFC.predict_proba(XX_va)[:,1:]
    auc = get_auc(Y_va, P);

    return (RFC, auc)


# %% Select random features; collapsing number of features with each pass

# Total num features: 29461
max_feature = DATA_tr.shape[1]

Num_Features = np.round(np.hstack((
    np.exp(np.linspace(np.log(10000), np.log(1000), 5)),
    np.exp(np.linspace(np.log(700), np.log(100), 7)) ))).astype(int)
print(Num_Features)

# num validations
K = 10


# %% run feature selection loop

# start with all features
Features = np.arange(max_feature, dtype = int)
XX_p_f = XX_p
XX_s_f = XX_s

print()
for t, num_features in enumerate(Num_Features):
    # reset feature importances
    Importances = 0

    # for each of the K random restarts
    for k in range(K):

        print(f'{t +1:2d}/{len(Num_Features)}\t{k +1:2d}/{K}\tp', end = '')
        (RFC_p, auc_p) = train(XX_p_f, Y_p, Ids_p)
        Importances_p = RFC_p.feature_importances_
        print('s', end = '')
        (RFC_s, auc_s) = train(XX_s_f, Y_s, Ids_s)
        Importances_s = RFC_s.feature_importances_
        print(f'\tacc: {(auc_p + auc_s)/2:.2%}')

        # keep track of the counts (weighted by classifier accuracy)
        Importances += Importances_p *auc_p + Importances_s *auc_s

    print(f'\tPruning down to {num_features} features.')

    # get sorted indices; only keep last num_features
    Inds = np.argsort(Importances)[-num_features:]
    import matplotlib.pyplot as plt
    plt.plot(Importances[Inds])
    plt.show()
    Features = Features[Inds]
    print(Features[-10:])
    Features = np.sort(Features)
    # append source feature if not included
    if max_feature -1 not in Features:
        Features = np.append(Features, max_feature -1)

    # use feature subset
    XX_p_f = XX_p[:,Features]
    XX_s_f = XX_s[:,Features]


# %% Train final classifiers

RFC_p = [None] *K
Auc_p = np.zeros(K)

RFC_s = [None] *K
Auc_s = np.zeros(K)

if 0:
    # get random data split
    print("YES validation")
    ((XX_p_tr, Y_p_tr), (XX_p_va, Y_p_va)) = split(XX_p_f, Y_p, Ids_p, **params)
    ((XX_s_tr, Y_s_tr), (XX_s_va, Y_s_va)) = split(XX_s_f, Y_s, Ids_s, **params)
else:
    # use all data
    print("NO validation")
    (XX_p_tr, Y_p_tr) = (XX_p_f, Y_p)
    (XX_s_tr, Y_s_tr) = (XX_s_f, Y_s)


# train models
for k in range(K):
    print(f'{k +1:2d}/{K}\tp', end = '')
    (RFC_p[k], Auc_p[k]) = train(XX_p_tr, Y_p_tr, Ids_p, resolution = 101)
    print('s', end = '')
    (RFC_s[k], Auc_s[k]) = train(XX_s_tr, Y_s_tr, Ids_s, resolution = 101)
    print(f'\tauc: {(Auc_p[k] + Auc_s[k])/2:.2%}')


# %%
def get_model( RFC, Auc, T = 5 ):
    # use T best models
    KK = np.argsort(Auc)[-T:]
    RFC = [RFC[k]  for k in KK]
    Auc = Auc[KK]

    print(Auc.round(2))

    def model( XX ):
        S = 0
        s = 0
        for i, rfc in enumerate(RFC):
            # get prediction
            P = rfc.predict_proba(XX)[:,1:]
            
            # weight classifier output according to auc
            a = Auc[i]
            assert(a >= 0)
            S += P *a
            s += a
        S /= s
        return S

    return model

model_p = get_model(RFC_p, Auc_p)
model_s = get_model(RFC_s, Auc_s)


# %% Plot ROC AND PRC curves using validation data

def plots( P, Y, x = '' ):
    from curves import plot as plot
    auc_roc = plot(Y, P, "ROC", resolution = 101)
    auc_prc = plot(Y, P, "PRC", resolution = 101)
    print(f'{x}\t{(auc_roc + auc_prc)/2}')

P_p_va = model_p(XX_p_va)
plots(P_p_va, Y_p_va, 'p')

P_s_va = model_s(XX_s_va)
plots(P_s_va, Y_s_va, 's')

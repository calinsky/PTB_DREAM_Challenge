#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebecca Calinsky
rdc2164
"""

import pyreadr
import torch

# %%

# will have the following: odict_keys(['anoSC2', 'esetSC2'])
Data = pyreadr.read_r("SC2_Data/eset_SC2_v20.RData")
print(Data.keys()) # to check what objects there are


# %% extract pandas data frames
anoSC2_df = Data["anoSC2"] # extract the pandas dataframe for object anoSC2
print(anoSC2_df)

ANO = anoSC2_df.values
print(ANO.shape)
#torch.save(ANO, "ANO.pl")


# %%
esetSC2_df = Data["esetSC2"] # extract the pandas dataframe for object esetSC2
print(esetSC2_df)

# reshape eset data into 739 samples x 29459 features
DATA = esetSC2_df.values.reshape(739, -1)
print(DATA.shape)
#torch.save(DATA, "DATA.pl")


# %% convert features and format labels

import numpy as np

# individual ids
Ids = ANO[:,1]

# add gestational age as feature
GA = ANO[:,2].astype(float)

# add source as feature
Source = ANO[:,0]
First_Letter = np.array([s[0]  for s in Source])
X = np.zeros(Source.shape)
X[First_Letter == 'S'] = 1
X[First_Letter == 'T'] = 2

# append GA and source to data features
DATA_X = np.hstack((DATA, GA[:,None], X[:,None]))

# create labels: 0=control, 1=pProm, 2=sPTD
Group = ANO[:,4]
Y = np.zeros(Group.shape, dtype = int)
Y[Group == "PPROM"] = 1
Y[Group == "sPTD"]  = 2

# which dataset (train/test)
Is_Train = ANO[:,6].astype(bool)


# %% split data into train/test sets

DATA_tr = DATA_X[ Is_Train]
DATA_ts = DATA_X[~Is_Train]
Labels_tr = Y[Is_Train]
print(DATA_tr.shape, DATA_ts.shape, Labels_tr.shape)

torch.save({ "DATA_tr": DATA_tr,
             "DATA_ts": DATA_ts,
             "Labels_tr": Labels_tr }, "SC2_Data/DATA.pl")


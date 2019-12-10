#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebecca Calinsky
rdc2164
"""
# SPLIT DATASET INTO TRAINING AND VALIDATION

import numpy as np


# %%
def source_split( DATA, Labels ):
    # source was encoded as the last feature; test source is always 2
    Source = DATA[:,-1]
    Inds_va = Source == 2
    Inds_tr = ~Inds_va

    Sets = _split_sets(DATA, Labels, (Inds_tr, Inds_va))
    return Sets


# %%
def random_split( DATA, Labels, validation_ratio = 0.2, same_prior = False ):
    num_samples = len(Labels)

    if same_prior:
        Inds_va = np.empty(0, dtype = int)
        Inds_tr = np.empty(0, dtype = int)
        
        for y in np.unique(Labels):
            Inds_y = np.argwhere(Labels == y)
            num_valids = int(round(len(Inds_y) * validation_ratio))    

            Inds_y = np.random.permutation(Inds_y)
            Inds_va = np.append(Inds_va, Inds_y[:num_valids])
            Inds_tr = np.append(Inds_tr, Inds_y[num_valids:])

    else:
        Inds = np.random.permutation(num_samples)    
        num_valids = int(num_samples * validation_ratio)
        Inds_va = Inds[:num_valids]
        Inds_tr = Inds[num_valids:]

    Sets = _split_sets(DATA, Labels, (Inds_tr, Inds_va))
    return Sets


# %% Does Not Split Across Ids!
def random_split_id( DATA, Labels, Ids, validation_ratio = 0.2, same_prior = False ):
    num_samples = len(Labels)
    Inds = np.arange(num_samples)

    if same_prior:
        Inds_va = np.empty(0, dtype = int)
        Inds_tr = np.empty(0, dtype = int)
        
        for y in np.unique(Labels):
            Inds_y = np.argwhere(Labels == y)
            num_valids = len(Inds_y) * validation_ratio

            Ids_y = Ids[Inds_y]
            for i in np.random.permutation(np.unique(Ids_y)):
                # ensure all samples from same individual are in same set
                Inds_i = Inds_y[Ids_y == i]
                Inds_va = np.append(Inds_va, Inds_i)

                num_valids -= len(Inds_i)
                if num_valids <= 0:  break

    else:
        num_valids = num_samples * validation_ratio

        Inds_va = np.empty(0, dtype = int)
        for i in np.random.permutation(np.unique(Ids)):
            # ensure all samples from same individual are in same set
            Inds_i = Inds[Ids == i]
            Inds_va = np.append(Inds_va, Inds_i)

            num_valids -= len(Inds_i)
            if num_valids <= 0:  break

    Inds_tr = np.setdiff1d(Inds, Inds_va, assume_unique = True)

    Sets = _split_sets(DATA, Labels, (Inds_tr, Inds_va))
    return Sets


# %%
def _split_sets( DATA, Labels, INDS ):
    Sets = []
    
    for Inds_i in INDS:
        Set_i = (DATA[Inds_i], Labels[Inds_i])
        Sets.append(Set_i)

    return Sets

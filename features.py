#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebecca Calinsky
rdc2164
"""

import numpy as np

Importances = 0

for rfc in RFC_p:  Importances += rfc.feature_importances_
for rfc in RFC_s:  Importances += rfc.feature_importances_

# %%
T = 10
Inds = np.argsort(Importances)[-T:]

import matplotlib.pyplot as plt
TT = np.arange(T)
plt.bar(TT, Importances[Inds])
plt.xticks(TT, labels = [f'{Features[i]}_at'  for i in Inds], rotation = 90)
plt.yticks(ticks = [])
plt.xlabel("Entrez IDs")
plt.ylabel("Relative Importance")
plt.title(f'Top {T} Genes in Predicting PTB')
plt.show()


# %%
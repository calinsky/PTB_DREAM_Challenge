#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebecca Calinsky
rdc2164
"""

import torch


# %% Dataloader

def dataloader( XX, Y, Norms = None, device = "cpu" ):
  num_samples = Y.numel()

  XX = XX.to(device)
  Y =   Y.to(device)

  # get or compute norms
  if Norms is None:
    Mean  = XX.mean(dim = 0, keepdim = True)
    StdDev = XX.std(dim = 0, keepdim = True)
    Norms = (Mean, StdDev)
  else:  (Mean, StdDev) = Norms

  # normalize data
  XX = (XX -Mean)/StdDev

  def iterator( batch_size = 0 ):
    if batch_size <= 0 or batch_size >= num_samples:  yield (XX, Y)
    else:
      # get random ordering of indices
      I = torch.randperm(num_samples, device = device)
      XXI = XX[I]
      YI  =  Y[I]

      # yield batches
      for i in range(0, num_samples, batch_size):
        slc = slice(i, i +batch_size)
        yield (XXI[slc], YI[slc])

  return (iterator, num_samples, Norms)


# %% Define Model

import torch.nn as nn
import torch.nn.functional as F

num_inputs  = XX_tr.shape[1]
num_hidden1 = 128
num_hidden2 = 16

class NN( nn.Module ):

  def __init__( self ):
    # calls initialization method of the parent (Module) class
    super().__init__()

    # add an initial layer to choke number of used inputs
    self._choker = nn.Parameter(torch.rand(1, num_inputs))

    self._layer1 = nn.Linear(num_inputs,  num_hidden1)
    self._layer2 = nn.Linear(num_hidden1, 1)
    # self._layer2 = nn.Linear(num_hidden1, num_hidden2)
    # self._layer3 = nn.Linear(num_hidden2, 1)

    self._batchnorm1 = nn.BatchNorm1d(num_hidden1)

    self._dropout1 = nn.Dropout(0.95)
    self._dropout2 = nn.Dropout(0.6)

  def forward( self, XX ):
    XX = self._choker * XX

    XX = self._layer1(XX)
    # XX = self._dropout1(XX)
    XX = self._batchnorm1(XX)
    XX = F.relu(XX)

    XX = self._layer2(XX)
    # XX = self._dropout2(XX)
    # XX = F.relu(XX)

    # XX = self._layer3(XX)

    return XX

  def classify( self, XX ):
    with torch.no_grad():
      H = self(XX)
      Y_predicted = H > 0
      return Y_predicted


# %% NN Training Routine

device = "cuda"
# device = "cpu"

# prepare train data
(batch_iterator, num_samples, Norms) = dataloader(XX_tr, Y_tr, device = device)
pos_prior = Y_tr.sum().float() /num_samples
print(f'pos-prior: {pos_prior:.2%}')
criterion = nn.BCEWithLogitsLoss(pos_weight = pos_prior)

# prepare validation data
(val_iterator, num_val_samples, *_) = dataloader(XX_va, Y_va, device = device)

# instantiate NN
# torch.manual_seed(0)
model = NN().to(device)

import time
start_time = time.time()

# set up optimization parameters and loss criterion
lr = 0.01
# optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 0.3)
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.1, weight_decay = 0.01)

penalty_1 = 0.01
penalty_2 = 10

num_epochs = 1000

# prepare plot
import numpy as np
X       = np.arange(0, num_epochs) +1
Losses  = np.tile(np.NaN, num_epochs)
Tr_Errs = np.tile(np.NaN, num_epochs)
Va_Errs = np.tile(np.NaN, num_epochs)
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
fig = plt.figure()

# run training loop
for epoch in range(num_epochs):
  # keep track of loss & error
  running_loss = 0.
  running_error = 0.

  # iterate over minibatches
  for (XX, Y) in batch_iterator(batch_size = 16):
    # reset gradients
    optimizer.zero_grad()
    # run forward pass (H is the hypothesis)
    H = model(XX)
    # compute loss given criterion
    loss = criterion(H, Y)
    loss += model._choker.abs().mean() *penalty_1
    # loss += (model._layer1.weight**2).mean() *penalty_2
    # compute back-prop derivates
    loss.backward()
    # perform back-prop update
    optimizer.step()

    # keep track of running loss and accuracy
    error = ((H > 0) != Y).sum().cpu()
    running_error += error
    running_loss += loss *len(Y)

  # # truncate choker weights
  # with torch.no_grad():
  #   k = (num_inputs -90) //(epoch*2 +1) +90
  #   (vals, _) = model._choker.topk(k, dim = 1, sorted = False)
  #   model._choker -= vals.min()
  #   model._choker.relu_()
  #   model._choker /= model._choker.sum()

  # record running time, loss, and accuracy
  elapsed_time = time.time() - start_time
  avg_loss  = running_loss  /num_samples
  avg_error = running_error /num_samples
  Losses[epoch]  = avg_loss
  Tr_Errs[epoch] = avg_error

  # print(f'epoch {epoch:3d}\tloss = {avg_loss:.3f}\terror = {avg_error:.2%}')

  # update plot every T iterations
  T = 5
  if epoch % T == 0:
    # save model
    # torch.save(model, "/content/drive/My Drive/TTS/model.pl")

    # compute test error
    (XX, Y) = next(val_iterator())
    va_error = (model.classify(XX) != Y).sum().float() /Y.numel()
    Va_Errs[epoch-T:epoch] = va_error.cpu().numpy()

    plt.clf()
    plt.plot(X,  Losses, 'b:', label = f'       Loss: {avg_loss:.4f}')
    plt.plot(X, Tr_Errs, 'g-', label = f'Train Err: {avg_error:.2%}')
    plt.plot(X, Va_Errs, 'r-', label = f'Valid Err: {va_error:.2%}')
    plt.xlim(X[[0, -1]])
    plt.ylim([0, 0.6])
    plt.xlabel("Epoch")
    plt.ylabel("Error, Loss")
    title = f'Epoch {epoch +1} [{elapsed_time:.2f} s, {elapsed_time/(epoch +1):2.2f} s/epoch]'
    title += f'  #:{(model._choker > 0).sum()}'
    plt.title(title)
    plt.legend(loc = "upper right")
    display(fig)
    clear_output(wait = True)

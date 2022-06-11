#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Weight Averaging: https://arxiv.org/abs/1803.05407

Implementaton in Keras from user defined epochs assuming constant 
learning rate

Cyclic learning rate implementation in https://arxiv.org/abs/1803.05407 
not implemented

Created on July 4, 2018

@author: Krist Papadopoulos
"""

import keras

class SWA(keras.callbacks.Callback):
    
    """
    A Keras callback function for stochastic weight averaging
    with a constant learning rate.
    ...

    Attributes
    ----------
    swa_epoch : int
        the training epoch to start stochatic weight averaging
    filepath : str
        filepath to save model weights
    """
    
    def __init__(self, swa_epoch, filepath = None):
        super(SWA, self).__init__()
        self.swa_epoch = swa_epoch 
        self.filepath = filepath
        
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']

        if self.nb_epoch <= self.swa_epoch:
            raise ValueError('Training ends before stochastic weight averaging begins, swa_epoch ({}) has to be '
                             'smaller than training epochs ({})'.format(self.swa_epoch, self.nb_epoch))

        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i, layer in enumerate(self.model.layers):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i]) / ((epoch - self.swa_epoch)  + 1)

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        if self.filepath:
            self.model.save_weights(self.filepath)
            print('Final stochastic averaged weights saved to file.')

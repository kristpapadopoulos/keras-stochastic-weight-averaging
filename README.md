# Keras Callbacks

## User Defined Keras Callback Functions

1.  Stochastic Weight Averaging - https://arxiv.org/abs/1803.05407

swa.py contains an implementation for stochastic weight averaging (SWA) with a constant learning rate for a user defined amount of epochs.

Callback is instantiated with filename for saving the final weights of the model after SWA and the number of epochs to average.

e.g. total number of training epochs 150, SWA to start from epoch 140 to average last 10 epochs.

number_of_epochs = 150

swa = SWA(filename, 140)

model.fit(..., callbacks = [swa])

### Stochastic Weight Averaging with Keras callback function

Stochastic Weight Averaging following paper [Averaging Weights Leads to Wider Optima and Better Generalization
](https://arxiv.org/abs/1803.05407)

The file swa.py contains an implementation for stochastic weight averaging (SWA) with a constant learning rate for a user defined amount of epochs.

Callback is instantiated with filename for saving the final weights of the model after SWA and the number of epochs to average.

<b>Example</b>

The total number of training epochs 150, SWA to start from epoch 140 to average last 10 epochs.

```
from swa import SWA

# specify number of training epochs
number_of_epochs = 150

# specify the start epoch of stochastic weight averaging
swa = SWA(140, filepath = None)

# call SWA during model fitting
model.fit(..., callbacks = [swa])
```

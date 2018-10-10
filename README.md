# Recare Data challenge

There is code for two different ways to predict the likelihood that a patient is accepted by a provider:
1. Logistic regression
2. Neural network

## Getting Started

Just clone the repository

### Prerequisites

All code is written to be executed with python3
There are a few things needed to run the code:
Keras
Tensorflow
Pandas
numpy
sklearn
imblearn
statsmodels
seaborn

Everything else should come with python3.x



## Running the tests

Start by running logReg.py, 
```
python3 logReg.py
'''
this will produce some printed output as well as some plots in the
logRegOutput/ directory.
All steps and why they have been done are commented in the code.

Despite the non-satisfactory result from the logistic regression, the gained knowledge
about the input variables and their predictive power can be used to train a neural network more efficiently.
The first part of the code here is basically identical to the one from logReg.py until it comes to
training the neural network.
The code also contains many comments about the strategy.
Run:
```
python3 neuralNet.py
'''
Also here, code will produce output and plots in
neuralNetOutput/


# Recare Data challenge

There are two scripts different ways to predict the likelihood that a patient is accepted by a provider:
1. Logistic regression
2. Neural network

## Getting Started

Just clone the repository and run the scripts after installing the prerequisites (if not yet installed)

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
seaborn -> this is not urgent, only for producing one plot  

Everything else should come with python3.x


## Running the scripts

Start by running logReg.py, 
```
python3 logReg.py
```
this will produce some printed output as well as some plots in the
logRegOutput/ directory.
All steps and why they have been done are commented in the code.
The dataset that is being used for performance evaluation can be
chosen by changing the boolean "useOSforTesting" to True/False.


The gained knowledge about the input variables and their predictive power
can be used to train a neural network more efficiently.
The first part of the code here is basically identical to the one from logReg.py until it comes to
training the neural network.
This code also contains many comments about the strategy.
Run:
```
python3 neuralNet.py
```
Also here, code will produce printed output and plots in
neuralNetOutput/

By default, the code is configured to use pretrained weigths and not train the network again.
This can be changed in the first few lines of code by setting "trainModel" to True.
It will train by default for 50 epochs which takes only a couple of minutes.




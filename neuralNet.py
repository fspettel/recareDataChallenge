#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys
import json
import os
import random as rn
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import model.myNetwork as myNetwork
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#  Following block is to make results reproducible
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(1234)
tf.set_random_seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#  Set global configuration values
trainModel = False  # Train model yourself (set to true) or use pretrained weights
test_size = 0.33      # Fraction of data to be used for testing
epochs = 50

#  Load dataset into pandas dataframe
dataset = 'dataset_09.2018.json'
with open(dataset) as datafile:
    data = json.load(datafile)
df = pd.DataFrame(data)


print('\nTotal number of requests %d\n' % df.shape[0])
print('Available variables:')
print(df.columns.values.tolist())
 

#  Manipulate data for better classification/plotting
#  Get month in year, shows to be a good predictor
#  Set large outliers in distance to 30 km, improves prediction
#  Only interested in request status of 1, so set request status of 2 to 0 as well
#  because it doesn't matter whether the provider declined or didn't answer
for i in df.index:
    date = df.at[i, 'request_sent']
    dt = date[0:10]
    year, month, day = (int(x) for x in dt.split('-'))
    df.at[i, 'request_sent'] = month
    
    dist = df.at[i, 'distance_to_patient_in_km']
    if dist > 30:
        df.at[i, 'distance_to_patient_in_km'] = 30

    status = df.at[i, 'request_status']
    if status == 2:
        df.at[i, 'request_status'] = 0


#  Use previously gained knowledge about predictive power of variables to select only the best ones
data_vars = ['distance_to_patient_in_km', 'hospital_id', 'request_status', 'request_sent', 'solution', 'provider_id']
data_final = df[data_vars]

X = data_final.loc[:, data_final.columns != 'request_status']
y = data_final.loc[:, data_final.columns == 'request_status']

#  Our dataset is strongly imbalanced (~4000 accepted vs 57000 not-accepted)
#  Use oversampling to generate pseudodata
#  Take 20% of dataset as test-set, use the rest for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
columns = X_train.columns

X_test_orig = X_test
y_test_orig = y_test

#  Create oversampled dataset
os = SMOTE(random_state=1)
os_data_X_train, os_data_y_train = os.fit_sample(X_train, y_train.values.ravel())
os_data_X_train = pd.DataFrame(data=os_data_X_train, columns=columns)
os_data_y_train = pd.DataFrame(data=os_data_y_train, columns=['request_status'])

os_data_X_test, os_data_y_test = os.fit_sample(X_test, y_test.values.ravel())
os_data_X_test = pd.DataFrame(data=os_data_X_test, columns=columns)
os_data_y_test = pd.DataFrame(data=os_data_y_test, columns=['request_status'])

#  Plausibility check for oversampling
print("\nApply oversampling to get equal ratio of acceptance/non-acceptance:\n")
print("New length of our oversampled dataset is ", len(os_data_X_train))
print("Number of non-acceptance in oversampled dataset", len(os_data_y_train[os_data_y_train['request_status'] == 0]))
print("Number of acceptance in oversampled dataset", len(os_data_y_train[os_data_y_train['request_status'] == 1]))
print("Ratio of non-acceptance in oversampled data is ", len(os_data_y_train[os_data_y_train['request_status'] == 0])/len(os_data_X_train))
print("Ratio of acceptance in oversampled data is \n", len(os_data_y_train[os_data_y_train['request_status'] == 1])/len(os_data_X_train))

#  Make oversampled train and test vectors
X_train = os_data_X_train
y_train = os_data_y_train['request_status']
X_test = os_data_X_test
y_test = os_data_y_test['request_status']

#  Scale all X vectors to have standardized distribution
#  Makes training easier for network
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)
scaler.fit(X_test_orig)
X_test_orig = scaler.transform(X_test_orig)


#  Prepare training and test dataset
#  Save weights
callbacks = [ModelCheckpoint(monitor='loss',
                             filepath='weights/newWeights.hdf5',
                             save_best_only=False,  
                             save_weights_only=True)]

#  Train model from scratch or load pre-trained model
print('\nThe neural network model has the following architecture:\n')
if(trainModel is True):
    model = myNetwork.getModel()
    model.summary()
    model.save('myModel.h5')
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks = callbacks, epochs=epochs, batch_size=20)
else:
    model = load_model('myModel.h5')
    model.summary()
    model.load_weights(('weights/pretrainedWeights.hdf5'))

#  evaluate the model on test dataset (20% of the original data)
scores = model.evaluate(X_test_orig, y_test_orig, verbose = 0)
print("\nThe classification accuracy on the original non-oversampled test set is:  %.2f%%" % (scores[1]*100))
scores = model.evaluate(X_test, y_test, verbose = 0)
print("\nThe classification accuracy on the oversampled test set is:  %.2f%%" % (scores[1]*100))
sys.exit
#  visualizing training accuracies and losses
if(trainModel is True):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    xc = range(epochs)
    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.style.use(['classic'])
    plt.savefig('neuralNetOutput/compareLoss')

    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(epochs)
    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('acc')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    plt.style.use(['classic'])
    plt.savefig('neuralNetOutput/compareAcc')


#  Plot predicted probability distribution for true accepted/non-accepted requests
y_pred = model.predict(X_test)
y_test = y_test.values
probDecline = np.empty(0)
probAccept = np.empty(0)
for i in range(len(y_test)):
    trueStatus = y_test[i]
    if(trueStatus == 0):
        probDecline = np.append(probDecline, y_pred[i])
    else:
        probAccept = np.append(probAccept, y_pred[i])
weightsDec = np.full_like(probDecline, 1/len(probDecline))
weightsAcc = np.full_like(probAccept, 1/len(probAccept))
plt.figure(3, figsize=(7, 5))
plt.hist(probDecline, bins=50, range=[-0.1, 1.1], weights=weightsDec, histtype='step')
plt.hist(probAccept, bins=50, range=[-0.1, 1.1], weights=weightsAcc, histtype='step')
plt.legend(['True declined', 'True accepted'], loc=2)
plt.xlabel('Predicted likelihood for acceptance')
plt.ylabel('Relative Frequency')
plt.title('Predicted likelihood for true Status of 0/1')
plt.style.use(['classic'])
plt.grid(True)
plt.savefig('neuralNetOutput/compareProbDist')

#  Create ROC curve (best indicator usually)
nn_score = roc_auc_score(y_test, model.predict(X_test))
print('\nArea under ROC curve: %0.2f' % nn_score)
fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
plt.figure()
plt.plot(fpr, tpr,  label='Neural Network (area = %0.2f)' % nn_score)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Wrongly classifed as accepted rate')
plt.ylabel('Correctly classifed as accepted rate')
plt.title('Receiver operator characteristic')
plt.legend(loc=4)
plt.grid(True)
plt.savefig('neuralNetOutput/ROC_Curve')

print('\nOutput plots (on oversampled test dataset) have been saved in ./neuralNetOutput')


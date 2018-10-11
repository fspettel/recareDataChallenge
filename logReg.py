#!/usr/bin/env python
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.api as sm

sns.set(style='white')
sns.set(style='whitegrid')

# Load json file into pandas dataframe
dataset = 'dataset_09.2018.json'
with open(dataset) as datafile:
    data = json.load(datafile)
df = pd.DataFrame(data)

# Print out some overview of the data
print('\nTotal number of requests %d' % df.shape[0])
print('Available variables:')
print(df.columns.values.tolist())
print()

# Round distance to integer numbers to be able to plot better
df = df.round({'distance_to_patient_in_km':0})

# Manipulate data for better classification/plotting
# Get month in year, shows to be a good predictor
# Set large outliers in distance to 30 km, improves prediction
# Only interested in request status of 1, so set req-status of 2 to 0 as well
# because it doesn't matter whether the provider declined or didn't answer

for i in df.index:
    date = df.at[i, 'request_sent']     
    dt = date[0:10]
    year, month, day = (int(x) for x in dt.split('-'))  
    df.at[i, 'request_sent'] = month
    
    dist = df.at[i, 'distance_to_patient_in_km']     
    if dist >30:
        df.at[i, 'distance_to_patient_in_km'] = 30
    
    status = df.at[i, 'request_status']     
    if status == 2:
        df.at[i, 'request_status'] = 0
        
# Have a quick  look at the data now
print('\nThis is the data for the first five requests:')
print(df.head(5))

# Get a better feeling for the data
print('\nThere are %d different patients in %d different hospitals' % (len(df['patient_id'].unique()),len(df['hospital_id'].unique())))
print('There are %d different providers' % len(df['provider_id'].unique()))
print('Patients are from %d different zipcodes\n' % len(df['zipcode'].unique()))


# Get percentage of acceptance and non-acceptance and print it
num_noAccept = len(df[df['request_status']!=1])
num_Accept = len(df[df['request_status']==1])
print('Out of %d requests, %d were accepted and %d were not accepted' % (df.shape[0],num_Accept,num_noAccept))

num_noAccept = len(df[df['request_status']!=1])
num_Accept = len(df[df['request_status']==1])
pct_of_no_Accept = 100*num_noAccept/(num_noAccept+num_Accept)

print('Overall percentage of nonacceptance is %0.2f' % pct_of_no_Accept)
pct_of_Accept = 100*num_Accept/(num_noAccept+num_Accept)
print('Overall percentage of acceptance is %0.2f\n' % pct_of_Accept)


# Plot percentage and save plot
sns.countplot(x='request_status', data = df)
plt.savefig('logRegOutput/req_stat')

# Plot distributions of potential input variables for 
# request status = 1 and 0 separately 
# in order to identify variables with a high predictive value
figsize = (6,4.4)
pd.crosstab(df.distance_to_patient_in_km,df.request_status, normalize ='columns').plot(kind='bar', figsize=figsize)
plt.title('Request Status for Distance')
plt.xlabel('distance')
plt.ylabel('Request Status')
plt.savefig('logRegOutput/distance')

pd.crosstab(df.hospital_id,df.request_status, normalize ='columns').plot(kind='bar', figsize=figsize)
plt.title('Request Status for hospital_id')
plt.xlabel('hospital_id')
plt.ylabel('Request Status')
plt.savefig('logRegOutput/hospital_id')

pd.crosstab(df.request_sent,df.request_status, normalize ='columns').plot(kind='bar', figsize=figsize)
plt.title('Request Status for request_sent month')
plt.xlabel('request_sent month')
plt.ylabel('Request Status')
plt.savefig('logRegOutput/request_sent month')

pd.crosstab(df.solution,df.request_status, normalize ='columns').plot(kind='bar', figsize=figsize)
plt.title('Request Status for solution')
plt.xlabel('solution')
plt.ylabel('Request Status')
plt.savefig('logRegOutput/solution')

# Takes long to run and doesn't give a good visible output because of large provider number
#pd.crosstab(df.provider_id,df.request_status, normalize ='columns').plot(kind='bar', figsize=figsize)
#plt.title('Request Status for provider_id')
#plt.xlabel('provider_id')
#plt.ylabel('Request Status')
#plt.savefig('logRegOutput/provider_id')


# Use only variables from which it is likely that they have predictive value (drop id and patient creation for example)
data_vars=['zipcode','distance_to_patient_in_km', 'hospital_id', 'provider_id', 'request_status' ,'request_sent', 'solution']
data_final=df[data_vars]

X = data_final.loc[:, data_final.columns != 'request_status']
y = data_final.loc[:, data_final.columns == 'request_status']

# Our dataset is strongly imbalanced (~4000 vs 57000)
# Use oversampling to generate pseudodata
# Take one third of dataset as test-set, use the rest for fitting the model
oversample = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
columns = X_train.columns

# Oversample train and test set but keep them strictly separate
os_data_X_train,os_data_y_train=oversample.fit_sample(X_train, y_train.values.ravel())
os_data_X_train = pd.DataFrame(data=os_data_X_train,columns=columns )
os_data_y_train= pd.DataFrame(data=os_data_y_train,columns=['request_status'])

os_data_X_test, os_data_y_test = oversample.fit_sample(X_test, y_test.values.ravel())
os_data_X_test = pd.DataFrame(data=os_data_X_test, columns=columns)
os_data_y_test = pd.DataFrame(data=os_data_y_test, columns=['request_status'])

# Plausibility check for oversampling in train set
print('Use oversampling to get equal size of acceptance/non-acceptance dataset:\n')
print('New length of our oversampled dataset is ',len(os_data_X_train))
print('Number of non-acceptance in oversampled dataset',len(os_data_y_train[os_data_y_train['request_status']==0]))
print('Number of acceptance in oversampled dataset',len(os_data_y_train[os_data_y_train['request_status']==1]))
print('Ratio of non-acceptance in oversampled data is ',len(os_data_y_train[os_data_y_train['request_status']==0])/len(os_data_X_train))
print('Ratio of acceptance in oversampled data is ',len(os_data_y_train[os_data_y_train['request_status']==1])/len(os_data_X_train))


# Use recursive reature elimination to simplify model
data_final_vars=data_final.columns.values.tolist()
y=['request_status']
X=[i for i in data_final_vars if i not in y]
logreg = LogisticRegression(solver = 'lbfgs', max_iter=100)
# Take maximally 5 variables as input
rfe = RFE(logreg, 5)
rfe = rfe.fit(os_data_X_train, os_data_y_train.values.ravel())
print('\nVariables to be considered:')
print(X)
print('\nShould variable be used according to recursive elimination technique:')
print(rfe.support_)
print('-> Drop zipcode')
# Based on this logRegOutput, drop zipcode

# Final variables for regression
final_Vars=['distance_to_patient_in_km', 'hospital_id', 'provider_id','request_sent', 'solution']
X_train=os_data_X_train[final_Vars]
y_train=os_data_y_train['request_status'] 

# Use oversampled/non-oversampled data for testing
useOSforTesting = True
if useOSforTesting:
    X_test=os_data_X_test[final_Vars]
    y_test=os_data_y_test['request_status'] 

# Use original test data (non-oversampling)
else:
    X_test=X_test[final_Vars]
    y_test=y_test['request_status'] 


# Define model
logit_model=sm.Logit(y_train, X_train)
result=logit_model.fit()
print(result.summary2())
 
# Train model
logreg = LogisticRegression(solver = 'lbfgs',max_iter=200, verbose = 0)
logreg.fit(X_train, y_train)
 
# Get predicted probability for test set
y_pred = logreg.predict_proba(X_test)
y_pred_round = logreg.predict(X_test)

 
# Create output plots to visualize prediction performance 
# Plot predicted probability distribution for true accepted/non-accepted requests
y_test = y_test.values
probDecline = np.empty(0)
probAccept  = np.empty(0)
for i in range(len(y_test)):
    trueStatus = y_test[i] 
    if(trueStatus == 0):     
        probDecline = np.append(probDecline, y_pred[i,0])
    else:
        probAccept = np.append(probAccept, y_pred[i,1])        
weightsDec = np.full_like(probDecline, 1/len(probDecline))
weightsAcc = np.full_like(probAccept, 1/len(probAccept))
plt.figure(8, figsize=(7, 5))    #check number
plt.hist(probDecline, bins=40, range=[0.2, 0.8], weights = weightsDec, histtype='step')    
plt.hist(probAccept, bins=40, range=[0.2, 0.8], weights = weightsAcc, histtype='step')
plt.legend(['True declined', 'True accepted'], loc = 2)
plt.xlabel('Predicted likelihood for acceptance')
plt.ylabel('Relative frequency')
plt.title('Predicted likelihood for true Status of 0/1')
plt.savefig('logRegOutput/compareProbDist')
print('Classification accuracy on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# Plot ROC curve, usually best performance indicator
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Wrongly classifed as accepted rate')
plt.ylabel('Correctly classifed as accepted rate')
plt.title('Receiver operator characteristic')
plt.legend(loc=4)
plt.savefig('logRegOutput/Log_ROC')
print('Area under ROC curve: %0.2f' % logit_roc_auc) 
print('Plots have been saved in ./logRegOutput') 

 


#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
import pandas as pd
from collections import Counter
import pprint

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments',
                'loan_advances', 'bonus', 'restricted_stock_deferred',
                'deferred_income', 'total_stock_value', 'expenses',
                'exercised_stock_options', 'other',
                'long_term_incentive', 'restricted_stock', 'director_fees',
                'from_poi_to_this_person', 'from_this_person_to_poi',
                'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

if len(data_dict) > 0:
    print 'data loaded!'
    print 'Number of initial data points: ', len(data_dict)
    print 'Number of initial features used: ', len(features_list)
    print '=========================='

# get number of POI in data
poi_count = []
for key in data_dict.keys():
    poi_count.append(data_dict[key]['poi'])

print 'Number of POI: ', poi_count.count(1)
print 'Number of non-POI: ', poi_count.count(0), '\n'

print 'list of all people in the dataset: '

print data_dict.keys()

print '=========================='

# get number of missing values per person
missing_feature = [0 for i in range(0, len(features_list))]
features_in_data = data_dict.values()

for loc_p, person in enumerate(features_in_data):
    for loc_f, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            missing_feature[loc_f] += 1

print 'Number of missing from each feature: '
for feature, num_missing in zip(features_list, missing_feature):
    print feature, ' -- ', num_missing


print '=========================='

### Task 2: Remove outliers
##following outlier cleaner adapted
## from outliers mini-project

# remove total
print 'Now removing outliers . . . '
print 'Removing obs named "Total" from data b/c it''s not a real data point'
data_dict.pop('TOTAL')
print 'Number of data points after removal: ', len(data_dict)

# TODO: fix outlier cleaner (below)

'''
def outlierCleaner(predictions, feature_1, feature_2, pct):
    import math as math
    n_cleaned = int(math.floor( (1-pct)*len(predictions)))
    data = []
    for prediction, feature_1, net_worth in zip(predictions, feature_1, feature_2):
        error = prediction - net_worth
        t = (feature_1, net_worth, error)
        data.append(t)
    data = sorted(data, key = lambda x: x[2], reverse=False)
    cleaned_data = data[0:n_cleaned]
    return cleaned_data

## go through labels and remove outliers
pct_outliers_to_remove = 0.05

for i in range(0, len(features_list)):
    for j in range(0, len(features_list)):
        X_train, X_test, y_train, y_test = train_test_split(data_dict[i], data_dict[j], test_size=0.1, random_state=42)
        reg = linear_model.LinearRegression()
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        _, temp_data_1, temp_data_2 = outlierCleaner( pred, data_dict[i], data_dict[j], )
'''

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

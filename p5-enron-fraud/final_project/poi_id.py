#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


import numpy as np
import pandas as pd
from collections import Counter
import pprint
from IPython.display import display

# ref: https://github.com/Corvids/ud120-projects (includes all module 5 starter code)
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
pp = pprint.PrettyPrinter(indent=4)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

if len(data_dict) > 0:
    print 'data loaded!'
    print 'Number of initial data points: ', len(data_dict)
    print 'Number of initial features used: ', len(features_list)

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

### Task 3: Create new feature(s)
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn import linear_model
from sklearn.cross_validation import train_test_split

### Store to my_dataset for easy export below.
### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

for key in my_dataset:
    #messages to POI
    if my_dataset[key]['from_messages'] != 'NaN':
        my_dataset[key]['to_poi_message_ratio'] = \
                1.0*my_dataset[key]['from_this_person_to_poi']/my_dataset[key]['from_messages']
    else:
        my_dataset[key]['to_poi_message_ratio'] = 'NaN'
    #messages from POI
    if my_dataset[key]['to_messages'] != 'NaN':
        my_dataset[key]['from_poi_message_ratio'] = \
                1.0*my_dataset[key]['from_poi_to_this_person']/my_dataset[key]['to_messages']
    else:
        my_dataset[key]['from_poi_message_ratio'] = 'NaN'


# update the feature list
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'to_poi_message_ratio', 'from_poi_message_ratio']

pp.pprint(my_dataset["SKILLING JEFFREY K"])

if len(data_dict) > 0:
    print 'After adding two new features: to_poi_message_ratio, from_poi_message_ratio'
    print 'Number of initial data points: ', len(my_dataset)
    print 'Number of initial features used: ', len(features_list)

# scaling features

def scaleFeatures(my_dataset, feature):
    min_scale = np.inf
    max_scale = -np.inf

    for key in my_dataset:
        # get min, max
        if my_dataset[key][feature] == 'NaN':
            pass
        else:
            if my_dataset[key][feature] < min_scale:
                min_scale = my_dataset[key][feature]
            if my_dataset[key][feature] > max_scale:
                max_scale =  my_dataset[key][feature]

    print 'min ' + str(feature) + ' is: ' + str(min_scale)
    print 'max ' + str(feature) + ' is: ' + str(max_scale)

    for key in my_dataset:
        if my_dataset[key][feature] == 'NaN':
            pass
        else:
            my_dataset[key][feature] = 1.0*(my_dataset[key][feature] - min_scale) / (max_scale - min_scale)

    print 'scaled feature ' + str(feature) + '!'
    print '=========='

features_to_scale = ['bonus', 'salary', 'restricted_stock',
                     'long_term_incentive', 'deferral_payments',
                     'deferred_income', 'director_fees',
                     'loan_advances', 'other',
                     'restricted_stock_deferred', 'total_payments',
                     'total_stock_value', 'exercised_stock_options',
                     'expenses']

for feature in features_to_scale:
    scaleFeatures(my_dataset, feature)

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier(random_state = 42)
clf = clf.fit(features_train, labels_train)

features_importance = zip(map(lambda x: round(x, 6), clf.feature_importances_), features_list)
features_importance.sort(key = lambda t: t[0], reverse = True)

print 'feature importances: ', features_importance

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from time import time
from tester import test_classifier

clf_GNB = GaussianNB()

t0_GNB = time()
clf_GNB.fit(features_train, labels_train)
print "training time:", round(time()-t0_GNB, 3), "s"

t1_GNB = time()
pred_GNB = clf_GNB.predict(features_test)
print "predicting time:", round(time()-t1_GNB, 3), "s"

accuracy_GNB = accuracy_score(labels_test, pred_GNB)
print 'accuracy of Naive Bayes: ' + str(accuracy_GNB)
print '\n'
test_classifier(clf_GNB, my_dataset, features_list, folds = 1000)

# Decision Tree

from sklearn import tree

clf_tree = tree.DecisionTreeClassifier(random_state=42)

t0_tree = time()
clf_tree.fit(features_train, labels_train)
print "training time:", round(time()-t0_tree, 3), "s"

t1_tree = time()
pred_tree = clf_tree.predict(features_test)
print "predicting time:", round(time()-t1_tree, 3), "s"

accuracy_tree = accuracy_score(labels_test, pred_tree)
print 'accuracy of Decision Tree: ' + str(accuracy_tree)
print '\n'
test_classifier(clf_tree, my_dataset, features_list, folds = 1000)

# Ada Boost
from sklearn.ensemble import AdaBoostClassifier

clf_Ada = AdaBoostClassifier(random_state=42)

t0_Ada = time()
clf_Ada.fit(features_train, labels_train)
print "training time:", round(time()-t0_Ada, 3), "s"

t1_Ada = time()
pred_Ada = clf_Ada.predict(features_test)
print "predicting time:", round(time()-t1_Ada, 3), "s"

accuracy_Ada = accuracy_score(labels_test, pred_Ada)
print 'accuracy of Ada Boost: ' + str(accuracy_Ada)
print '\n'
test_classifier(clf_Ada, my_dataset, features_list, folds = 1000)



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from tester import test_classifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import grid_search
from sklearn.metrics import fbeta_score, make_scorer

def fit_model(X, y):
    cv_sets = StratifiedShuffleSplit(y, n_iter = 10, test_size = 0.333, random_state = 42)

    regressor = AdaBoostClassifier(random_state=42)

    parameters = {'n_estimators':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
              'learning_rate':[0.01, 0.1, 0.5, 0.75, 1.0],
              'algorithm':['SAMME','SAMME.R']}

    scoring_fnc = make_scorer(fbeta_score, beta=2)

    grid = grid_search.GridSearchCV(regressor, parameters, scoring=scoring_fnc, cv=cv_sets)

    grid = grid.fit(X, y)

    return grid.best_estimator_


t0 = time()
clf = fit_model(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
print clf

t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"

accuracy = accuracy_score(labels_test, pred)
print 'accuracy of tuned Ada Boost model: ' + str(accuracy)



# tuned
print 'tuned model:'
t3 = time()
test_classifier(clf, my_dataset, features_list, folds = 1000)
print round(time()-t3, 3), "s"


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

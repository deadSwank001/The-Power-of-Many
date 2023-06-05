# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 02:01:21 2023

@author: swank
"""

#Creating classification and regression trees

from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
features = iris.feature_names
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=5, 
                        shuffle=True,
                        random_state=1)

import numpy as np
from sklearn import tree

for depth in range(1,10):
    tree_classifier = tree.DecisionTreeClassifier(
        max_depth=depth, random_state=0)
    if tree_classifier.fit(X,y).tree_.max_depth < depth:
        break
    score = np.mean(cross_val_score(tree_classifier, 
                                    X, y, 
                                    scoring='accuracy', 
                                    cv=crossvalidation))
    print('Depth: %i Accuracy: %.3f' % (depth,score))
tree_classifier = tree.DecisionTreeClassifier(
    min_samples_split=30, min_samples_leaf=10, 
    random_state=0)
tree_classifier.fit(X,y)
score = np.mean(cross_val_score(tree_classifier, X, y, 
                                scoring='accuracy', 
                                cv=crossvalidation))
print('Accuracy: %.3f' % score)

from sklearn.datasets import load_boston

# NOT AGAIN

boston = load_boston()
X, y = boston.data, boston.target
features = boston.feature_names
​
from sklearn.tree import DecisionTreeRegressor
regression_tree = tree.DecisionTreeRegressor(
    min_samples_split=30, min_samples_leaf=10, 
    random_state=0)
regression_tree.fit(X,y)
score = np.mean(cross_val_score(regression_tree, 
                   X, y, 
                   scoring='neg_mean_squared_error', 
                   cv=crossvalidation))
print('Mean squared error: %.3f' % abs(score))


#Getting Lost in a Random Forest

#Making machine learning accessible to all

from sklearn.datasets import load_digits
digit = load_digits()
X, y = digit.data, digit.target

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

tree_classifier = DecisionTreeClassifier(random_state=0)
crossvalidation = KFold(n_splits=5, shuffle=True, 
                        random_state=1)
bagging = BaggingClassifier(tree_classifier, 
                            max_samples=0.7, 
                            max_features=0.7, 
                            n_estimators=300)
scores = np.mean(cross_val_score(bagging, X, y, 
                                 scoring='accuracy', 
                                 cv=crossvalidation))
print ('Accuracy: %.3f' % scores)

#Working with a random forest classifier
X, y = digit.data, digit.target
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=5, shuffle=True, 
                        random_state=1)
RF_cls = RandomForestClassifier(n_estimators=300,
                               random_state=1)
score = np.mean(cross_val_score(RF_cls, X, y, 
                                scoring='accuracy', 
                                cv=crossvalidation))
print('Accuracy: %.3f' % score)
from sklearn.model_selection import validation_curve
param_range = [10, 50, 100, 200, 300, 500, 800, 1000, 1500]
crossvalidation = KFold(n_splits=3, 
                        shuffle=True, 
                        random_state=1)
RF_cls = RandomForestClassifier(n_estimators=300,
                               random_state=0)
train_scores, test_scores = validation_curve(RF_cls, X, y,
                                  'n_estimators', 
                                  param_range=param_range, 
                                  cv=crossvalidation, 
                                  scoring='accuracy')
mean_test_scores = np.mean(test_scores, axis=1)

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(param_range, mean_test_scores, 
         'bD-.', label='CV score')
plt.grid()
plt.xlabel('Number of estimators')
plt.ylabel('accuracy')
plt.legend(loc='lower right', numpoints= 1)
plt.show()


#Working with a random forest regressor
X, y = boston.data, boston.target
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
RF_rg = RandomForestRegressor (n_estimators=300, 
                               random_state=1)
crossvalidation = KFold(n_splits=5, shuffle=True, 
                        random_state=1)
score = np.mean(cross_val_score(RF_rg, X, y, 
                    scoring='neg_mean_squared_error', 
                    cv=crossvalidation))
print('Mean squared error: %.3f' % abs(score))


#Optimizing a random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
X, y = digit.data, digit.target
crossvalidation = KFold(n_splits=5, shuffle=True, 
                        random_state=1)
RF_cls = RandomForestClassifier(random_state=1)
scorer = 'accuracy'

from sklearn.model_selection import GridSearchCV
max_features = [X.shape[1]//3, 'sqrt', 'log2', 'auto']
min_samples_leaf = [1, 10, 30]
n_estimators = [50, 100, 300]
search_grid =  {'n_estimators':n_estimators,
                'max_features': max_features, 
                'min_samples_leaf': min_samples_leaf}
search_func = GridSearchCV(estimator=RF_cls, 
                           param_grid=search_grid, 
                           scoring=scorer, 
                           cv=crossvalidation)
search_func.fit(X, y)

best_params = search_func.best_params_
best_score = search_func.best_score_

print('Best parameters: %s' % best_params)
print('Best accuracy: %.3f' % best_score)


#Boosting predictions

X, y = digit.data, digit.target

#Knowing that many weak predictors win
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
ada = AdaBoostClassifier(n_estimators=1000, 
                         learning_rate=0.01, 
                         random_state=1)
crossvalidation = KFold(n_splits=5, shuffle=True, 
                        random_state=1)
score = np.mean(cross_val_score(ada, X, y, 
                                scoring='accuracy', 
                                cv=crossvalidation))
print('Accuracy: %.3f' % score)

### THIS WOULD HAVE BEEN NICE 6 bCHAPTERS AGO
###\\//###
#Creating a gradient boosting classifier

X, y = digit.data, digit.target
crossvalidation = KFold(n_splits=5, 
                        shuffle=True, 
                        random_state=1)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

GBC = GradientBoostingClassifier(n_estimators=300, 
                                 subsample=1.0, 
                                 max_depth=2, 
                                 learning_rate=0.1, 
                                 random_state=1)
score = np.mean(cross_val_score(GBC, X, y, 
                                scoring='accuracy', 
                                cv=crossvalidation))
print('Accuracy: %.3f' % score)


#Creating a gradient boosting regressor
X, y = boston.data, boston.target
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
GBR = GradientBoostingRegressor(n_estimators=1000, 
                                subsample=1.0, 
                                max_depth=3, 
                                learning_rate=0.01, 
                                random_state=1)
crossvalidation = KFold(n_splits=5, 
                        shuffle=True, 
                        random_state=1)
score = np.mean(cross_val_score(GBR, X, y, 
                                scoring='neg_mean_squared_error', 
                                cv=crossvalidation))
print('Mean squared error: %.3f' % abs(score))


#Using GBM hyper-parameters
X, y = boston.data, boston.target
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=5, shuffle=True, 
                        random_state=1)
GBR = GradientBoostingRegressor(n_estimators=1000, 
                                subsample=1.0, 
                                max_depth=3, 
                                learning_rate=0.01, 
                                random_state=1)

from sklearn.model_selection import GridSearchCV
subsample = [1.0, 0.9]
max_depth = [2, 3, 5]
n_estimators = [500 , 1000, 2000]
search_grid =  {'subsample': subsample, 
                'max_depth': max_depth, 
                'n_estimators': n_estimators}
search_func = GridSearchCV(estimator=GBR, 
             param_grid=search_grid, 
             scoring='neg_mean_squared_error',
             cv=crossvalidation)
search_func.fit(X,y)
​
best_params = search_func.best_params_
best_score = abs(search_func.best_score_)
print('Best parameters: %s' % best_params)
print('Best mean squared error: %.3f' % best_score)
​
​
#No Errors
### This marks the end of :
​
# Python for Data Science for Dummies
​
​
### * Not actually for real dummies ;) * ###
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:26:56 2020

@author: incspn
"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from preprocessing_methods import probas_categorical_prss

dataset = pd.read_csv('train.csv')
#dataset = dataset.dropna()
dataset = probas_categorical_prss(dataset)

del dataset['Name']
del dataset['Ticket']
del dataset['PassengerId']

print(dataset.shape)

# dataset.Pclass = dataset.Pclass.astype('str')

# dfDummy = pd.get_dummies(dataset)

# # Dummy variable trap
# del dfDummy['Pclass_3']
# del dfDummy['Sex_male']
# del dfDummy['Cabin_T']
# del dfDummy['Embarked_S']

# train_size = 0.7
# splitPoint = np.round(len(dfDummy) * train_size).astype(int)

# dataset = dfDummy.values

# x_train = dataset[:splitPoint,1:]
# y_train = dataset[:splitPoint,0]

# x_test = dataset[splitPoint:,1:]
# y_test = dataset[splitPoint:,0]

# del splitPoint
# del train_size

# print("Train X:", x_train.shape)
# print("Train Y:", y_train.shape)

# print("Test X:", x_test.shape)
# print("Test Y:", y_test.shape, '\n')

# #regression = LinearRegression()
# #regression.fit(x_train, y_train)

# #y_pred = np.round(regression.predict(x_test))

# #def removeExplosions(x):
# #    if x > 1.0:
# #        return 1
# #    elif x < 0.0:
# #        return 0
# #    return x

# #y_pred = np.array(list(map(removeExplosions, y_pred)))

# #second regressor
# def evaluate(y_pred, y_real):
#     return np.sum(y_pred == y_real)/len(y_real)

# #print('sklearn:', evaluate(y_pred, y_test), '%')

# # Ordinary least squares model
# print(y_train.shape, x_train.shape)
# regression_OLS = sm.OLS(y_train, x_train).fit()
# #np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# y_sm = regression_OLS.predict(x_test)
# print('statsmodels:', evaluate(y_sm.round(), y_test), '%')

# # x4 = 0.97
# # const? = 0.991
# # x19 = 0.985
# # x28            0.0048      0.409      0.012      0.991 
# # x33         1.247e-16   1.81e-14      0.007      0.995
# # x53           -0.0135      0.402     -0.033      0.973
# # x57           -0.0220      0.402     -0.055      0.956
# # x64            0.0365      0.404      0.090      0.928
# # x81            0.0239      0.408      0.059      0.953
# # x118           0.0508      0.404      0.126      0.900
# # x125           0.0297      0.404      0.074      0.941
# # x128       -2.254e-17   2.16e-16     -0.104      0.917
# # x129          -0.0131      0.402     -0.032      0.974

# pLimit = 0.05
# keys = dfDummy.keys()[1:]
# x_opt = x_train.copy()
# x_test_opt = x_test.copy()

# print('Elimination started . . .\n')

# while max(regression_OLS.pvalues) > pLimit:
#     index = np.argmax(regression_OLS.pvalues)
#     print('{} descarted with p-value of {}'.format(
#         keys[index],
#         max(regression_OLS.pvalues)
#         ))
#     x_opt = np.delete(x_opt, index, 1)
#     x_test_opt = np.delete(x_test_opt , index, 1)
    
#     keys = np.delete(keys, index)
#     regression_OLS = sm.OLS(y_train, x_opt).fit()
    
# y_sm = regression_OLS.predict(x_test_opt)
# print('\nstatsmodels:', evaluate(y_sm.round(), y_test), 
#       '% with {} parameters\n'.format(len(keys)), keys)


# regression_OLS = sm.OLS(y_train, x_opt).fit()
# #np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# y_sm = regression_OLS.predict(x_opt)

# print(y_train.shape, x_opt.shape, y_sm.shape, y_train.shape)
# print('statsmodels:', evaluate(y_sm.round(), y_train), '%')

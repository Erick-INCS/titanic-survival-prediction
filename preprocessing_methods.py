#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:31:43 2020

@author: incspn
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing

def probBy(dt, prop):
    data = dt[[prop, 'Survived']].dropna()
    data[prop] = data[prop].values.round()
    totalCount = data.groupby(prop, as_index=False).count()
    survCount = data.groupby(prop, as_index=False).sum()
    totalCount['count'] = totalCount['Survived']
    del totalCount['Survived']
    data = survCount.merge(totalCount)
    data['not'] = data['count'] - data['Survived']
    return data

def probas_categorical_prss(dt):
    res = dt.copy()
    data = probBy(res[['Pclass', 'Survived']], 'Pclass')
    data = data['Survived']/data['count']
    res['PclassP'] = data.values[res['Pclass'].values - 1]
    
    res['Sex'] = (res['Sex'].values == 'female').astype(int)
    data = probBy(res, 'Sex')
    data['percent'] = data['Survived']/data['count']
    res['SexP'] = data['percent'].values[res['Sex'].values - 1] 
    
    data = res[['Age', 'Survived']]
    data = probBy(data, 'Age')
    data['percent'] = data['Survived']/data['count']
    res['Age'] = res['Age'].values.round()
    
    res = res[np.isfinite(res['Age'])]
    data = data[['percent', 'Age']].set_index('Age')
    res['AgeP'] = data.loc[res['Age'].values].values

    data = dt.copy()
    data = probBy(data, 'SibSp')
    data['percent'] = data['Survived']/data['count']
    res['SibSpP'] = data.set_index('SibSp')['percent'].loc[res['SibSp'].values].values
    
    data = dt.copy()
    data = probBy(data[['Parch', 'Survived']], 'Parch')
    data['percent'] = data['Survived']/data['count']
    res['ParchP'] = data.set_index('Parch')['percent'].loc[res['Parch'].values].values
    
    data = dt[['Fare', 'Survived']].copy()
    data['Fare'] = np.log(1 + data['Fare'].values)
    data = probBy(data, 'Fare')
    data['percent'] = data['Survived']/data['count']
    res['Fare'] = np.log(1 + res['Fare'].values).round()
    res['FareP'] = data.set_index('Fare')['percent'].loc[res['Fare'].values].values
    
    del res['Cabin']
    
    data = dt.copy(['Embarked', 'Survive'])
    data = data.dropna()
    lblEncoder = preprocessing.LabelEncoder().fit(data['Embarked'])
    data['Embarked'] = lblEncoder.transform(data['Embarked'])

    res = res.dropna()
    data = probBy(data, 'Embarked')
    data['percent'] = data['Survived']/data['count']
    
    res['Embarked'] = lblEncoder.transform(res['Embarked']) 
    res['EmbarkedP'] = data.set_index('Embarked')['percent'].loc[res['Embarked'].values].values
    
    res['Pclass'] = res['Pclass']/3
    res['Age'] = res['Age']/80
    res['SibSp'] = res['SibSp']/max(res['SibSp'])
    return res
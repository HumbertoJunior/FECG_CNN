#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:54:32 2018

@author: HumbertoJunior
"""
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential 
from keras.layers import Dense

data = pd.read_csv('Documents/GitHub/CNN/Test_VerySimpleRNN.csv')

X = data.iloc[:,0].values
y = data.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


model = Sequential()

model.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 10, epochs = 20)

result = model.predict(np.array([1.9]))
print(result)


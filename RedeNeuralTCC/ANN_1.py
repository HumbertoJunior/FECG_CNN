# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('pacienteSimuladoFecg18B.csv')
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
classifier.add(Dropout(p = 0.2))

# Adding the second hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.2))

# Adding the third hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.2))

# Adding the fourth hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.2))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 64, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_2 = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

dataset2 = pd.read_csv('pacienteSimulado7500amostras.csv')
Z = dataset2.iloc[:,0:4].values

new_prediction = classifier.predict(sc.transform(np.array(Z)))
new_prediction = (new_prediction > 0.5)

batimento = 0
flag = 0
amostrasContadas = 0

for i in range(0,len(new_prediction)):
    if flag == 0:
        if new_prediction[i] == 1:
            amostrasContadas += 1
            if amostrasContadas > 4:
                batimento += 1
                flag = 1
                old_i = i
                amostrasContadas = 0
    else:
          if i > old_i + 100:
              flag = 0
        

#
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred_2)
#
## Part 4 - Evaluating, Improving and Tuning the ANN
#
## Evaluating the ANN
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from keras.models import Sequential
#from keras.layers import Dense
#def build_classifier():
#    classifier = Sequential()
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
#mean = accuracies.mean()
#variance = accuracies.std()
#
## Improving the ANN
## Dropout Regularization to reduce overfitting if needed
#
## Tuning the ANN
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#def build_classifier(optimizer):
#    classifier = Sequential()
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#classifier = KerasClassifier(build_fn = build_classifier)
#parameters = {'batch_size': [25, 32],
#              'epochs': [100, 500],
#              'optimizer': ['adam', 'rmsprop']}
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10)
#grid_search = grid_search.fit(X_train, y_train)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_
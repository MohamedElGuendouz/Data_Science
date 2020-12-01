# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('../Datasets/pima-indians-diabetes.data',header=None)
nb_dims=dataset.shape[1]-1
target_column_id=nb_dims
columns=range(nb_dims)

X = dataset.iloc[:, columns].values
y = dataset.iloc[:, target_column_id].values


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

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer,arch):
    classifier = Sequential()
    classifier.add(Dense(units = arch[0], kernel_initializer = 'uniform', activation = 'relu', input_dim = nb_dims))
    for i in range(1,len(arch)):
        classifier.add(Dense(units = arch[i], kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [32],
              'epochs': [500],
              'arch':[(6,6),(10,10)],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 3)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters,best_accuracy)


# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
classifier=grid_search.best_estimator_
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
print('Precision: ',(cm[0][0]+cm[1][1])/sum(sum(cm)))

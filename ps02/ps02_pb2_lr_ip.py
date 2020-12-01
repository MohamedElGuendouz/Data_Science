# Dataset: PIMA Indians Diabete

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pandas.plotting import scatter_matrix


# Importing the dataset

dataset = pd.read_csv('../Datasets/iris_proc.data',header=None)
nb_dims=dataset.shape[1]-1
feat_cols = [ 'V'+str(i+1) for i in range(nb_dims) ]
feat_cols.append('Target')
dataset.columns = feat_cols

columns=list(range(nb_dims))
X = dataset.iloc[:, columns].values
target_column_id=nb_dims
y = dataset.iloc[:, target_column_id].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('Confusion Matrix: ', cm)
print('  Precision: ',(cm[0][0]+cm[1][1])/sum(sum(cm)))

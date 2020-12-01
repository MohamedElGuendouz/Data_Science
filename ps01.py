# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 08:32:17 2020

@author: LAURI
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pandas.plotting import scatter_matrix
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import time


def displayScatter(dataset,nb_columns,diagonal='kde',figsize=(6, 6),alpha=0.1):
    # Display scatter plots of the first nb_columns columns
    scatter_matrix(dataset.iloc[:,:nb_columns], alpha=alpha, figsize=figsize, diagonal=diagonal)

    
def displayTSNE(dataset,nb_classes):
    # T-SNE vizualisation
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(dataset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    dataset['tsne-2d-one'] = tsne_results[:,0]
    dataset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Class",
        palette=sns.color_palette("hls", nb_classes),
        data=dataset,
        legend="full",
        alpha=0.3
    )


# Dataset: PIMA Indians

file='../Datasets/pima-indians-diabetes.data'
dataset = pd.read_csv(file,header=None)
dataset.columns=['V1','V2','V3','V4','V5','V6','V7','V8','Class']

displayScatter(dataset,8)

displayTSNE(dataset,2)


# Dataset: IRIS Plants

file='../Datasets/iris_proc.data'
dataset = pd.read_csv(file,header=None)
dataset.columns=['SepalLength','SepalWidth','PetalLength','PetalWidth','Class']

# Display scatter plots of only the first four columns
displayScatter(dataset,4)


displayTSNE(dataset,3)


# Dataset: Churn Modelling

file='../Datasets/Churn_Modelling.csv'
dataset = pd.read_csv(file)

X = dataset.iloc[:, 3:14].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# SKLearn >=0.22
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer([("Geography",OneHotEncoder(),[1])], remainder='passthrough')
X = onehotencoder.fit_transform(X)

# SKLearn <0.22
#onehotencoder = OneHotEncoder(categorical_features=[1])
#X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

X=pd.DataFrame(X)
X.columns=['Geography0','Geography1','CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Class']

displayTSNE(X,2)


# Dataset: Social Network

file='../Datasets/Social_Network_Ads.csv'
dataset = pd.read_csv(file)

X = dataset.iloc[:, 1:5].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])

X=pd.DataFrame(X)
X.columns=['Gender','Age','EstimatedSalary','Class']

# TSNE with gender
displayTSNE(X,2)

# TSNE without gender
displayTSNE(X.iloc[:,1:],2)

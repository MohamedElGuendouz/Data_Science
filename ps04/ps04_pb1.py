# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:43:18 2019

@author: LAURI
"""

import numpy as np
import seaborn as sb
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def display_figure(name,X,labels):
    X[name]=labels
    sb.catplot(x='X',y='Y',data=X,hue=name,s=40,cmap='viridis')
    plt.title(name)


X, labels = make_blobs(n_samples=50, n_features=2, centers=2)
X = pd.DataFrame(X,columns=['X','Y'])
X['Targets']=labels

sb.catplot(x='X',y='Y',data=X,hue='Targets',s=40,cmap='viridis')


from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=2).fit(X)
labels_gmm = gmm.predict(X)

display_figure('GMM',X,labels_gmm)

acc = np.sum(labels_gmm==labels)/float(len(labels_gmm))
print("Accuracy of GMM is %s " % acc)


from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=2).fit(X)
labels_ac = ac.labels_

display_figure('AgglomerativeClustering',X,labels_ac)

acc = np.sum(labels_ac==labels)/float(len(labels_ac))
print("Accuracy of AgglomerativeClustering is %s " % acc)

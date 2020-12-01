from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from keras.datasets import mnist


def prepareData(dataset):
    columns=['V'+str(i) for i in range(1,dataset.shape[1])]
    columns.append('Target')
    dataset.columns=columns
    
    feat_cols=list(range(dataset.shape[1]-1))
    data_subset = dataset.iloc[:,feat_cols]
    df_subset=data_subset.copy()
    return df_subset


# Dataset: PIMA Indians

dataset = pd.read_csv('../Datasets/pima-indians-diabetes.data',header=None)

df_subset=prepareData(dataset)


from sklearn.cluster import KMeans
clustering_kmeans = KMeans(n_clusters=2).fit(df_subset)
dataset['KMeans']=clustering_kmeans.labels_

from sklearn.cluster import MeanShift
clustering_meanshift = MeanShift().fit(df_subset)
dataset['Meanshift']=clustering_meanshift.labels_

from sklearn.cluster import DBSCAN
clustering_dbscan = DBSCAN(eps=3., min_samples=2,algorithm='brute').fit(df_subset)
dataset['DBScan']=clustering_dbscan.labels_

from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=2).fit(df_subset)
labels_gmm = gmm.predict(df_subset)
dataset['GMM']=labels_gmm

from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=2).fit(df_subset)
dataset['AC']=ac.labels_

t=np.array(dataset['Target'])

a=np.array(dataset['KMeans'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('KMeans Accuracy: ',acc*100.0)

a=np.array(dataset['Meanshift'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('Meanshift Accuracy: ',acc*100.0)

a=np.array(dataset['DBScan'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('DBScan Accuracy: ',acc*100.0)

a=np.array(dataset['GMM'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('GMM Accuracy: ',acc*100.0)

a=np.array(dataset['AC'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('AC Accuracy: ',acc*100.0)


# Dataset: Iris Plant

dataset = pd.read_csv('../Datasets/iris_proc.data',header=None)

df_subset=prepareData(dataset)

from sklearn.cluster import KMeans
clustering_kmeans = KMeans(n_clusters=3).fit(df_subset)
dataset['KMeans']=clustering_kmeans.labels_

from sklearn.cluster import MeanShift
clustering_meanshift = MeanShift().fit(df_subset)
dataset['Meanshift']=clustering_meanshift.labels_

from sklearn.cluster import DBSCAN
clustering_dbscan = DBSCAN(eps=3., min_samples=2,algorithm='brute').fit(df_subset)
dataset['DBScan']=clustering_dbscan.labels_

from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=3).fit(df_subset)
labels_gmm = gmm.predict(df_subset)
dataset['GMM']=labels_gmm

from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3).fit(df_subset)
dataset['AC']=ac.labels_

t=np.array(dataset['Target'])

a=np.array(dataset['KMeans'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('KMeans Accuracy: ',acc*100.0)

a=np.array(dataset['Meanshift'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('Meanshift Accuracy: ',acc*100.0)

a=np.array(dataset['DBScan'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('DBScan Accuracy: ',acc*100.0)

a=np.array(dataset['GMM'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('GMM Accuracy: ',acc*100.0)

a=np.array(dataset['AC'])
acc=float(np.sum(a==t))/dataset.shape[0]
print('AC Accuracy: ',acc*100.0)

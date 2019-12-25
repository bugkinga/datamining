# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:50:31 2019

@author: 12090
"""

print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import AffinityPropagation,KMeans,MeanShift,estimate_bandwidth,SpectralClustering,AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

X_digits, y_digits = load_digits(return_X_y=True)
data = scale(X_digits)

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('homo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

def p(x,y):
    print('%.3f\t%.3f\t%.3f'
          % (
         metrics.homogeneity_score(x, y),
         metrics.completeness_score(x, y),
         metrics.normalized_mutual_info_score(x, y)
                                  ))


#begin kmeans
k_means=KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
k_means.fit(data)
print('kmeans')
p(labels,k_means.labels_)
#end kmeans

#begin affi

Affi=AffinityPropagation(damping=0.8,preference=-2600)
Affi.fit(data)
print('affi')
p(labels,Affi.labels_)
print(Affi.cluster_centers_indices_.shape)

#end affi
#begin meanshift
#bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=100)   
ms = MeanShift(bandwidth=1.5, bin_seeding=True)
#ms=MeanShift()
ms.fit(data)
print('meanshift')
p(labels,ms.labels_)
print(ms.cluster_centers_.shape)
#end meanshift
#begin spectral
sc=SpectralClustering(n_clusters=10,affinity='nearest_neighbors')
sc.fit(data)
print('spclu:')
p(labels,sc.labels_)
#end
#begin
Ag=AgglomerativeClustering(linkage='ward',n_clusters=10)
Ag.fit(data)
print('ward')
p(labels,Ag.labels_)
#end
#begin
for i in ('average','complete','single'):
    tmp=AgglomerativeClustering(linkage=i,n_clusters=10)
    tmp.fit(data)
    print(i)
    p(labels,tmp.labels_)
    
#end
#begin
DB=DBSCAN(eps=4.5,min_samples=10).fit(data)
print('DBSCAN')
p(labels,DB.labels_)
print(DB.core_sample_indices_.shape)
#end
GMM=GaussianMixture(n_components=10).fit(data)
print('GMM')
p(labels,GMM.predict(data))


print(82 * '_')

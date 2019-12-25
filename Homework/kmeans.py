# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:50:31 2019

@author: 12090
"""

print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation,KMeans,MeanShift,estimate_bandwidth,SpectralClustering,AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn import metrics
#from sklearn.cluster import KMeans
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
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')




# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
ms = GaussianMixture(n_components=10)
ms.fit(reduced_data)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].


'''
cluster_centers=ms.cluster_centers_
print(cluster_centers.shape)'''


#labels=ms.labels_
labels=ms.predict(reduced_data)
labels_unique = np.unique(labels)
n = len(labels_unique)
print(n)


from itertools import cycle
plt.figure(1)
plt.clf()
tmp_c=['b','g','r','c','m','y','k','c','g','m']
colors = cycle(tmp_c)
for k, col in zip(range(n), colors):
    my_members = labels == k
    #cluster_center = cluster_centers[k]
    plt.plot(reduced_data[my_members, 0], reduced_data[my_members, 1], col + '.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
     #        markeredgecolor='k', markersize=14)
plt.title('ward')
plt.show()









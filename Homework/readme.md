# Cluster with Sklearn

## Dataset

sklearn.datasets.load_digits
one.py
手写数字数据集，结构化数据的经典数据，共有1797个样本，每个样本有64的元素，对应到一个8x8像素点组成的矩阵，每个值是其灰度值。

|       Name        |    number     |
| :---------------: | :-----------: |
|      Classes      |      10       |
| Samples per class |     ~180      |
|   Samples total   |     1797      |
|  Dimensionality   |      64       |
|     Features      | integers 0-16 |

sklearn.datasets.fetch_20newsgroups
two.py
20新闻语料，含有20个类

|      Name      | number |
| :------------: | :----: |
|    Classes     |   20   |
| Samples total  | 18846  |
| Dimensionality |   1    |
|    Features    |  text  |

## Approach

### Kmeans 

先从样本集中随机选取 *k* 个样本作为**簇中心**，并计算所有样本与这 k 个“簇中心”的距离，对于每一个样本，将其划分到与其**距离最近**的“簇中心”所在的簇中，对于新的簇计算各个簇的新的“簇中心”。

*class* `sklearn.cluster.KMeans`(*n_clusters=8*, *init='k-means++'*, *n_init=10*, *max_iter=300*, *tol=0.0001*, *precompute_distances='auto'*, *verbose=0*, *random_state=None*, *copy_x=True*, *n_jobs=None*, *algorithm='auto'*)

### AffinityPropagation

AP算法的基本思想是将全部样本看作网络的节点，然后通过网络中各条边的消息传递计算出各样本的聚类中心。聚类过程中，共有两种消息在各节点间传递，分别是吸引度( responsibility)和归属度(availability) 。AP算法通过迭代过程不断更新每一个点的吸引度和归属度值，直到产生m个高质量的Exemplar（类似于质心），同时将其余的数据点分配到相应的聚类中。

*class* `sklearn.cluster.AffinityPropagation`(*damping=0.5*, *max_iter=200*, *convergence_iter=15*, *copy=True*, *preference=None*, *affinity='euclidean'*, *verbose=False*)

### MeanShift

算法的关键操作是通过感兴趣区域内的数据密度变化计算中心点的漂移向量，从而移动中心点进行下一次迭代，直到到达密度最大处（中心点不变）。从每个数据点出发都可以进行该操作，在这个过程，统计出现在感兴趣区域内的数据的次数。该参数将在最后作为分类的依据。

*class* `sklearn.cluster.MeanShift`(*bandwidth=None*, *seeds=None*, *bin_seeding=False*, *min_bin_freq=1*, *cluster_all=True*, *n_jobs=None*, *max_iter=300*)

### Spectral clustering

基本思想是利用样本数据的相似矩阵(拉普拉斯矩阵)进行特征分解后得到的特征向量进行聚类。

*class* `sklearn.cluster.SpectralClustering`(*n_clusters=8*, *eigen_solver=None*, *n_components=None*, *random_state=None*, *n_init=10*, *gamma=1.0*, *affinity='rbf'*, *n_neighbors=10*, *eigen_tol=0.0*, *assign_labels='kmeans'*, *degree=3*, *coef0=1*, *kernel_params=None*, *n_jobs=None*)

### Ward hierarchical clustering

下面的方法的一个特例，ward最小化了集群合并的方差。

### Agglomerative clustering

算法一开始将每个实例都看成一个簇，在一个确定的”**聚类间度量函数**“的驱动下，算法的每次迭代中都将最相似的两个簇合并在一起，该过程不断重复直到只剩下一个簇为止。该方法称为**层次聚类**

*class* `sklearn.cluster.AgglomerativeClustering`(*n_clusters=2*, *affinity='euclidean'*, *memory=None*, *connectivity=None*, *compute_full_tree='auto'*, *linkage='ward'*, *distance_threshold=None*)

### DBSCAN

一个比较有代表性的基于密度的聚类算法。与划分和层次聚类方法不同，它将簇定义为密度相连的点的最大集合，能够把具有足够高密度的区域划分为簇，并可在噪声的空间数据库中发现任意形状的聚类。

*class* `sklearn.cluster.DBSCAN`(*eps=0.5*, *min_samples=5*, *metric='euclidean'*, *metric_params=None*, *algorithm='auto'*, *leaf_size=30*, *p=None*, *n_jobs=None*)

### Gaussian mixtures

高斯混合模型就是用高斯[概率密度函数](https://baike.baidu.com/item/概率密度函数)（[正态分布](https://baike.baidu.com/item/正态分布)曲线）精确地量化事物，它是一个将事物分解为若干的基于高斯概率密度函数（正态分布曲线）形成的[模型](https://baike.baidu.com/item/模型/1741186)。

## Evaluation Metrics

homogeneity_score  每一个聚出的类仅包含一个类别的程度度量。

completeness_score  每一个类别被指向相同聚出的类的程度度量。

normalized_mutual_info_score   基于互信息

## Results of digist

|           Methods            | homo  | comp  | norm  |
| :--------------------------: | :---: | :---: | :---: |
|            Kmeans            | 0.602 | 0.650 | 0.626 |
|     Affinitypropagation      | 0.647 | 0.658 | 0.652 |
|          MeanShift           |   1   | 0.307 | 0.554 |
|     Spectral clustering      | 0.805 | 0.853 | 0.829 |
| ward hierarchical clustering | 0.758 | 0.836 | 0.797 |
|    Agglomerative(average)    | 0.007 | 0.238 | 0.041 |
|   Agglomerative(complete)    | 0.017 | 0.249 | 0.065 |
|    Agglometrative(single)    | 0.006 | 0.276 | 0.039 |
|            DBSCAN            | 0.424 | 0.623 | 0.514 |
|      Gaussian mixtures       | 0.600 | 0.646 | 0.622 |

通过观察结果，可以看到整体上的三个指标，Spectrcal clustering表现的最好，层次聚类的方法则在这个上面表现较差. 并且在实验过程中可以感受到模型的一些参数对于最后结果的影响很大。





## Results of 20ngnews

原始的数据的维度过大，通过LSA进行降维，如下是降为10维的结果。如果不降维，对于某些方法则输入的特征矩阵过于稀疏无法很好的运算。

| Method                       | homo  | comp  | norm  |
| ---------------------------- | ----- | ----- | ----- |
| Kmeans                       | 0.607 | 0.669 | 0.637 |
| Affinitypropagation          | 0.730 | 0.236 | 0.415 |
| Meanshift                    | 0.969 | 0.171 | 0.407 |
| Spectral clustering          | 0.601 | 0.687 | 0.642 |
| ward hierarchical clustering | 0.557 | 0.559 | 0.558 |
| Agglometrative(Average)      | 0.071 | 0.222 | 0.125 |
| Agglometrative(Complete)     | 0.230 | 0.330 | 0.275 |
| Agglometrative(single)       | 0.002 | 0.171 | 0.016 |
| DBSCAN                       | 0     | 1     | 0     |
| Gaussian Mixtures            | 0.579 | 0.579 | 0.579 |


# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

df = datasets.load_iris()
df = pd.DataFrame(df.data, columns = df.feature_names)

X = df.iloc[:, :].values




# METHOD 1: K-Means
from sklearn.cluster import KMeans
sumOfSq=[]
for i in range(1,7):
    model = KMeans(n_clusters=i, init='k-means++')
    model.fit(X)
    sumOfSq.append(model.inertia_)

plt.plot(range(1, 7), sumOfSq)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Cluster sum of squares')
plt.show()
"""
Results: 3 Clusters
"""

"WITHOUT PCA"
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')
plt.title('KMeans Clustering')
plt.legend()
plt.show()



"WITH PCA"
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf')
X2d = kpca.fit_transform(X)

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X2d)

plt.scatter(X2d[y_kmeans == 0, 0], X2d[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X2d[y_kmeans == 1, 0], X2d[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X2d[y_kmeans == 2, 0], X2d[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')
plt.title('KMeans Clustering (WITH PCA)')
plt.legend()
plt.show()





# METHOD 2: Hierarchical Clustering
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.title("Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Euclidean Distance")
plt.show()

"WITHOUT PCA"
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(3, "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
plt.title('Hierarchical Clustering')
plt.legend()
plt.show()



"WITH PCA"
y_hc = hc.fit_predict(X2d)
plt.scatter(X2d[y_hc == 0, 0], X2d[y_hc == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X2d[y_hc == 1, 0], X2d[y_hc == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X2d[y_hc == 2, 0], X2d[y_hc == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
plt.title('KMeans Clustering (WITH PCA)')
plt.legend()
plt.show()


#import neccessary libraries

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#step 1: create the data points
X = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])

#step 2: apply k-means algorithm with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X)

cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:,0],cluster_centers[:,1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering with 2 Clusters')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.show()

print("Cluster Centers:\n",cluster_centers)
print("Labels:",labels)

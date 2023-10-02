import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load the Excel file into a DataFrame

df=pd.read_csv("data.csv")
df.drop(index=df.index[-5:])

X=df.iloc[:-2,1:len(df)-1].values

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
cluster_labels =  kmeans.labels_
#df["Cluster"] = cluster_labels

plt.figure(figsize=(8, 6))
for cluster_label in range(3):
    kmeans.fit(X)
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=200, color='red', label='Cluster Centers')

plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Create an Elbow Graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squares')
plt.title('Elbow Graph')
plt.xticks(np.arange(1, 11, 1))
plt.grid(True)
plt.show()

y = df.iloc[:, -1] .values
data_points = np.hstack((X, y.reshape(-1, 1)))
labels = kmeans.labels_
silhouette_avg = silhouette_score(data_points, labels)
davies_bouldin = davies_bouldin_score(data_points, labels)
calinski_harabasz = calinski_harabasz_score(data_points, labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Score: {davies_bouldin}")
print(f"Calinski-Harabasz Score: {calinski_harabasz}")

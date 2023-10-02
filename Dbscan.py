import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# Load data from a CSV file
df = pd.read_csv('cleaned_data.csv')
features = df.drop(['S.NO'], axis=1)

eps = 0.5  # Set the epsilon (neighborhood radius) parameter (you can adjust this)
min_samples = 5  # Set the minimum number of samples in a neighborhood (you can adjust this)
dbscan_clustering = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the model to the data and obtain cluster labels
dbscan_labels = dbscan_clustering.fit_predict(features)

plt.scatter(features['Plant height'], features['YIELD'], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering Results')
plt.xlabel('Plant Height ')
plt.ylabel('Yield')
plt.show()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1] .values
data_points = np.hstack((X, y.reshape(-1, 1)))
silhouette_avg = silhouette_score(data_points, dbscan_labels)
davies_bouldin = davies_bouldin_score(data_points, dbscan_labels)
calinski_harabasz = calinski_harabasz_score(data_points, dbscan_labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Score: {davies_bouldin}")
print(f"Calinski-Harabasz Score: {calinski_harabasz}")

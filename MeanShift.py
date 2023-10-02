import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load data from a CSV file
df = pd.read_csv('cleaned_data.csv')
features = df.drop(['S.NO'], axis=1)


bandwidth = 2.0  # Set the bandwidth parameter (you can adjust this)
ms_clustering = MeanShift(bandwidth=bandwidth)

# Fit the model to the data
ms_labels = ms_clustering.fit_predict(features)

plt.scatter(features['Plant height'], features['YIELD'], c=ms_labels, cmap='viridis')
plt.title('Mean Shift Clustering Results')
plt.xlabel('Plant height')
plt.ylabel('Yield')
plt.show()


X = df.iloc[:, :-1].values
y = df.iloc[:, -1] .values
data_points = np.hstack((X, y.reshape(-1, 1)))
ms = MeanShift()
ms.fit(data_points)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

silhouette_avg = silhouette_score(data_points, labels)
davies_bouldin = davies_bouldin_score(data_points, labels)
calinski_harabasz = calinski_harabasz_score(data_points, labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Score: {davies_bouldin}")
print(f"Calinski-Harabasz Score: {calinski_harabasz}")

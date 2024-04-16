# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Pick customer segment quantity (k).
2. Seed cluster centers with random data points.
3. Assign customers to closest centers.
4. Re-center clusters and repeat until stable.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Gokul C
RegisterNumber: 212223240040

import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data = pd.read_csv('/content/Mall_Customers_EX8.csv')
data
X = data[['Annual Income (k$)','Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data ['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r','g','b','c','m']
for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster {i+1}')
  distances = euclidean_distances(cluster_points,[centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
*/
```

## Output:
![Screenshot 2024-04-16 164440](https://github.com/Gokul1410/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/153058321/167ae9cc-26bd-41bf-bb1f-456fd0b66754)
![Screenshot 2024-04-16 162258](https://github.com/Gokul1410/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/153058321/563e6e01-809f-4393-94ce-3238e437c1dd)
![Screenshot 2024-04-16 162317](https://github.com/Gokul1410/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/153058321/002b6c02-38c7-4a01-8146-9ec1c9b54625)
![Screenshot 2024-04-16 162332](https://github.com/Gokul1410/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/153058321/79376681-9abf-4947-a9a9-2ab67e132ecc)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

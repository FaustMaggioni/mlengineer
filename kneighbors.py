from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=6)
iris= datasets.load_iris()
knn.fit(iris['data'], iris['target'])
print(iris['data'].shape)
print(iris['target'].shape)

X_new = np.array([[5.6,2.8,3.9,1.1],
    [5.7,2.6,3.8,1.3],
    [4.7,3.2,1.3,0.2]])

prediction= knn.predict(X_new)
print(X_new.shape)
print('Prediction: {}'.format(prediction))
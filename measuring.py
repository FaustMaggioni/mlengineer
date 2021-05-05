from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

data= datasets.load_iris()
y= data['target']
X= data['data']
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
y_pred= knn.predict(X_test)
print("test set preds: \\n {}".format(y_pred))
print(knn.score(X_test,y_test))

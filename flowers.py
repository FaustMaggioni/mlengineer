from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris= datasets.load_iris()
print(type(iris))

print(iris.keys())
print(type(iris.data), type(iris.target))

print(iris.data.shape) 
print(iris.target_names)

X= iris.data
Y= iris.target

df= pd.DataFrame(X, columns= iris.feature_names)
print(df.head())
plt.figure()
_ = pd.plotting.scatter_matrix(df, c=Y, figsize=[8,8], s=150, marker='D')
print(_)
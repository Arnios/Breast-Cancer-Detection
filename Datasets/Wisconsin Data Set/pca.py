##### Import libraries #####

import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

##### Importing data and splitting into test and train set #####

dataset = pandas.read_csv('data.csv')
features = ['id', 'diagnosis', 'radius_mean', 'Unnamed: 32', 'radius_worst', 'perimeter_mean', 'area_mean', 'area_worst', 'radius_se', 'perimeter_se', 'radius_worst']

X = dataset.drop(features, axis = 1) # Revised Independent Variable Set
Y = dataset['diagnosis'] # Dependant variables

X = StandardScaler().fit_transform(X) # Feature scaling
Y = LabelEncoder().fit_transform(Y) # Enconding the categorical dependant variable

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.20, random_state = 0) # Test-Train Split
X_Train = StandardScaler().fit_transform(X_Train) 
X_Test = StandardScaler().fit_transform(X_Test)

##### Principal Component Analysis #####

pca = PCA(n_components = None)

X_Train = pca.fit_transform(X_Train) 
X_Test = pca.fit_transform(X_Test)

explained_variance = pca.explained_variance_ratio_

##### Plot #####

plt.figure(1, figsize = (18, 18))
plt.plot(pca.explained_variance_ratio_, linewidth = 2)
plt.axis('tight')
plt.xlabel('Number of components')
plt.ylabel('Explained Variance Ratio')
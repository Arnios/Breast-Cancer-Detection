################################################## Importing libraries ###############################################

import keras
import xgboost
import tensorflow
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

##################################################### Importing dataset ###############################################

import numpy
import pandas
import seaborn

dataset = pandas.read_csv('breast-cancer-wisconsin.data', delimiter = ",", header = None, names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])
information = pandas.concat([dataset.dtypes, dataset.nunique(dropna = False).sort_values(ascending = False), dataset.isnull().sum().sort_values(ascending = False), (100*dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending = False)], axis = 1, keys = ['Type', 'Unique Values', 'Null Values', 'Null Percentage']) # Null Value Check

X = dataset.drop(['id', 'diagnosis', 'Unnamed: 32'], axis = 1) # Independent variables
Y = dataset.diagnosis # Dependant variables

ax1 = seaborn.countplot(Y, label = 'Count')
B, M = Y.value_counts()
print('Number of Benign: ', B)
print('Number of Malignant : ', M)

ax2 = seaborn.lineplot(data = X.mean()) # Line plot of mean of individual features
ax2.set(xticklabels = [])
ax2.set(xlabel = 'Features')

X = (X - X.min()) / (X.max() - X.min()) # Normalization of features

###################################################### Feature Importance #############################################

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

features = {}
model = RandomForestClassifier()
model.fit(X, Y)

for a, b in zip(X.columns, model.feature_importances_):
    features[a] = b 

importances = pandas.DataFrame.from_dict(features, orient = 'index').rename(columns = {0: 'Importance'})

importances.sort_values(by = 'Importance').plot(kind = 'bar', rot = 90)

######################################### Swarm Plot of Insignificant Features ########################################

# columns = ['smoothness_mean', 'compactness_mean', 'symmetry_mean']
# columns = ['fractal_dimension_mean', 'radius_se', 'texture_se']
# columns = ['smoothness_se', 'compactness_se', 'concavity_se']
# columns = ['concave points_se', 'symmetry_se', 'fractal_dimension_se']
# columns = ['symmetry_worst', 'fractal_dimension_worst']

features = pandas.concat([Y, X[columns]], axis = 1)
features = pandas.melt(features, id_vars = 'diagnosis', var_name = 'features', value_name = 'value')

plot = seaborn.set(style = 'whitegrid', palette = 'muted')
plot = seaborn.swarmplot(x = 'features', y = 'value', hue = 'diagnosis', data = features)

############################################### Drop Insignificant Features ###########################################

insignificant_features = ['smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'symmetry_worst', 'fractal_dimension_worst']
X = X.drop(insignificant_features, axis = 1)
dataset = pandas.concat([Y, X], axis = 1)

############################################ Pearson Correlation Matrix ###############################################

mask = numpy.zeros_like(X.corr())
mask[numpy.triu_indices_from(mask)] = True

with seaborn.axes_style("white") :
    
    f, ax = plt.subplots(figsize = (18, 18))
    ax = seaborn.heatmap(X.corr(), mask = mask, annot = True, linewidths = .5, fmt = '.1f', square = True)
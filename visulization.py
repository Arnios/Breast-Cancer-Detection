################################################## Importing libraries ###############################################

import keras
import numpy
import pandas
import seaborn
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

################################################# Data Visualization #################################################

##### Importing data and Standardization #####

dataset = pandas.read_csv('data.csv')

X = dataset.drop(['id', 'diagnosis', 'Unnamed: 32'], axis = 1) # Independent variables
Y = dataset.diagnosis # Dependant variables
# X.mean().plot(kind = 'bar')
# X = (X - X.mean()) / (X.std()) # Standardization of features

##### Initial Correlation Matrix #####

mask = numpy.zeros_like(X.corr())
mask[numpy.triu_indices_from(mask)] = True

with seaborn.axes_style("white") :
    
    f, ax = plt.subplots(figsize=(18, 18))
    ax = seaborn.heatmap(X.corr(), mask=mask, annot = True, linewidths = .5, fmt = '.1f', square=True)
    
##### Drawing Violin Plot and Box Plots #####

# features = pandas.concat([Y, X.iloc[:, 0:3]], axis = 1)
# features = pandas.melt(features, id_vars = 'diagnosis', var_name = 'features', value_name = 'value')
# violinplot = seaborn.violinplot(x = 'value', y = 'features', hue = 'diagnosis', data = features, split = True, inner = "quart")
# violinplot.set_xticklabels(violinplot.get_xticklabels(), rotation = 90)
# boxplot = seaborn.boxplot(x = 'value', y = 'features', hue = "diagnosis", data = features)
# boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation = 90)

##### Swarm Plots of first cohort of features #####

# columns = ['radius_mean', 'perimeter_mean', 'area_mean', 'radius_worst', 'perimeter_worst']
# columns = ['perimeter_mean', 'perimeter_worst', 'radius_worst', 'area_mean']
# columns = ['area_mean', 'area_worst', 'perimeter_worst', 'radius_worst']
# columns = ['radius_se', 'perimeter_se', 'area_se']
# columns = ['radius_worst', 'perimeter_worst', 'area_worst']
# columns = ['perimeter_worst', 'area_worst']

# features = pandas.concat([Y, X[columns]], axis = 1)
# features = pandas.melt(features, id_vars = 'diagnosis', var_name = 'features', value_name = 'value')

# plot = seaborn.set(style = 'whitegrid', palette = 'muted')
# plot = seaborn.swarmplot(x = 'features', y = 'value', hue = 'diagnosis', data = features)

##### Dropping features after swarm plot analysis and drawing new correlation matrix #####

features = ['radius_mean', 'radius_worst', 'perimeter_mean', 'area_mean', 'area_worst', 'radius_se', 'perimeter_se', 'radius_worst']
X = X.drop(features, axis = 1)
dataset = pandas.concat([Y, X], axis = 1)

# mask = numpy.zeros_like(X.corr())
# mask[numpy.triu_indices_from(mask)] = True

# with seaborn.axes_style("white") :
    
#     f, ax = plt.subplots(figsize = (18, 18))
#     ax = seaborn.heatmap(X.corr(), mask = mask, annot = True, linewidths = .5, fmt = '.1f', square = True)
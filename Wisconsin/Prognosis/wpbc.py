################################################## Importing libraries ###############################################

import keras
import xgboost
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

##################################################### Importing dataset ###############################################

import numpy
import pandas

dataset = pandas.read_csv('wpbc.data', delimiter = ",", header = None, names = ['ID number', 'Outcome', 'Time', 'radius mean', 'texture mean', 'perimeter mean', 'area mean', 'smoothness mean', 'compactness mean', 'concavity mean', 'concave points mean', 'symmetry mean', 'fractal dimension mean', 'radius se', 'texture se', 'perimeter se', 'area se', 'smoothness se', 'compactness se', 'concavity se', 'concave points se', 'symmetry se', 'fractal dimension se', 'radius worst', 'texture worst', 'perimeter worst', 'area worst', 'smoothness worst', 'compactness worst', 'concavity worst', 'concave points worst', 'symmetry worst', 'fractal dimension worst', 'Tumor size', 'Lymph node status'])
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

### Chi Squared Test ###

from scipy.stats import chi2
from scipy.stats import chi2_contingency

columns = ['Gender', 'Country', 'self_employed', 'family_history',
           'work_interfere', 'no_employees', 'remote_work',
           'tech_company', 'benefits', 'care_options', 'wellness_program',
           'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
           'phys_health_consequence', 'coworkers', 'supervisor',
           'mental_health_interview', 'phys_health_interview',
           'mental_vs_physical', 'obs_consequence']

# Define Chi-Squared Test Parameters

prob = 0.95
alpha = 1.0 - prob

table = pandas.crosstab(dataset['treatment'], dataset['Gender'], margins = False)
stat, p, dof, expected = chi2_contingency(table)

# Interpret Test statistic and cross check with inference via p-value

critical = chi2.ppf(prob, dof)

print('Critical = %.3f, P-Value = %.3f, Stat = %.3f' % (critical, p, stat))

if abs(stat) >= critical :
	print('Dependent (reject H0)')
else :
	print('Independent (fail to reject H0)')
    
if p <= alpha :
	print('Dependent (reject H0)')
else :
	print('Independent (fail to reject H0)')

# Deleting temporary variable

del prob, alpha, table
del stat, p, dof, expected, critical

### Random Forest ###

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
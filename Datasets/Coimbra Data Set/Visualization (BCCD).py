################################################### Importing dataset ################################################

### Importing Libraries ###

import numpy
import pandas
import seaborn

### Import Dataset ###

dataset = pandas.read_csv('Data/dataR2.csv')
dataset.columns = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP1', 'Classification'] # Reformat Column Names
datasetinfo = pandas.concat([dataset.dtypes, dataset.nunique(dropna = False).sort_values(ascending = False), dataset.isnull().sum().sort_values(ascending = False), (100*dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending = False)], axis = 1, keys = ['Type', 'Unique Values', 'Null Values', 'Null Percentage']) # Null Value Check

X = pandas.DataFrame()
Y = pandas.DataFrame()

################################################### Preliminary Analysis #############################################

### Line plot of mean of individual features ###

ax = seaborn.lineplot(data = X.mean())
ax.set(xticklabels = [])
ax.set(xlabel = 'Features')

del ax # Delete Temporary Variables

### Understanding the distribution of the target variable among classes ###

P, H = Y.value_counts()
print('Number of Patients : ', P)
print('Number of Non-Patients : ', H)
ax = seaborn.countplot(Y, label = 'Count')

del ax, P, H # Delete Temporary Variables

############################################ Outlier Treatment and Normalization #####################################

### Box and Whisker Plot of feature ###

dataset.boxplot(column = 'Adiponectin', vert = True)

for feature in dataset :
    
    if feature != 'Classification' :
    
        Q1 = dataset[feature].quantile(0.25) # First Quartile
        Q3 = dataset[feature].quantile(0.75) # Third Quartile
        IQR = Q3 - Q1 # Interquartile Range
        
        dataset.query('(@Q1 - 1.5*@IQR) <= {} <= (@Q3 + 1.5*@IQR)'.format(feature), inplace = True) # Filtering Values between (Q1-1.5IQR) and (Q3+1.5IQR)
            
        del Q1, Q3, IQR # Delete Temporary Variables

del feature # Delete Temporary Variables

### Datatype Conversion and Normalization of features ###

Y['Classification'] = dataset['Classification'] # Target variables
dataset = dataset.drop(['Classification'], axis = 1)
X = dataset

Y = numpy.asarray(Y).astype('float32')
X = numpy.asarray(X).astype('float32')

X = (X - X.min()) / (X.max() - X.min()) # Normalization

###################################################### Feature Importance #############################################

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

features = {}
model = RandomForestClassifier()
model.fit(X, Y)

for a, b in zip(X.columns, model.feature_importances_):
    features[a] = b 

importances = pandas.DataFrame.from_dict(features, orient = 'index').rename(columns = {0: 'Importance'})
importances.sort_values(by = 'Importance').plot(kind = 'barh')

# Delete Temporary Variables

del a, b
del features, importances, model

######################################### Swarm Plot of Insignificant Features ########################################

columns = ['HOMA', 'Insulin']

features = pandas.concat([Y, X[columns]], axis = 1)
features = pandas.melt(features, id_vars = 'Classification', var_name = 'features', value_name = 'value')

plot = seaborn.set(style = 'whitegrid', palette = 'muted')
plot = seaborn.swarmplot(x = 'features', y = 'value', hue = 'Classification', data = features)

# Delete Temporary Variables

del columns, features, plot

############################################ Pearson Correlation Matrix ###############################################

mask = numpy.zeros_like(X.corr())
mask[numpy.triu_indices_from(mask)] = True

with seaborn.axes_style("white") :
    
    f, ax = plt.subplots(figsize = (18, 18))
    ax = seaborn.heatmap(X.corr(), mask = mask, annot = True, linewidths = .5, fmt = '.1f', square = True)
    
############################################### Drop Insignificant Features ###########################################

insignificant_features = ['Insulin']
X = X.drop(insignificant_features, axis = 1)

del insignificant_features # Delete Temporary Variables
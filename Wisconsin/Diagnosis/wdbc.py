################################################## Importing libraries ###############################################

import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

##################################################### Importing dataset ###############################################

from sklearn.preprocessing import LabelEncoder

dataset = pandas.read_csv('wdbc.data', delimiter = ",", header = None, names = ['ID_number', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'])
information = pandas.concat([dataset.dtypes, dataset.nunique(dropna = False).sort_values(ascending = False), dataset.isnull().sum().sort_values(ascending = False), (100*dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending = False)], axis = 1, keys = ['Type', 'Unique Values', 'Null Values', 'Null Percentage']) # Null Value Check

X = dataset.drop(['ID_number', 'Diagnosis'], axis = 1) # Independent variables
Y = dataset['Diagnosis'] # Dependant variables

X = (X - X.min()) / (X.max() - X.min()) # Min-Max Normalization
Y = LabelEncoder().fit_transform(Y) # Enconding the categorical dependant variable

X = numpy.asarray(X).astype('float32')
Y = numpy.asarray(Y).astype('float32')

################################################# Principal Component Analysis ##########################################

for i in range(10, 55, 5) :

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = i/100, random_state = 1) # Split data into train and test sets
    pca(X_Train, X_Test, None, i) # Type None for all features
    
    del X_Train, X_Test, Y_Train, Y_Test # Deleting Temporary Variables
    
del i

######################################################### ANOVA F-Test ################################################

for i in range(10, 55, 5) :    

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = i/100, random_state = 1) # Split data into train and test sets
    anova(X_Train, Y_Train, X_Test, 'all') # Type all for all features
    
    del X_Train, X_Test, Y_Train, Y_Test # Deleting Temporary Variables
    
del i

################################################# Understanding the data #############################################

import matplotlib.pyplot as plt

### Target variable distribution ###

ax1 = seaborn.countplot(Y, label = 'Count')
B, M = Y.value_counts()
print('Number of Benign: ', B)
print('Number of Malignant : ', M)

### Feature mean distribution ###

ax2 = seaborn.lineplot(data = X.mean()) # Line plot of mean of individual features
ax2.set(xticklabels = [])
ax2.set(xlabel = 'Features')

### Creating a 5X6 Boxplot matrix of the independent variables ###

figure = plt.figure()

for feature, num in zip(X.columns, range(1, 31)) :
    
    ax = figure.add_subplot(5, 6, num)
    ax.boxplot(X[feature])
    ax.set_title(feature, size = 5, pad = -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

### Pearson Correlation Matrix ###

mask = numpy.zeros_like(X.corr())
mask[numpy.triu_indices_from(mask)] = True

with seaborn.axes_style("white") :
    
    f, ax = plt.subplots(figsize = (18, 18))
    ax = seaborn.heatmap(X.corr(), mask = mask, annot = True, linewidths = .5, fmt = '.1f', square = True)
    
### Swarm Plot Analysis ###

columns = ['smoothness_mean', 'compactness_mean', 'symmetry_mean']
columns = ['fractal_dimension_mean', 'radius_se', 'texture_se']
columns = ['smoothness_se', 'compactness_se', 'concavity_se']
columns = ['concave points_se', 'symmetry_se', 'fractal_dimension_se']
columns = ['symmetry_worst', 'fractal_dimension_worst']

features = pandas.concat([Y, X[columns]], axis = 1)
features = pandas.melt(features, id_vars = 'diagnosis', var_name = 'features', value_name = 'value')

plot = seaborn.set(style = 'whitegrid', palette = 'muted')
plot = seaborn.swarmplot(x = 'features', y = 'value', hue = 'diagnosis', data = features)

### Deleting Temporary Variables ###

del ax1, ax2
del B, M
del figure, feature, num, ax, plt

#################################################### Outlier Elimination #############################################

for feature in dataset :
    
    if feature != 'ID_number' and feature != 'Diagnosis' :
    
        Q1 = dataset[feature].quantile(0.25) # First Quartile
        Q3 = dataset[feature].quantile(0.75) # Third Quartile
        IQR = Q3 - Q1 # Interquartile Range
        
        # Filtering Values between (Q1-1.5IQR) and (Q3+1.5IQR)
        dataset.query('(@Q1 - 1.5*@IQR) <= {} <= (@Q3 + 1.5*@IQR)'.format(feature), inplace = True)
        
        del Q1, Q3, IQR # Delete Temporary Variables

del feature # Delete Temporary Variables

###################################################### Chi Squared Test ##############################################

from scipy.stats import chi2
from scipy.stats import chi2_contingency

columns = ['Gender', 'Country', 'self_employed', 'family_history',
           'work_interfere', 'no_employees', 'remote_work',
           'tech_company', 'benefits', 'care_options', 'wellness_program',
           'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
           'phys_health_consequence', 'coworkers', 'supervisor',
           'mental_health_interview', 'phys_health_interview',
           'mental_vs_physical', 'obs_consequence']

### Define Chi-Squared Test Parameters ###

prob = 0.95
alpha = 1.0 - prob

table = pandas.crosstab(dataset['Diagnosis'], dataset['radius mean'], margins = False)
stat, p, dof, expected = chi2_contingency(table)

### Interpret Test statistic and cross check with inference via p-value ###

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

############################################ Feature Importance via Random Forest ####################################

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

features = {}
model = RandomForestClassifier()
model.fit(X, Y)

for a, b in zip(X.columns, model.feature_importances_):
    features[a] = b 

importances = pandas.DataFrame.from_dict(features, orient = 'index').rename(columns = {0: 'Importance'})

importances.sort_values(by = 'Importance').plot(kind = 'bar', rot = 90)

################################################# Drop Irrelevant Features ###########################################

irrelevant_features = ['smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'symmetry_worst', 'fractal_dimension_worst']
X = X.drop(irrelevant_features, axis = 1)
dataset = pandas.concat([Y, X], axis = 1)
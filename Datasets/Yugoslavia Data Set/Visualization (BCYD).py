################################################# Importing dataset ##################################################

import numpy
import pandas
import seaborn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pandas.read_csv('Data/breast-cancer.data', delimiter = ",", header = None, names = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'])
information = pandas.concat([dataset.dtypes, dataset.nunique(dropna = False).sort_values(ascending = False), dataset.isnull().sum().sort_values(ascending = False), (100*dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending = False)], axis = 1, keys = ['Type', 'Unique Values', 'Null Values', 'Null Percentage']) # Null Value Check

### Understanding the distribution of the target variable among classes ###

P, N = dataset['Class'].value_counts()
print('Number of Patients : ', P)
print('Number of Non-Patients : ', N)
ax = seaborn.countplot(dataset['Class'], label = 'Count')

del ax, P, N # Delete Temporary Variables

############################################### Encode Categorical Variable ##########################################

X = pandas.DataFrame()

for feature in dataset.columns :
    
    if feature != 'Class' and feature != 'breast' and feature != 'irradiat' :
    
        temporary = pandas.get_dummies(dataset[feature])
        X = pandas.concat([X, temporary], axis = 1)
        dataset.drop(feature, axis = 1, inplace = True)
        
    else :
        
        LE = LabelEncoder() # LabelEncoder Instance
        
        temporary = dataset[feature]
        temporary = LE.fit_transform(temporary)
        temporary = pandas.DataFrame(data = temporary, columns = ['{}'.format(feature)])
        
        if feature == 'Class' :
            
            Y = pandas.DataFrame(data = temporary)
        
        else :

            X = pandas.concat([X, temporary], axis = 1)
        
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

############################################ Principle Component Analysis ############################################

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
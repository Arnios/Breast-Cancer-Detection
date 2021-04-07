################################################## Importing libraries ################################################

import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

############################################ Stacking Ensemble of Classifiers #########################################

from sklearn.ensemble import StackingClassifier

### Define Stacking Ensemble ###

def get_stacking() :
    
    # Define Level-0 of ensemble with base models
    
    level0 = list()
    
    level0.append(('NB', GaussianNB()))
    level0.append(('LR', LogisticRegression()))
    level0.append(('KNN', KNeighborsClassifier()))
    level0.append(('SVM', SVC()))
    level0.append(('RF', RandomForestClassifier(random_state = 0)))
    
    # Define Level-1 meta learner model
    
    level1 = LogisticRegression()
    
    # Define ensemble
    
    model = StackingClassifier(estimators = level0, final_estimator = level1)
	
    return model

########################################## Comparison of traditional models ###########################################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

accuracy = pandas.DataFrame()
  
for concentration in range(10, 55, 5) :
    
    # Define Parameters
    
    inputs = 25 # Number of features to be fed to the model
    specific_accuracy = pandas.DataFrame() # Dataframe to store accuracy obtained for a particular test set concentration
      
    # Split Data into Train and Test Set
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = concentration/100, shuffle = True, random_state = 0) # Test-Train Split
    X_Train, X_Test = anova(X_Train, Y_Train, X_Test, inputs) # Feed Data To ANOVA (Type all for all features)
    
    # Define Model
    
    # classifier = GaussianNB()
    # classifier = LogisticRegression()
    # classifier = KNeighborsClassifier()
    # classifier = SVC()
    # classifier = RandomForestClassifier(random_state = 0)
    classifier = get_stacking() # Call the Stacking Ensemble Model
    
    # Fit data to classifier and make Predictions
    
    classifier.fit(X_Train, Y_Train) # Fit data into model
    Y_Prediction = classifier.predict(X_Test) # Test Prediction ability on test set
    specific_accuracy = specific_accuracy.append({'{}%'.format(concentration) : accuracy_score(Y_Test, Y_Prediction)}, ignore_index = True)
    accuracy = pandas.concat([accuracy, specific_accuracy], axis = 1) # Concat to Global Dataframe
    
    # Deleting Temporary Variables
    
    del classifier, specific_accuracy
          
# Transfer Prediction Accuracy Data Into CSV File

accuracy.to_csv('Results.csv')
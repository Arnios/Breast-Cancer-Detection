################################################## Importing libraries ################################################

import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

####################################################### Data Preprocess ###############################################

Y = LabelEncoder().fit_transform(Y) # Enconding the categorical dependant variable

X = numpy.asarray(X).astype('float32')
Y = numpy.asarray(Y).astype('float32')

########################################## Comparison of traditional models ###########################################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

final_acc = pandas.DataFrame()
  
for j in range(0, 1) :
    
    interim_acc = pandas.DataFrame()
    
    for i in range(10, 55, 5) :
        
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = i/100, shuffle = True, random_state = 0) # Test-Train Split
        
        # classifier = GaussianNB() # Naive Bayes    
        classifier = LogisticRegression(random_state = 0) # Logistic Regression
        # classifier = KNeighborsClassifier() # K-Nearest Neighbors
        # classifier = SVC(random_state = 0) # Support Vector Machine
        # classifier = RandomForestClassifier(random_state = 0) # Random Forest
        
        X_Train_fs, X_Test_fs = anova(X_Train, Y_Train, X_Test, 15)
        
        classifier.fit(X_Train_fs, Y_Train)
        Y_Prediction = classifier.predict(X_Test_fs)

        interim_acc = interim_acc.append({'Iteration - {}'.format(j+1) : accuracy_score(Y_Test, Y_Prediction)}, ignore_index = True)
        
        # Deleting Temporary Variables
        
        del classifier
        del X_Train, X_Test, Y_Train, Y_Test, Y_Prediction
    
    final_acc = pandas.concat([final_acc, interim_acc], axis = 1)
    del i, interim_acc

del j

final_acc['Median'] = final_acc.median(axis = 1)           
final_acc.to_csv('Validation Accuracy.csv')
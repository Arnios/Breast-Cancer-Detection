######################################################### ANOVA F-Test ################################################

import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

### Define ANOVA F-Test ###

def select_features(X_Train, Y_Train, X_Test, num) :
    
    fs = SelectKBest(score_func = f_classif, k = num) # Select number of features and assign F-Test
    fs.fit(X_Train, Y_Train) # Learn relationship from training data
    
    X_Train = fs.transform(X_Train) # Transform train input data
    X_Test = fs.transform(X_Test) # Transform test input data
    
    ### Plot Feature Scores (Larger the better) ###
    
    fs.scores_.sort() # Sorting the F-score list in ascending order
    
    plt.figure(1, figsize = (18, 18))
    plt.bar([x for x in range(len(fs.scores_))], fs.scores_)
    plt.xlabel('Number of components', fontsize = 30, labelpad = 25)
    plt.ylabel('ANOVA F-Score', fontsize = 30, labelpad = 25)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 25)
    plt.show()
    
    return X_Train, X_Test

### Call ANOVA F-Test ###

def anova(X_Train, Y_Train, X_Test, num) :
    
    X_Train, X_Test = select_features(X_Train, Y_Train, X_Test, num) # Feed data into feature selection function
    
    return X_Train, X_Test
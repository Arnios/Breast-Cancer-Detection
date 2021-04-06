################################################# Principal Component Analysis ##########################################

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def pca(X_Train, X_Test, num) :
    
    pca = PCA(n_components = num)

    X_Train = pca.fit_transform(X_Train) 
    X_Test = pca.fit_transform(X_Test)

    ### Plot ###
    
    plt.figure(1, figsize = (18, 18))
    plt.plot(pca.explained_variance_ratio_, linewidth = 2, label = '{}%'.format(i))
    plt.xlabel('Number of components', fontsize = 30, labelpad = 25)
    plt.ylabel('Explained Variance Ratio', fontsize = 30, labelpad = 25)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 25)
    plt.legend(loc = 'upper right', prop = {'size': 30})
    
    ### Delete Temporary Variables ###
    
    del pca
    
    return X_Train, X_Test, plt
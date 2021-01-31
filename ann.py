##### Importing the libraries #####

import keras
import numpy
import pandas
import tensorflow
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

##### Importing data and splitting into test and train set #####

dataset = pandas.read_csv('data.csv')

features = ['id', 'diagnosis', 'radius_mean', 'Unnamed: 32', 'radius_worst', 'perimeter_mean', 'area_mean', 'area_worst', 'radius_se', 'perimeter_se', 'radius_worst']

X = dataset.drop(features, axis = 1) # Independent variables
Y = dataset['diagnosis'] # Dependant variables

X = StandardScaler().fit_transform(X) # Feature scaling
Y = LabelEncoder().fit_transform(Y) # Enconding the categorical dependant variable

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.20, random_state = 0) # Test-Train Split
X_Train = PCA(n_components = 3).fit_transform(X_Train) 
X_Test = PCA(n_components = 3).fit_transform(X_Test)

##### Artifical Neural Network #####

classifier = Sequential()
classifier.add(Dense(2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3)) # First hidden layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')) # Output layer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

##### Cross Validation of Hyperparameters #####

score = pandas.DataFrame(columns = ['Batch Size', 'Number of Epochs', 'AUC Score'])

for i in range(11) :
    for j in [10, 20, 30, 40, 50]:
    
        classifier.fit(X_Train, Y_Train, batch_size = i, epochs = j)
        fpr_classifier, tpr_classifier, thresholds_classifier = roc_curve(Y_Test, classifier.predict(X_Test), pos_label = 1) # ROC curve for the Artifical Neural Network
        auc_score_classifier = roc_auc_score(Y_Test, classifier.predict(X_Test)) # AUC Score for Artifical Neural Network
        score = score.append({'Batch Size' : i, 'Number of Epochs' : j, 'AUC Score' : auc_score_classifier}, ignore_index = True)

print('Maximum AUC Score = %.4f' %score['AUC Score'].max())
index = score.index[score['AUC Score'] == score['AUC Score'].max()]
print('Optimum batch size = %d' %score['Batch Size'].loc[index])
print('Optimum number of epochs : %d' %score['Number of Epochs'].loc[index])

##### Performance Evaluation with AUC-ROC Method #####

random_probabilities = [0 for i in range(len(Y_Test))]
fpr_noskill, tpr_noskill, _ = roc_curve(Y_Test, random_probabilities, pos_label = 1) # ROC curve for no skill approach

fpr_ANN, tpr_ANN, thresholds_ANN = roc_curve(Y_Test, classifier.predict(X_Test), pos_label = 1) # ROC curve for the Artifical Neural Network
auc_score_ANN = roc_auc_score(Y_Test, classifier.predict(X_Test)) # AUC Score for Artifical Neural Network
print('AUC Score For Neural Network Model : %.3f' %auc_score_ANN)

fpr_LRM, tpr_LRM, thresholds_LRM = roc_curve(Y_Test, LRM.predict(X_Test), pos_label = 1) # ROC curve for Logistic Regression Model
auc_score_LRM = roc_auc_score(Y_Test, LRM.predict(X_Test)) # AUC Score for Logistic Regression
print('AUC Score For Logistic Regression Model : %.3f' %auc_score_LRM)

fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(Y_Test, SVM.predict(X_Test), pos_label = 1) # ROC curve for Support Vector Machines
auc_score_SVM = roc_auc_score(Y_Test, SVM.predict(X_Test)) # AUC Score for Support Vector Machines
print('AUC Score For SVM Model : %.3f' %auc_score_SVM)

fpr_RF, tpr_RF, thresholds_RF = roc_curve(Y_Test, RF.predict(X_Test), pos_label = 1) # ROC curve for Random Forest
auc_score_RF = roc_auc_score(Y_Test, RF.predict(X_Test)) # AUC Score for Random Forest
print('AUC Score For Random Forest Model : %.3f' %auc_score_RF)

fpr_NB, tpr_NB, thresholds_NB = roc_curve(Y_Test, NB.predict(X_Test), pos_label = 1) # ROC curve for Naive Bayes Model
auc_score_NB = roc_auc_score(Y_Test, NB.predict(X_Test)) # AUC Score for Naive Bayes
print('AUC Score For Naive Bayes Model : %.3f' %auc_score_NB)

###### Plot Result of AUC-ROC Curve #####

plt.title('AUC-ROC Curve')

plt.plot(fpr_noskill, tpr_noskill, linestyle = '--', color = 'blue', label = 'No Skill')
plt.plot(fpr_ANN, tpr_ANN, linestyle = '--', color = 'orange', label = 'Neural Network')
plt.plot(fpr_LRM, tpr_LRM, linestyle = '--', color = 'brown', label = 'Logistic Regression')
plt.plot(fpr_SVM, tpr_SVM, linestyle = '--', color = 'olive', label = 'SVM')
plt.plot(fpr_RF, tpr_RF, linestyle = '--', color = 'lime', label = 'Random Forest')
plt.plot(fpr_NB, tpr_NB, linestyle = '--', color = 'magenta', label = 'Naive Bayes')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()

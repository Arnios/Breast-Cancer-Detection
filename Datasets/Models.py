############################################### Importing the libraries ###############################################

import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

################################################### Importing data ###################################################

dataset = pandas.read_csv('data.csv')
dropped_features = ['id', 'diagnosis', 'Unnamed: 32', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'symmetry_worst', 'fractal_dimension_worst']

X = dataset.drop(dropped_features, axis = 1) # Independent variables
X = (X - X.min()) / (X.max() - X.min()) # Normalization of features

Y = dataset['diagnosis'] # Dependant variables
Y = LabelEncoder().fit_transform(Y) # Enconding the categorical dependant variable

del dropped_features

########################################## Comparison of traditional models ###########################################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

test_set_concentration = [10, 15, 20, 25, 30, 35, 40, 45, 50]
final_acc = pandas.DataFrame()
  
for j in range(0, 100) :
    
    interim_acc = pandas.DataFrame()
    
    for i in test_set_concentration :
        
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = i/100, shuffle = True, random_state = 0) # Test-Train Split
        
        # classifier = GaussianNB() # Naive Bayes    
        # classifier = LogisticRegression(random_state = 0) # Logistic Regression
        # classifier = KNeighborsClassifier() # K-Nearest Neighbors
        # classifier = SVC(random_state = 0) # Support Vector Machine
        # classifier = RandomForestClassifier(random_state = 0) # Random Forest
        
        classifier.fit(X_Train, Y_Train)
        Y_Prediction = classifier.predict(X_Test)

        interim_acc = interim_acc.append({'Iteration - {}'.format(j+1) : accuracy_score(Y_Test, Y_Prediction)}, ignore_index = True)
        
        # Deleting Temporary Variables
        
        del classifier
        del X_Train, X_Test, Y_Train, Y_Test, Y_Prediction
    
    final_acc = pandas.concat([final_acc, interim_acc], axis = 1)
    del i, interim_acc

del j

final_acc['Median'] = final_acc.median(axis = 1)           
final_acc.to_csv('Validation Accuracy.csv')

################################################# ANN Model (Single Layer) ###########################################

import keras
import tensorflow
import matplotlib.pyplot as plt

from keras import callbacks
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# X = numpy.asarray(X).astype('float32')
# Y = numpy.asarray(Y).astype('float32')

test_set_concentration = [10]
accuracy_history = pandas.DataFrame()

for i in test_set_concentration :
    
    es = callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = False, verbose = 1)
    
    for j in range(0, 1) :
        
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = i/100, shuffle = True, random_state = 0) # Test-Train Split
        
        classifier = Sequential()
        classifier.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9)) # First hidden layer
        classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')) # Output layer
        classifier.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = 'accuracy')
        
        model_data = classifier.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 1000, verbose = 1, callbacks = [es])
    
        very_temporary_history = pandas.DataFrame()
    
        temporary_history = pandas.DataFrame(model_data.history['val_accuracy'], columns = ['Accuracy'])
        very_temporary_history = very_temporary_history.append({'I-{}'.format(j + 1) : temporary_history['Accuracy'].iloc[-1]}, ignore_index = True)
        accuracy_history = pandas.concat([accuracy_history, very_temporary_history], axis = 1)
        
        # Loss Plot
    
        training_loss = model_data.history['loss']
        test_loss = model_data.history['val_loss']
        epoch_count = range(1, len(training_loss) + 1)
    
        plt.plot(epoch_count, training_loss, '--', color = 'red')
        plt.plot(epoch_count, test_loss, '--', color = 'blue')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Number of Epoch')
        plt.ylabel('Validation Loss')
        plt.show()
        
        # Deleting temporary variables
        
        del X_Test, X_Train, Y_Test, Y_Train, classifier, model_data
        del temporary_history, very_temporary_history
        del training_loss, test_loss, epoch_count
      
    # Print the result in a CSV File
    
    accuracy_history['Median'] = accuracy_history.median(axis = 1)
    accuracy_history.to_csv('Result - {}%.csv'.format(i))

##################################### Performance Evaluation with AUC-ROC Method ######################################

# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score

# random_probabilities = [0 for i in range(len(Y_Test))]
# fpr_noskill, tpr_noskill, _ = roc_curve(Y_Test, random_probabilities, pos_label = 1) # ROC curve for no skill approach

# fpr_classifier_1, tpr_classifier_1, thresholds_classifier_1 = roc_curve(Y_Test, classifier_1.predict(X_Test), pos_label = 1) # ROC curve for the Artifical Neural Network
# auc_score_classifier_1 = roc_auc_score(Y_Test, classifier_1.predict(X_Test)) # AUC Score for Artifical Neural Network
# print('AUC Score of model : %.4f' %auc_score_classifier_1) # AUC Score = 98.29%

# # Plot

# plt.title('AUC-ROC Curve')

# plt.plot(fpr_noskill, tpr_noskill, linestyle = '--', color = 'blue', label = 'No Skill')
# plt.plot(fpr_classifier_1, tpr_classifier_1, linestyle = '--', color = 'red', label = 'Model')

# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc = 'lower right')
# plt.show()
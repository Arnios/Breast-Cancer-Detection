############################################### Importing the libraries ###############################################

import keras
import numpy
import pandas
import tensorflow
import matplotlib.pyplot as plt

from keras import callbacks
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

################################ Importing data and splitting into train and test set #################################

dataset = pandas.read_csv('data.csv')

features = ['id', 'diagnosis', 'radius_mean', 'Unnamed: 32', 'radius_worst', 'perimeter_mean', 'area_mean', 'area_worst', 'radius_se', 'perimeter_se', 'radius_worst']

X = dataset.drop(features, axis = 1) # Independent variables
Y = dataset['diagnosis'] # Dependant variables

X = StandardScaler().fit_transform(X) # Feature scaling
Y = LabelEncoder().fit_transform(Y) # Enconding the categorical dependant variable

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.20, random_state = 0) # Test-Train Split

X_Train = PCA(n_components = 5).fit_transform(X_Train) # Principal Component Analysis on Training Set
X_Test = PCA(n_components = 5).fit_transform(X_Test) # Principal Component Analysis on Test Set

######################################################## Models ########################################################

classifier_1 = Sequential()
classifier_1.add(Dense(3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5)) # First hidden layer
classifier_1.add(Dense(1, kernel_initializer = 'uniform', activation = 'relu')) # Output layer
classifier_1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')

classifier_2 = Sequential()
classifier_2.add(Dense(3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5)) # First hidden layer
classifier_2.add(Dense(2, kernel_initializer = 'uniform', activation = 'relu')) # Second hidden layer
classifier_2.add(Dense(1, kernel_initializer = 'uniform', activation = 'relu')) # Output layer
classifier_2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')

#################################################### Ablation Study ##################################################

earlystopping = callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5, restore_best_weights = True, verbose = 1)

classifier_1.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 50, callbacks = [earlystopping]) # Optimum epoch - 10 Validation Accuracy - 94.74%
classifier_2.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 50, callbacks = [earlystopping]) # Optimum epoch - 20 Validation Accuracy - 94.74%

history = classifier_1.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 50) # Validation Accuracy - 94.74%
history = classifier_2.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 50) # Validation Accuracy - 92.98%

training_loss = history.history['loss']
test_loss = history.history['val_loss']

epoch_count = range(1, len(training_loss) + 1) # Create count of the number of epochs

# Visualize loss history

plt.plot(epoch_count, training_loss, '--', color = 'red')
plt.plot(epoch_count, test_loss, 'x', color = 'blue')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Number of Epoch')
plt.ylabel('Validation Loss')
plt.show()

##################################### Performance Evaluation with AUC-ROC Method ######################################

classifier_1.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 10) # Validation Accuracy - 94.74%
classifier_2.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 20) # Validation Accuracy - 92.98%

random_probabilities = [0 for i in range(len(Y_Test))]
fpr_noskill, tpr_noskill, _ = roc_curve(Y_Test, random_probabilities, pos_label = 1) # ROC curve for no skill approach

fpr_classifier_1, tpr_classifier_1, thresholds_classifier_1 = roc_curve(Y_Test, classifier_1.predict(X_Test), pos_label = 1) # ROC curve for the Artifical Neural Network
auc_score_classifier_1 = roc_auc_score(Y_Test, classifier_1.predict(X_Test)) # AUC Score for Artifical Neural Network

fpr_classifier_2, tpr_classifier_2, thresholds_classifier_2 = roc_curve(Y_Test, classifier_2.predict(X_Test), pos_label = 1) # ROC curve for the Artifical Neural Network
auc_score_classifier_2 = roc_auc_score(Y_Test, classifier_2.predict(X_Test)) # AUC Score for Artifical Neural Network

print('AUC Score of 2-Layer ANN : %.4f' %auc_score_classifier_1) # AUC Score = 98.29%
print('AUC Score of 3-Layer ANN : %.4f' %auc_score_classifier_2) # AUC Score = 98.02%

############################################## Plot Result of AUC-ROC Curve ###########################################

plt.title('AUC-ROC Curve')

plt.plot(fpr_noskill, tpr_noskill, linestyle = '--', color = 'blue', label = 'No Skill')
plt.plot(fpr_classifier_1, tpr_classifier_1, linestyle = '--', color = 'red', label = '2-Layer ANN')
plt.plot(fpr_classifier_2, tpr_classifier_2, linestyle = '--', color = 'green', label = '3-Layer ANN')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()

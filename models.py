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

################################################### Importing data ###################################################

dataset = pandas.read_csv('data.csv')

dropped_features = ['id', 'diagnosis', 'Unnamed: 32', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'symmetry_worst', 'fractal_dimension_worst']

Y = dataset['diagnosis'] # Dependant variables
X = dataset.drop(dropped_features, axis = 1) # Independent variables

X = (X - X.min()) / (X.max() - X.min()) # Normalization of features
Y = LabelEncoder().fit_transform(Y) # Enconding the categorical dependant variable

########################################## Comparison of traditional models ###########################################

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

test_set_concentration = [10, 15, 20, 25, 30, 35, 40, 45, 50]

for i in test_set_concentration :
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = i, random_state = 0) # Test-Train Split
    
    # classifier = GaussianNB() # Naive Bayes
    classifier = LogisticRegression(random_state = 0) # Logistic Regression
    # classifier = KNeighborsClassifier() # K-Nearest Neighbors
    # classifier = SVC(random_state = 0) # Support Vector Machine
    # classifier = RandomForestClassifier(random_state = 0) # Random Forest
    
    classifier.fit(X_Train, Y_Train)
    
    Y_Prediction = classifier.predict(X_Test)
    print('{}% test set concentration accuracy : {:.2f}'.format(i, 100*accuracy_score(Y_Test, Y_Prediction)))

###################################################### ANN Model ######################################################

earlystopping = callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5, restore_best_weights = True, verbose = 1)

classifier_1 = Sequential()
classifier_1.add(Dense(3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5)) # First hidden layer
classifier_1.add(Dense(1, kernel_initializer = 'uniform', activation = 'relu')) # Output layer
classifier_1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')
model_data = classifier_1.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 50) # Validation Accuracy - 94.74%

training_loss = model_data.history['loss']
test_loss = model_data.history['val_loss']

epoch_count = range(1, len(training_loss) + 1) # Create count of the number of epochs

# Plot

plt.plot(epoch_count, training_loss, '--', color = 'red')
plt.plot(epoch_count, test_loss, '--', color = 'blue')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Number of Epoch')
plt.ylabel('Validation Loss')
plt.show()

##################################### Performance Evaluation with AUC-ROC Method ######################################

random_probabilities = [0 for i in range(len(Y_Test))]
fpr_noskill, tpr_noskill, _ = roc_curve(Y_Test, random_probabilities, pos_label = 1) # ROC curve for no skill approach

fpr_classifier_1, tpr_classifier_1, thresholds_classifier_1 = roc_curve(Y_Test, classifier_1.predict(X_Test), pos_label = 1) # ROC curve for the Artifical Neural Network
auc_score_classifier_1 = roc_auc_score(Y_Test, classifier_1.predict(X_Test)) # AUC Score for Artifical Neural Network
print('AUC Score of model : %.4f' %auc_score_classifier_1) # AUC Score = 98.29%

############################################## Plot Result of AUC-ROC Curve ###########################################

plt.title('AUC-ROC Curve')

plt.plot(fpr_noskill, tpr_noskill, linestyle = '--', color = 'blue', label = 'No Skill')
plt.plot(fpr_classifier_1, tpr_classifier_1, linestyle = '--', color = 'red', label = 'Model')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
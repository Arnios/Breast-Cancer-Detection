################################################## Importing libraries ###############################################

import math
import keras
import tensorflow
import matplotlib.pyplot as plt

from keras import callbacks
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

######################################################### ANN Model ###################################################

### Define Learning Rate Schedule Function ###

def step_decay(epoch) :
    
    ### Define Hyperparameters ###
    
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10
    
	lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch)/epochs_drop)) # Equation to calculate new Learning Rate
    
	return lrate

### Defining the Model ###

accuracy_history = pandas.DataFrame() # Store Final Accuracy Data

for i in range(10, 55, 5) :
    
    ### Define Callbacks ###
    
    es = callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = False, verbose = 1)
    
    for j in range(0, 1) :
        
        ### Define Hyperparameters ###
        
        inputs = 25
        
        # opt = SGD() # Default SGD with Learning Rate = 0.01
        # opt = SGD(learning_rate = 0.001) # Learning Rate = 0.001
        opt = SGD(learning_rate = 0.0) # For use in Adaptive Learning Rate Model and Learning Rate = 0.0 so that the LR defined is not used in the model
        
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = i/100, shuffle = True, random_state = 0) # Test-Train Split
        X_Train, X_Test = anova(X_Train, Y_Train, X_Test, inputs) # Calling ANOVA (Type 'all' for all features)
        
        ### Neural Network ###
       
        classifier = Sequential()
        classifier.add(Dense(inputs, kernel_initializer = 'uniform', activation = 'relu', input_dim = inputs)) # First hidden layer
        classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')) # Output layer
        
        classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = 'accuracy')
        
        ### Fit Neural Network To Data ###
        
        # Default Model
        
        # model_data = classifier.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 1000, verbose = 1, callbacks = [es])
        
        # Adaptive Model
        
        lrate = LearningRateScheduler(step_decay)
        model_data = classifier.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 1000, verbose = 1, callbacks = [es, lrate])
        
        ### Store History ###

        very_temporary_history = pandas.DataFrame()
        temporary_history = pandas.DataFrame(model_data.history['val_accuracy'], columns = ['Accuracy'])
        very_temporary_history = very_temporary_history.append({'I-{}'.format(j + 1) : temporary_history['Accuracy'].iloc[-1]}, ignore_index = True)
        accuracy_history = pandas.concat([accuracy_history, very_temporary_history], axis = 1)
        
        ### Loss Plot ###
    
        training_loss = model_data.history['loss']
        test_loss = model_data.history['val_loss']
        epoch_count = range(1, len(training_loss) + 1)
    
        plt.plot(epoch_count, training_loss, '--', color = 'red')
        plt.plot(epoch_count, test_loss, '--', color = 'blue')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Number of Epoch')
        plt.ylabel('Validation Loss')
        plt.show()
        
        ### Deleting temporary variables ###
        
        # del X_Test, X_Train, Y_Test, Y_Train, classifier, model_data
        # del temporary_history, very_temporary_history
        # del training_loss, test_loss, epoch_count
      
    ### Print the result in a CSV File ###
    
    accuracy_history['Median'] = accuracy_history.median(axis = 1)
    # accuracy_history.to_csv('Result - {}%.csv'.format(i))

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
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
    
    initial_lrate = 0.1 # Setting an initially high learning rate
    epochs_drop = 10
    drop = 0.5 # Learning rate Will drop by half
    
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch)/epochs_drop)) # Equation to calculate new Learning Rate
    
    return lrate

### Defining the Model ###

accuracy_summary = pandas.DataFrame()

min_concentration = 10 # Minimum TEST SET Concentration
max_concentration = 50 # Maximum TEST SET Concentration

for i in range(min_concentration, max_concentration + 5, 5) : # Looping through the different test set concentrations
    
    ### Define Parameters and Callbacks ###
    
    inputs = 25
    
    # opt = SGD() # Default SGD with Learning Rate = 0.01
    opt = SGD(learning_rate = 0.0) # To be used in adaptive model
    
    es = callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = False, verbose = 1)
    
    for j in range(0, 5) : # Loop for number of iterations to be performed
    
        accuracy_history = pandas.DataFrame() # Store Final Accuracy Data
        very_temporary_history = pandas.DataFrame() # Extracting the validation accuracy obtained the very last epoch
    
        ### Split Data ###
        
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = i/100, shuffle = True, random_state = 0) # Test-Train Split
        X_Train, X_Test = anova(X_Train, Y_Train, X_Test, inputs) # Calling ANOVA (Type 'all' for all features in input)
        
        ### Neural Network ###
       
        classifier = Sequential()
        classifier.add(Dense(inputs, kernel_initializer = 'uniform', activation = 'relu', input_dim = inputs)) # First hidden layer
        classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')) # Output layer
        
        classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = 'accuracy')
        
        ### Feed Data To Neural Network ###
        
        # Constant Learning Rate Model (Default)
        
        # model_data = classifier.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 1, epochs = 1000, verbose = 1, callbacks = [es])
        
        # Adaptive Learning Rate Model (Step Decay)
        
        lrate = LearningRateScheduler(step_decay)
        model_data = classifier.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = 32, epochs = 1000, verbose = 1, callbacks = [es, lrate])
        
        ### Store Model History ###
        
        temporary_history = pandas.DataFrame(model_data.history['val_accuracy'], columns = ['Accuracy']) # Validation accuracy record of every individual epoch
        very_temporary_history = very_temporary_history.append({'I-{}'.format(j + 1) : temporary_history['Accuracy'].iloc[-1]}, ignore_index = True) # Appending with itertaion number
        accuracy_history = pandas.concat([accuracy_history, very_temporary_history], axis = 1) # Appending the final accuracy obtained in an individual iteration
        
        median = accuracy_history.median(axis = 1) # Calculating Median Accuracy
        
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
        
        del X_Test, X_Train, Y_Test, Y_Train
        del classifier, model_data
        del temporary_history, very_temporary_history, accuracy_history
        del training_loss, test_loss, epoch_count
        
    accuracy_summary['{}%'.format(i)] = median # Storing the median accuracy for the specific test set concentration
    
    ### Deleting temporary variables ###
    
    del es, j, median, inputs, opt

### Print the result in a CSV and Deleting temporary variables ###

accuracy_summary.to_csv('Accuracy Summary.csv')
del i, min_concentration, max_concentration
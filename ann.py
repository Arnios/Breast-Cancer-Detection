################################################## Importing libraries ###############################################

import math
import keras
import tensorflow
import matplotlib.pyplot as plt

from keras import callbacks
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

######################################################### ANN Model ###################################################

### Insert Parameters ###

batch_size = 1 # Set Batch Size
min_concentration = 10 # Minimum TEST SET Concentration
max_concentration = 50 # Maximum TEST SET Concentration
num_of_iterations = 100

### Define Learning Rate Schedule Function ###

def step_decay(epoch) :
    
    ### Define Hyperparameters ###
    
    initial_lrate = 0.1 # Setting an initially high learning rate
    epochs_drop = 10
    drop = 0.5 # Learning rate will drop by half
    
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch)/epochs_drop)) # Equation to calculate new Learning Rate
    
    return lrate

### Defining the Model ###

accuracy_summary = pandas.DataFrame()
epoch_summary = pandas.DataFrame()

for i in range(min_concentration, max_concentration + 5, 5) : # Looping through the different test set concentrations
    
    ### Define Parameters and Callbacks ###
    
    inputs = 25
    
    # opt = SGD() # Default Stochastic Gradient Descent
    # opt = SGD(learning_rate = 0.0) # Stochastic Gradient Descent for use in step decay model
    # opt = RMSprop() # Root Mean Square Propagation
    # opt = Adagrad() # Adaptive Gradient Algorithm
    opt = Adam() # Adaptive Moment Estimation
    
    es = callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = False, verbose = 1)
    
    accuracy_history = pandas.DataFrame() # Store Final Accuracy Data
    epoch_history = pandas.DataFrame() # Store Epoch History For a Specific Test Case Concentration
    
    for j in range(0, num_of_iterations) : # Loop for number of iterations to be performed
    
        ### Declaring Variables ###
    
        specific_accuracy_history = pandas.DataFrame() # Extracting the validation accuracy obtained the very last epoch
        specific_epoch_history = pandas.DataFrame() # Store Specific Epoch Data
        
        ### Split Data ###
        
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = i/100, shuffle = True, random_state = 0) # Test-Train Split
        X_Train, X_Test = anova(X_Train, Y_Train, X_Test, inputs) # Calling ANOVA (Type 'all' for all features in input)
        
        ### Build Neural Network ###
       
        classifier = Sequential()
        classifier.add(Dense(inputs, kernel_initializer = 'uniform', activation = 'relu', input_dim = inputs)) # First hidden layer
        classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')) # Output layer
        classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = 'accuracy')
        
        ### Feed Data To Neural Network ###
        
        # Default Model : Constant Learning Rate
        
        model_data = classifier.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = batch_size, epochs = 1000000, verbose = 1, callbacks = [es])
        
        # Adaptive Learning Rate Model : Step Decay
        
        # lrate = LearningRateScheduler(step_decay)
        # model_data = classifier.fit(X_Train, Y_Train, validation_data = (X_Test, Y_Test), batch_size = batch_size, epochs = 1000, verbose = 1, callbacks = [es, lrate])
        
        ### Save model ###
        
    	# filename = 'Model' + '.h5'
    	# classifier.save(filename)
    	# print('Saved %s' % filename)
        
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
        
        ### Store Important Model Data ###
        
        validation_accuracy = pandas.DataFrame(model_data.history['val_accuracy'], columns = ['Accuracy']) # Validation accuracy record of every individual epoch
        
        specific_accuracy_history = specific_accuracy_history.append({'I-{}'.format(j + 1) : validation_accuracy['Accuracy'].iloc[-1]}, ignore_index = True) # Appending with itertaion number
        accuracy_history = pandas.concat([accuracy_history, specific_accuracy_history], axis = 1) # Appending the final accuracy obtained in an individual iteration
        
        specific_epoch_history = specific_epoch_history.append({'I-{}'.format(j + 1) : len(training_loss)}, ignore_index = True) # Appending with itertaion number
        epoch_history = pandas.concat([epoch_history, specific_epoch_history], axis = 1) # Appending the final accuracy obtained in an individual iteration
        
        ### Deleting temporary variables ###
        
        del X_Test, X_Train, Y_Test, Y_Train
        del classifier, model_data, validation_accuracy
        del specific_accuracy_history, specific_epoch_history
        del training_loss, test_loss, epoch_count
        # del lrate
    
    ### Storing Summary of Test Set Concentration ###
    
    median_accuracy = accuracy_history.median(axis = 1) # Calculating Median Accuracy
    accuracy_summary['{}%'.format(i)] = median_accuracy # Storing the median accuracy for the specific test set concentration
    
    median_epoch = epoch_history.median(axis = 1)
    epoch_summary['{}%'.format(i)] = median_epoch
    
    ### Deleting temporary variables ###
    
    del es, j, inputs, opt
    del median_accuracy, median_epoch

### Print the result in a CSV and Deleting temporary variables ###

accuracy_summary.to_csv('Accuracy Summary.csv')
epoch_summary.to_csv('Epoch Summary.csv')

del accuracy_history, epoch_history
del i, min_concentration, max_concentration, num_of_iterations
del batch_size
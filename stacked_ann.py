################################################## Importing libraries ###############################################

from numpy import dstack
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

##################################################### Stacked ANN Model ###############################################

### Load saved models from file ###

def load_all_models(n_models) :
    
	all_models = list() # Create list to store models
    
    # Loading the individual models from stored location
    
	for i in range(n_models) :
		
		filename = 'models/model_' + str(i + 1) + '.h5' # define filename for this ensemble
		model = load_model(filename) # load model from file
		all_models.append(model) # add to list of members
		print('>loaded %s' % filename)
        
	return all_models

### Create input dataset for stacked model as outputs from the ensemble ###

def stacked_dataset(members, inputX) :
    
	stackX = None
    
	for model in members:
		
		yhat = model.predict(inputX, verbose = 0) # make prediction
		
        # stack predictions into [rows, members, probabilities]
        
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))

	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2])) # flatten predictions to [rows, members x probabilities]
	
    return stackX

### Fit stacked model based on the outputs from the ensemble members ###

def fit_stacked_model(members, inputX, inputy) :
    
	stackedX = stacked_dataset(members, inputX) # create dataset using ensemble
	
    # fit standalone model
    
	model = LogisticRegression()
	model.fit(stackedX, inputy)
	
    return model

### Make a prediction with the stacked model ###

def stacked_prediction(members, model, inputX) :
	
	stackedX = stacked_dataset(members, inputX) # create dataset using ensemble
	yhat = model.predict(stackedX) # make a prediction
	
    return yhat

################################################ Call Stacked ANN Model ###############################################

# load all models

n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# evaluate standalone models on test dataset

for model in members :
    
	testy_enc = to_categorical(testy)
	_, acc = model.evaluate(testX, testy_enc, verbose=0)
	print('Model Accuracy: %.3f' % acc)

model = fit_stacked_model(members, testX, testy) # fit stacked model using the ensemble
yhat = stacked_prediction(members, model, testX) # evaluate model on test set
acc = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
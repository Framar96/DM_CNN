import os
import pandas as pd
import tensorflow as tf
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D , MaxPool1D ,Flatten 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

data_dir='/gpfs/home2/marangio/project/H5/'

def get_data(data_dir):
    
    c=0
    data = [] 
    y_rho=[]
    y_gamma=[]
    y_m2=[]
    files = os.listdir(data_dir)
    seed_value = 42
    random.seed(seed_value)
    random.shuffle(files)
    for img in files:
        
        if 'csv' in img:
            string=img.split('_')
            rho=float(string[6])
            gamma=float(string[8])
            m2=float(string[4])
            y_rho.append([rho,c])
            y_gamma.append([gamma,c])
            y_m2.append([m2,c])
            df=pd.read_csv(os.path.join(data_dir, img))
            df=df.T
            data.append(df)
            c+=1
         
    return data,np.array(y_rho),np.array(y_gamma),np.array(y_m2)

x,y_rho,y_gamma,y_m2= get_data(data_dir)

columns = [str(i) for i in range(20000)]
inputdata = pd.concat(x,ignore_index=True)
inputdata.columns = columns
inputs=inputdata[columns].values
inputs=inputs-np.min(inputs)
inputs=inputs/np.max(inputs)
x_data = inputs.reshape((inputs.shape[0], inputs.shape[1],1))




max_rho=max(y_rho[:, 0])
max_gamma=max(y_gamma[:, 0])

scaler = StandardScaler()
#RHO
x_train_rho, x_temp_rho, y_train_rho, y_temp_rho = train_test_split(x_data, y_rho, test_size=0.30, stratify=y_rho[:, 0])    
x_val_rho, x_test_rho, y_val_rho, y_test_rho = train_test_split(x_temp_rho, y_temp_rho, test_size=0.5, stratify=y_temp_rho[:, 0])
y_train_rho_reshaped = y_train_rho[:, 0].reshape(-1, 1)
y_val_rho_reshaped = y_val_rho[:, 0].reshape(-1, 1)
y_train_rho_normalized = scaler.fit_transform(y_train_rho_reshaped)
y_val_rho_normalized =scaler.fit_transform(y_val_rho_reshaped )



def create_model(filters, layersConv, kernel_size ,layersDense,units, lr):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size , activation='relu', input_shape=(20000, 1)))
    model.add(MaxPool1D(2))
    x=2
    for _ in range(layersConv - 1):  # Add more hidden layers if needed
       
        model.add(Conv1D(filters=filters*x, kernel_size=kernel_size , activation='relu'))
        model.add(MaxPool1D(2))
        x*=2
        
    model.add(Flatten())
    c=1
    for _ in range(layersDense - 1):  # Add more hidden layers if needed
       
        model.add(Dense(units/c, activation='relu'))
        c*=2
        
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    
    optimizer = Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

# Create KerasRegressor wrapper for use in GridSearchCV
model = KerasRegressor(build_fn=create_model, epochs=5, verbose=0)

# Define hyperparameters to tune
layersConv = [1,2, 3]  # Number of dense layers
kernel_size = [2, 5, 10] 
layersDense = [1, 2, 3, 4] 
# Number of units in each layer
filters= [16,32]
units = [256,512]  # Number of units in each layer
lr = [0.0001]  # Learning rates
batch_size = [32]  # Batch sizes

# Create the parameter grid
param_grid = dict(filters=filters, kernel_size=kernel_size , layersConv=layersConv, layersDense=layersDense, units=units, lr=lr, batch_size=batch_size)

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_result = grid.fit(x_train_rho,y_train_rho_normalized, validation_data=(x_val_rho, y_val_rho_normalized ))

# Print best parameters and corresponding mean test score
best_model = grid_result.best_estimator_
y_pred_normalized = best_model.predict(x_test_rho)
y_pred=scaler.inverse_transform(y_pred_normalized.reshape(-1, 1))

r2_test = r2_score(y_test_rho[:, 0], y_pred)
print("Best Parameters: ", grid_result.best_params_)
print("Best R2 Score on Test Set: ", r2_test)
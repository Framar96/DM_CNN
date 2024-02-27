
#PYTHON LIBRARIES IMPORT
import os
import pandas as pd
import tensorflow as tf
import random
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D , MaxPool1D , Flatten 
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers

#SEED INITIATOR
np.random.seed(42)
tf.random.set_seed(42)

#GPU CHECK
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#DIRECTORIES
home_dir='/gpfs/home2/marangio/project/'
data_dir='/gpfs/home2/marangio/project/H5/'

def get_data(data_dir):  #Takes in directory path and returns x and y values
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

RHO_train=True
GAMMA_train=True
M2_train=True
CNN=True

#DATA MANIPULATION AND RESHAPING
x,y_rho,y_gamma,y_m2= get_data(data_dir)
columns = [str(i) for i in range(20000)]
inputdata = pd.concat(x,ignore_index=True)
inputdata.columns = columns
inputs=inputdata[columns].values
inputs=inputs-np.min(inputs)
inputs=inputs/np.max(inputs)

if CNN==True:
    
    x_data = inputs.reshape((inputs.shape[0], inputs.shape[1],1))
else:
     x_data = inputs.reshape((inputs.shape[0], inputs.shape[1]))
    

max_rho=max(y_rho[:, 0])
max_gamma=max(y_gamma[:, 0])
max_m2=max(y_m2[:, 0])


#CNN MODEL
def CNN1D(shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(shape, 1)))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

lr=0.0001
batch_size=32
epochs=20
integer_epochs = list(range(1,epochs+1))

#RHO PARAMETER
if RHO_train==True:
    
    scaler = StandardScaler()
    x_train_rho, x_temp_rho, y_train_rho, y_temp_rho = train_test_split(x_data, y_rho, test_size=0.30, stratify=y_rho[:, 0],random_state=42)
    x_val_rho, x_test_rho, y_val_rho, y_test_rho = train_test_split(x_temp_rho, y_temp_rho, test_size=0.5, stratify=y_temp_rho[:, 0],random_state=42)
    y_train_rho_reshaped = y_train_rho[:, 0].reshape(-1, 1)
    y_val_rho_reshaped = y_val_rho[:, 0].reshape(-1, 1)
    y_train_rho_normalized = scaler.fit_transform(y_train_rho_reshaped)
    y_val_rho_normalized =scaler.fit_transform(y_val_rho_reshaped )


    model=CNN1D(x_data.shape[1])
    opt = Adam(lr=lr)
    checkpoint = ModelCheckpoint(home_dir+'best_weights_rho_strain_H5.h5',monitor='val_loss',verbose=1, save_best_only=True,mode='min' )
    model.compile(optimizer = opt , loss = "mean_squared_error" , metrics = ['mse'])
    history = model.fit(x_train_rho, y_train_rho_normalized, epochs=epochs, batch_size=batch_size, validation_data=(x_val_rho,y_val_rho_normalized) , callbacks=[checkpoint], shuffle= True, verbose=1)
    print("Finished training")

    val_loss=history.history['val_loss']
    loss=history.history['loss']

    plt.figure(figsize=(10,6))
    plt.plot(integer_epochs,val_loss, 'g', label='Validation Loss')
    plt.plot(integer_epochs,loss, 'b', label='Loss')
    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('MSE',fontsize=15)
    plt.title('Training ρ6 LR='+str(lr)+' Batch size='+str(batch_size))
    plt.grid()
    plt.legend(fontsize=15)
    plt.xticks(integer_epochs,fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(home_dir+'Loss function Rho inference CNN1D frequencyLR='+str(lr)+' Batch size='+str(batch_size)+'.png')
    plt.show()
    plt.close()

    model.load_weights(home_dir+'best_weights_rho_strain_H5.h5')
    y_predict_rho_normalized= model.predict(x_test_rho)
    y_predict_rho=scaler.inverse_transform( y_predict_rho_normalized)
   
    mse = mean_squared_error(y_test_rho[:, 0], y_predict_rho)
    mae = mean_absolute_error(y_test_rho[:, 0], y_predict_rho)
    r2 = r2_score(y_test_rho[:, 0], y_predict_rho)
    rmse= np.sqrt(mse)
    std=np.std(y_predict_rho)
    mean=np.mean(y_predict_rho)
    print("The accuracy of our model is"+str(r2 *100))
    print("The Mean Absolute Error of our Model is"+str(mae))
    print("The RMSE  of our Model is "+str(rmse))
    print("The STD of our Model is"+str(std))
    print( "The Mean of our predictions is"+str(mean))

    index_test_rho =y_test_rho[:, 1]
    list_gamma=[]
    list_m2=[]
    for i in range(len(y_test_rho)):
        index_rho=index_test_rho[i]
        index_gamma= np.where(y_gamma[:, 1] == index_rho)[0]
        index_m2= np.where(y_m2[:, 1] == index_rho)[0]
        m2=y_m2[index_m2, 0]
        gamma= y_gamma[index_gamma, 0]
        list_gamma.append(gamma)
        list_m2.append(m2)

    values_gamma=np.array(list_gamma)
    flattened_values_gamma = np.concatenate(values_gamma).ravel()
    values_m2=np.array(list_m2)
    flattened_values_m2 = np.concatenate(values_m2).ravel()

    plt.figure(figsize=(8,6))
    plt.scatter(y_predict_rho,y_test_rho[:, 0],c=flattened_values_gamma, cmap='viridis', s=50)
    plt.plot(y_test_rho[:, 0], y_test_rho[:, 0], color = 'red', label = 'x=y')
    cbar=plt.colorbar()
    cbar.set_label('γs')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xscale('log')  # Set x-axis scale to logarithmic
    plt.yscale('log')
    plt.xlabel('Predicted ρ6 [$M☉/pc^3$]',fontsize=15)
    plt.ylabel('Test ρ6 [$M☉/pc^3$]',fontsize=15)
    plt.title("Parameter estimation ρ6")
    plt.legend(fontsize=15)
    plt.savefig(home_dir+'Parameter estimation Rho with Gamma CNN1D LR='+str(lr)+' Batch size='+str(batch_size)+'.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(8,6))
    plt.scatter(y_predict_rho,y_test_rho[:, 0],c=np.log10(flattened_values_m2), cmap='viridis', s=50)
    plt.plot(y_test_rho[:, 0], y_test_rho[:, 0], color = 'red', label = 'x=y')
    cbar=plt.colorbar()
    cbar.set_label('$\log_{10}q$')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xscale('log')  # Set x-axis scale to logarithmic
    plt.yscale('log')
    plt.xlabel('Predicted ρ6 [$M☉/pc^3$]',fontsize=15)
    plt.ylabel('Test ρ6 [$M☉/pc^3$]',fontsize=15)
    plt.title("Parameter estimation ρ6")
    plt.legend(fontsize=15)
    plt.savefig(home_dir+'Parameter estimation Rho with M2 CNN1D.png')
    plt.show()
    plt.close()
     

    accuracy_percentage = []
    test_values_predictions = []

    for test_value in np.unique(y_test_rho[:, 0]):  # Loop through unique test values
        if test_value==1.46779927e+13:
            test_values_predictions.append(test_value)

            # Find indices corresponding to the current test value
            indices = np.where(y_test_rho[:, 0] == test_value)[0]

            # Gather predictions associated with the current test value
            individual_predictions = [y_predict_rho[i] for i in indices]
            mse = mean_squared_error(np.repeat(test_value, len(individual_predictions)), individual_predictions)
            rmse= np.sqrt(mse)
            std=np.std(individual_predictions)
            mean=np.mean(individual_predictions)
            accuracy=rmse/test_value
            accuracy_percentage.append(accuracy)
            print("STD" +str(std))
            print("Mean"+str(mean))
            print("RMSE"+str(rmse))

    # Plotting Accuracy for each test value
    plt.figure(figsize=(10, 6))
    plt.scatter(test_values_predictions,  accuracy_percentage, color='red')

    # Connecting the dots (lines between points)
    plt.plot(test_values_predictions, accuracy_percentage, linestyle='-', color='red')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Test ρ6 [$M☉/pc^3$]',fontsize=15)
    plt.ylabel('Accuracy',fontsize=15)
    plt.grid(axis='y')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(home_dir + 'Accuracy_Rho.png')
    plt.show()

#GAMMA PARAMETER
if GAMMA_train==True:
    
    scaler = StandardScaler()
    x_train_gamma, x_temp_gamma, y_train_gamma, y_temp_gamma = train_test_split(x_data, y_gamma, test_size=0.30, stratify=y_gamma[:, 0],random_state=42)
    x_val_gamma, x_test_gamma, y_val_gamma, y_test_gamma = train_test_split(x_temp_gamma, y_temp_gamma, test_size=0.5, stratify=y_temp_gamma[:, 0],random_state=42) 
    y_train_gamma_reshaped = y_train_gamma[:, 0].reshape(-1, 1)
    y_val_gamma_reshaped = y_val_gamma[:, 0].reshape(-1, 1)
    y_train_gamma_normalized = scaler.fit_transform(y_train_gamma_reshaped )
    y_val_gamma_normalized =scaler.fit_transform(y_val_gamma_reshaped)
    

    model=CNN1D(x_data.shape[1])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                          patience=5, min_lr=0.00001)
    opt = Adam(lr=lr)
    checkpoint = ModelCheckpoint(home_dir+'best_weights_gamma_strain_H5.h5',monitor='val_loss',verbose=1, save_best_only=True,mode='min' )
    model.compile(optimizer = opt , loss = "mean_squared_error" , metrics = ['mse'])
    history = model.fit(x_train_gamma, y_train_gamma_normalized, epochs=epochs, batch_size=batch_size, validation_data=( x_val_gamma,y_val_gamma_normalized) ,callbacks=[checkpoint,reduce_lr ] ,shuffle= True, verbose=1)
    
    val_loss=history.history['val_loss']
    loss=history.history['loss']
    plt.figure(figsize=(10,6))
    plt.plot(integer_epochs,val_loss, 'g', label='Validation Loss')
    plt.plot(integer_epochs,loss, 'b', label='Loss')
    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('MSE',fontsize=15)
    plt.xticks(integer_epochs,fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Training γs LR='+str(lr)+' Batch size='+str(batch_size))
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig(home_dir+'Loss function Gamma inference CNN1D frequencyLR='+str(lr)+' Batch size='+str(batch_size)+'.png')
    plt.show()
    plt.close()
        
    model.load_weights(home_dir+'best_weights_gamma_strain_H5.h5')
    y_predict_gamma_normalized= model.predict(x_test_gamma)
    y_predict_gamma=scaler.inverse_transform(y_predict_gamma_normalized)
    index_test_gamma =y_test_gamma[:, 1]
    list_rho=[]
    list_m2=[]
    for i in range(len(y_test_gamma)):
        index_gamma=index_test_gamma[i]
        index_rho= np.where(y_rho[:, 1] == index_gamma)[0]
        index_m2= np.where(y_m2[:, 1] == index_gamma)[0]
        m2=y_m2[index_m2, 0]
        rho= y_rho[index_rho, 0]
        list_rho.append(rho)
        list_m2.append(m2)
        
    values_rho=np.array(list_rho)
    values_m2=np.array(list_m2)
    flattened_values_rho = np.concatenate(values_rho).ravel()
    flattened_values_m2 = np.concatenate(values_m2).ravel()
    mse = mean_squared_error(y_test_gamma[:, 0], y_predict_gamma)
    mae = mean_absolute_error(y_test_gamma[:, 0], y_predict_gamma)
    r2 = r2_score(y_test_gamma[:, 0], y_predict_gamma)
    rmse= np.sqrt(mse)
    accuracy_percentage = 100 * (1 - rmse / np.mean(y_test_gamma[:, 0]))
    std=np.std(y_predict_gamma)
    mean=np.mean(y_predict_gamma)
          
    print("The accuracy of our model is"+str(r2 *100))
    print("The Mean Absolute Error of our Model is"+str(mae))
    print("The RMSE  of our Model is "+str(rmse))
    print("The STD of our Model is"+str(std))
    print("The Mean of our predictions is"+str(mean))
  
    plt.figure(figsize=(8,6))
    plt.scatter(y_predict_gamma,y_test_gamma[:, 0],c=np.log10(flattened_values_rho), cmap='viridis', s=50)
    plt.plot(y_test_gamma[:, 0], y_test_gamma[:, 0], color = 'red', label = 'x=y')
    cbar=plt.colorbar()
    cbar.set_label(r'$\log_{10}ρ6$ [$M☉/pc^3$]')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Predicted γs',fontsize=15)
    plt.ylabel('Test γs',fontsize=15)
    plt.title("Parameter estimation γs ")
    plt.legend(fontsize=15)
    plt.savefig(home_dir+'Parameter estimation Gamma with Rho CNN1D LR='+str(lr)+' Batch size='+str(batch_size)+'.png')
    plt.show()
    plt.close()
    plt.figure(figsize=(8,6))
    plt.scatter(y_predict_gamma,y_test_gamma[:, 0],c=np.log10(flattened_values_m2), cmap='viridis', s=50)
    plt.plot(y_test_gamma[:, 0], y_test_gamma[:, 0], color = 'red', label = 'x=y')
    cbar=plt.colorbar()
    cbar.set_label(r'$\log_{10}q$')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Predicted γs',fontsize=15)
    plt.ylabel('Test γs',fontsize=15)
    plt.title("Parameter estimation γs ")
    plt.legend(fontsize=15)
    plt.savefig(home_dir+'Parameter estimation Gamma with M2 CNN1D .png')
    plt.show()
    plt.close()
    
    accuracy_percentage = []
    test_values_predictions = []

    # Assuming y_test_rho[:, 0] contains 25 repeated test values and y_predict_rho has corresponding predictions
    for test_value in np.unique(y_test_gamma[:, 0]):  # Loop through unique test values
        if test_value==2.5:
            test_values_predictions.append(test_value)

            # Find indices corresponding to the current test value
            indices = np.where(y_test_gamma[:, 0] == test_value)[0]

            # Gather predictions associated with the current test value
            individual_predictions = [y_predict_gamma[i] for i in indices]

            mse = mean_squared_error(np.repeat(test_value, len(individual_predictions)), individual_predictions)
            rmse= np.sqrt(mse)
            # Calculate R2 score for the current test value
            std=np.std(individual_predictions)
            mean=np.mean(individual_predictions)
            accuracy=rmse/test_value
            accuracy_percentage.append(accuracy)
            print("STD" +str(std))
            print("Mean"+str(mean))
            print("RMSE"+str(rmse))

    # Plotting Accuracy for each test value
    plt.figure(figsize=(10, 6))
    plt.scatter(test_values_predictions, accuracy_percentage, color='red')

    # Connecting the dots (lines between points)
    plt.plot(test_values_predictions, accuracy_percentage, linestyle='-', color='red')
    plt.xlabel('Test γs',fontsize=15)
    plt.ylabel('Accuracy ',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis='y')
    plt.savefig(home_dir + 'Accuracy_Gamma.png')
    plt.show()
    
   
#M2    
if M2_train==True:
    
    scaler = StandardScaler()
    x_train_m2, x_temp_m2, y_train_m2, y_temp_m2 = train_test_split(x_data, y_m2, test_size=0.30 ,random_state=42)
    x_val_m2, x_test_m2, y_val_m2, y_test_m2 = train_test_split(x_temp_m2, y_temp_m2, test_size=0.5,random_state=42) 
    
    if "H5" in data_dir:
        y_train_m2_reshaped = y_train_m2[:, 0].reshape(-1, 1)
        y_val_m2_reshaped = y_val_m2[:, 0].reshape(-1, 1)
        y_train_m2_normalized = scaler.fit_transform(y_train_m2_reshaped)
        y_val_m2_normalized =scaler.fit_transform(y_val_m2_reshaped )
    
    elif "H4" in data_dir:
        
        y_train_m2_normalized = y_train_m2[:, 0]/(max_m2)
        y_val_m2_normalized = y_val_m2[:, 0]/(max_m2)

    
    
    model=CNN1D(x_data.shape[1])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                           patience=5, min_lr=0.00001)
    opt = Adam(lr=lr)
    checkpoint = ModelCheckpoint(home_dir+'best_weights_q_strain_H5.h5',monitor='val_loss',verbose=1, save_best_only=True,mode='min' )
    model.compile(optimizer = opt , loss = "mean_squared_error" , metrics = ['mse'])
    history = model.fit(x_train_m2, y_train_m2_normalized, epochs=epochs, batch_size=batch_size, validation_data=( x_val_m2,y_val_m2_normalized) ,callbacks=[checkpoint,reduce_lr ] ,shuffle= True, verbose=1)
    val_loss=history.history['val_loss']
    loss=history.history['loss']
    plt.figure(figsize=(10,6))
    plt.plot(integer_epochs, val_loss, 'g', label='Validation Loss')
    plt.plot(integer_epochs, loss, 'b', label='Loss')
    plt.xlabel('Epochs',fontsize=15)
    plt.xticks(integer_epochs,fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('MSE',fontsize=15)
    plt.grid()
    plt.title('Training q LR='+str(lr)+' Batch size='+str(batch_size))
    plt.legend()
    plt.savefig(home_dir+'Loss function q inference CNN1D frequencyLR='+str(lr)+' Batch size='+str(batch_size)+'.png')
    plt.show()
    plt.close()
    model.load_weights(home_dir+'best_weights_q_strain_H5.h5')
    y_predict_m2_normalized= model.predict(x_test_m2)

    if "H5" in data_dir:
        y_predict_m2=scaler.inverse_transform(y_predict_m2_normalized)

    elif "H4" in data_dir:
        y_predict_m2= y_predict_m2_normalized*(max_m2)

    index_test_m2 =y_test_m2[:, 1]
    list_rho=[]
    list_gamma=[]
    
    
    for i in range(len(y_test_m2)):
        index_m2=index_test_m2[i]
        index_rho= np.where(y_rho[:, 1] == index_m2)[0]
        index_gamma= np.where(y_gamma[:, 1] == index_m2)[0]
        rho= y_rho[index_rho, 0]
        gamma=y_gamma[index_gamma, 0]
        list_rho.append(rho)
        list_gamma.append(gamma)

    
    values_rho=np.array(list_rho)
    values_gamma=np.array(list_gamma)
    flattened_values_rho = np.concatenate(values_rho).ravel()
    flattened_values_gamma = np.concatenate(values_gamma).ravel()
    mse = mean_squared_error(y_test_m2[:, 0], y_predict_m2)
    mae = mean_absolute_error(y_test_m2[:, 0], y_predict_m2)
    r2 = r2_score(y_test_m2[:, 0], y_predict_m2)
    rmse= np.sqrt(mse)
    std=np.std(y_predict_m2)
    mean=np.mean(y_predict_m2)
    print("The accuracy of our model is"+str(r2 *100))
    print("The Mean Absolute Error of our Model is"+str(mae))
    print("The RMSE  of our Model is "+str(rmse))
    print("The STD of our Model is"+str(std))
    print("The Mean of our predictions is"+str(mean))
    plt.figure(figsize=(8,6))
    plt.scatter(np.log10(y_predict_m2),np.log10(y_test_m2[:, 0]),c=np.log10(flattened_values_rho), cmap='viridis', s=50)

    if "H5" in data_dir:
        
        plt.plot(np.log10(y_test_m2[:, 0]), np.log10(y_test_m2[:, 0]), color = 'red', label = 'x=y')
        
    elif "H4" in data_dir:
        
        plt.axvline(x=np.log10(max_m2), color='r', linestyle='--', label=r'x=$\log_{10}(m2/m1)$')
    cbar=plt.colorbar()
    cbar.set_label(r'$\log_{10}ρ6$ [$M☉/pc^3$]')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(r'Predicted $\log_{10}q$',fontsize=15)
    plt.ylabel(r'Test $\log_{10}q$',fontsize=15)
    plt.title("Parameter estimation q ")
    plt.legend(fontsize=15)
    plt.savefig(home_dir+'Parameter estimation Q with Rho CNN1D LR='+str(lr)+' Batch size='+str(batch_size)+'.png')
    plt.show()
    plt.close()    
    plt.figure(figsize=(8,6))
    plt.scatter(np.log10(y_predict_m2),np.log10(y_test_m2[:, 0]),c=flattened_values_gamma, cmap='viridis', s=50)
   

    if "H5" in data_dir:
        
        plt.plot(np.log10(y_test_m2[:, 0]), np.log10(y_test_m2[:, 0]), color = 'red', label = 'x=y')
        
    
    elif "H4" in data_dir:
        
        plt.axvline(x=np.log10(max_m2), color='r', linestyle='--', label=r'x=$\log_{10}(m2/m1)$')
    cbar=plt.colorbar()
    cbar.set_label('γs')

    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(r'Predicted $\log_{10}q$',fontsize=15)
    plt.ylabel(r'Test $\log_{10}q$',fontsize=15)
    plt.title("Parameter estimation q")
    plt.legend(fontsize=15)
    plt.savefig(home_dir+'Parameter estimation Q with Gamma CNN1D LR='+str(lr)+' Batch size='+str(batch_size)+'.png')
    plt.show()
    plt.close()    
    
    accuracy_percentage = []
    test_values_predictions = []

    
    for test_value in np.unique(y_test_m2[:, 0]):  # Loop through unique test values
        if test_value==10**-3:
            test_values_predictions.append(test_value)

            # Find indices corresponding to the current test value
            indices = np.where(y_test_m2[:, 0] == test_value)[0]

            # Gather predictions associated with the current test value
            individual_predictions = [y_predict_m2[i] for i in indices]
            mse = mean_squared_error(np.repeat(test_value, len(individual_predictions)), individual_predictions)
            rmse= np.sqrt(mse)
            std=np.std(individual_predictions)
            mean=np.mean(individual_predictions)
            accuracy=rmse/test_value
            accuracy_percentage.append(accuracy)
            print("STD" +str(std))
            print("Mean"+str(mean))
            print("RMSE"+str(rmse))

    # Plotting Accuracy for each test value
    plt.figure(figsize=(10, 6))
    plt.scatter(np.log10(test_values_predictions),  accuracy_percentage, color='red')

    # Connecting the dots (lines between points)
    plt.plot(np.log10(test_values_predictions), accuracy_percentage, linestyle='-', color='red')
    plt.xlabel(r'Test $\log_{10}q$',fontsize=15)
    plt.ylabel('Accuracy',fontsize=15)
    plt.grid(axis='y')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(home_dir + 'Accuracy_M2.png')
    plt.show()
    
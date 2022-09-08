

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
import datetime
from sklearn.model_selection import train_test_split

def residual_stack(x, f,reg_const):    

    # residual unit 1    
    x_shortcut = x
    x = Conv1D(f, 1, strides=1, padding="same",kernel_regularizer=regularizers.l2(reg_const), data_format='channels_last')(x)
    x = Activation('relu')(x)
    x = Conv1D(f, 1, strides=1, padding="same", kernel_regularizer=regularizers.l2(reg_const),data_format='channels_last')(x)
    x = Activation('tanh')(x)
    # add skip connection
    if x.shape[1:] == x_shortcut.shape[1:]:
      x = Add()([x, x_shortcut])
    else:
      raise Exception('Skip Connection Failure!')
      
    # residual unit 2    
    x_shortcut = x
    x = Conv1D(f, 1, strides=1, padding="same", kernel_regularizer=regularizers.l2(reg_const),data_format='channels_last')(x)
    x = Activation('relu')(x)
    x = Conv1D(f, 1, strides=1, padding = "same", kernel_regularizer=regularizers.l2(reg_const),data_format='channels_last')(x)
    x = Activation('tanh')(x)
    # add skip connection
    if x.shape[1:] == x_shortcut.shape[1:]:
      x = Add()([x, x_shortcut])
    else:
      raise Exception('Skip Connection Failure!')

    return x

def residual_stack2(x, f,reg_const):
    
    # residual unit 1    
    x_shortcut = x
    x = Conv2D(f, (1,2), strides=1, padding="same", kernel_regularizer=regularizers.l2(reg_const),data_format='channels_last')(x)
    x = Activation('relu')(x)
    x = Conv2D(f, (1,2), strides=1, padding="same", kernel_regularizer=regularizers.l2(reg_const),data_format='channels_last')(x)
    x = Activation('tanh')(x)
    # add skip connection
    print(x.shape[1:])
    print(x_shortcut.shape[1:])
    if x.shape[1:] == x_shortcut.shape[1:]:
      x = Add()([x, x_shortcut])
    else:
      raise Exception('Skip Connection Failure!')
      
    # residual unit 2    
    x_shortcut = x
    x = Conv2D(f, (1,2), strides=1, padding="same", kernel_regularizer=regularizers.l2(reg_const),data_format='channels_last')(x)
    x = Activation('relu')(x)
    x = Conv2D(f, (1,2), strides=1, padding = "same", kernel_regularizer=regularizers.l2(reg_const),data_format='channels_last')(x)
    x = Activation('tanh')(x)
    # add skip connection
    if x.shape[1:] == x_shortcut.shape[1:]:
      x = Add()([x, x_shortcut])
    else:
      raise Exception('Skip Connection Failure!')
      
    return x

def print_time():
    parser = datetime.datetime.now() 
    return parser.strftime("%d-%m-%Y %H:%M:%S")


def normalize(inputs): 
    maxes = []
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        maxes.append(maks)
        if maks != 0:
            normalized.append(eq/maks)
        else:
            normalized.append(eq)
    return np.array(normalized), np.array(maxes)        

def targets_to_list(targets): 
    targets = targets.transpose(2,0,1)
    targetList = []
    for i in range(0, len(targets)):
        targetList.append(targets[i,:,:])
        
    return targetList


seed = 1
import tensorflow
def k_fold_split(inputs, targets): 

    # make sure everything is seeded
    import os
    os.environ['PYTHONHASHSEED']=str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    np.random.permutation(seed)
    tensorflow.random.set_seed(seed)

    p = np.random.permutation(len(targets))
    
    print('min of p = ',np.array(p)[50:100].min())
    print('max of p = ',np.array(p)[50:100].max())
    print('mean of p = ',np.array(p)[50:100].mean())
    inputs = inputs[p]
    targets = targets[p]
    
    ind = int(len(inputs)/5)
    inputsK = []
    targetsK = []
    for i in range(0,5-1):
        inputsK.append(inputs[i*ind:(i+1)*ind])
        targetsK.append(targets[i*ind:(i+1)*ind])
    
    inputsK.append(inputs[(i+1)*ind:])
    targetsK.append(targets[(i+1)*ind:])
    
    return inputsK, targetsK
        
def merge_splits(inputs, targets, k): 
    if k != 0:
        z=0
        inputsTrain = inputs[z]
        targetsTrain = targets[z]
    else:
        z=1
        inputsTrain = inputs[z]
        targetsTrain = targets[z]

    for i in range(z+1, 5):
        if i != k:
            inputsTrain = np.concatenate((inputsTrain, inputs[i]))
            targetsTrain = np.concatenate((targetsTrain, targets[i]))
    
    return inputsTrain, targetsTrain, inputs[k], targets[k]

def wd_layers(wav_input):
    reg_const = 0.0001
    model_list = []
    input_shape2 = []
    first_layer = 32/3
    kernel_sizes = int(125/3)
    for i in range(3):
        input_shape = Input((wav_input))
        xx = Conv1D(filters=first_layer,kernel_size=kernel_sizes,strides=1,kernel_regularizer=regularizers.l2(reg_const))(input_shape)
        xx = Activation('relu')(xx)
        xx = residual_stack(xx, xx.shape[3],reg_const)
        xx = MaxPooling2D(pool_size= (1,2),strides=(1,2))(xx)
        model_list.append(xx)
        input_shape2.append(input_shape)
    return(model_list,input_shape2)



def build_model(wav_input): # houden

    reg_const = 0.0001
    activation_func = 'relu'
    
    xx,wav_input = wd_layers(wav_input)
    xx = layers.concatenate([k for k in xx])
    conv1 = layers.Conv1D(64, 125, strides = 1, kernel_regularizer=regularizers.l2(reg_const))(xx)
    conv1 = Activation('relu')(conv1)
    
    conv1 = residual_stack(conv1, conv1.shape[3],reg_const)
    conv1 = MaxPooling2D(pool_size= (1,2),strides=(1,2))(conv1)

    conv1 = layers.Conv2D(64, (39, 5), strides=(39, 5), padding = 'same', kernel_regularizer=regularizers.l2(reg_const))(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = residual_stack2(conv1, conv1.shape[3],reg_const)
    conv1 = layers.Flatten()(conv1)
    conv1 = layers.Dropout(0.4, seed=seed)(conv1)

    graph_features = layers.Input(shape=(39,2), name='graph_features')
    graph_features_flattened = layers.Flatten()(graph_features)
    merged = layers.concatenate(inputs=[conv1, graph_features_flattened])
    merged = layers.Dense(128)(merged)
    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    final_model = models.Model(inputs=[wav_input, graph_features], outputs=[pga, pgv, sa03, sa10, sa30]) #, pgv, sa03, sa10, sa30
    
    rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.)
    keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=156324)

    final_model.compile(optimizer=rmsprop, loss='mse')#, metrics=['mse'])
    
    return final_model

from tensorflow import keras

import tensorflow as tf


import sys
#network_choice = str(sys.argv[1])
network_choice = network_2
#%%
def main():

    if network_choice == 'network_1':
        test_set_size = 0.20
        inputs = np.load('data/inputs.npy', allow_pickle = True)
        targets = np.load('data/targets.npy', allow_pickle = True)
        graph_features = np.load('data/station_coords.npy', allow_pickle=True)
        graph_features = np.array([graph_features] * inputs.shape[0])
    else:
        test_set_size = 0.20
        inputs = np.load('data/othernetwork/inputs.npy', allow_pickle = True)
        targets = np.load('data/othernetwork/targets.npy', allow_pickle = True)
        graph_features = np.load('data/othernetwork/station_coords.npy', allow_pickle=True)
        graph_features = np.array([graph_features] * inputs.shape[0])
        

    import random
    #random_state_here = int(sys.argv[2])
    random_state_here = 5
    train_inputs, test_inputs, train_graphfeature, test_graphfeature, train_targets, testTargets = train_test_split(inputs, graph_features, targets, test_size=test_set_size, random_state=random_state_here)
    testInputs, testMaxes = normalize(test_inputs[:, :, :1000, :])       

    import math
    
    length_size_min = int((train_inputs.shape[0] / 5))
    print(f"size of length_size_min = {length_size_min}")
    print(f"size of length_size_min = {length_size_min}")

    length_size_max = int((train_inputs.shape[0]) + -(train_inputs.shape[0] / 5))
    print(f"size of length_size_max = {length_size_max}")
    print(f"size of length_size_max = {length_size_max}")
    inputsK, targetsK = k_fold_split(train_inputs, train_targets)
    
    mse_list = []
    rmse_list = []
    mae_list = []
    rsquared_list = []
    maxerror_list = []
    euclideanerror_list = []
    mape_list = []


    for k in range(0,5):

        keras.backend.clear_session()
        tf.keras.backend.clear_session()

        trainInputsAll, trainTargets, valInputsAll, valTargets = merge_splits(inputsK, targetsK, k)
    
        trainInputs, trainMaxes = normalize(trainInputsAll[:, :, :1000, :]) # 100 samples per second
        valInputs, valMaxes = normalize(valInputsAll[:, :, :1000, :])


    
        train_graphfeatureinput = train_graphfeature[0:trainInputsAll.shape[0],:,:]
        val_graphfeatureinput = train_graphfeature[0:valInputsAll.shape[0],:,:]


        trainInputs0 = np.split(trainInputs, 3,axis=3)
        valInputs0 = np.split(valInputs, 3,axis=3)
        testInputs0 = np.split(testInputs, 3,axis=3)
        

        model = build_model(valInputs0[0][0].shape)
    
        iteration_checkpoint = keras.callbacks.ModelCheckpoint(
            'models/ada_cnn_model.hdf5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )
        print(model.summary())
        history = model.fit(x=[[i for i in trainInputs0], train_graphfeatureinput], 
                            y=targets_to_list(trainTargets),
              epochs=100, batch_size=20,
              validation_data=([[j for j in valInputs0],val_graphfeatureinput], targets_to_list(valTargets)),verbose=0, callbacks=[iteration_checkpoint])#
       

        print(f"total parameters in this model: {model.count_params():,.2f}")
        print('total number of epochs ran = ',len(history.history['loss']))
        print('Fold number:' + str(k))
        print('Loss: ',history.history['loss'][-1])
        print('val_loss: ',history.history['val_loss'][-1])

        bestmodel = load_model('models/ada_cnn_model.hdf5')
        
        # Predictions on the test set
        predictions = bestmodel.predict([[k for k in testInputs0],test_graphfeature])

        new_predictions = np.array(predictions)
        new_predictions = np.swapaxes(new_predictions,0,2)
        new_predictions = np.swapaxes(new_predictions,0,1)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,max_error,mean_absolute_percentage_error
        MSE = []
        for i in range(0,5):
            MSE.append(mean_squared_error(testTargets[:,:,i], new_predictions[:,:,i]))
        print('mse = ',np.array(MSE).mean())
        MSE = np.array(MSE).mean()
        
        RMSE = []
        for i in range(0,5):
            RMSE.append(mean_squared_error(testTargets[:,:,i], new_predictions[:,:,i], squared=False))
        print('rmse = ',np.array(RMSE).mean())
        RMSE = np.array(RMSE).mean()
        
        MAE = []
        for i in range(0,5):
            MAE.append(mean_absolute_error(testTargets[:,:,i], new_predictions[:,:,i]))
        print('MAE = ',np.array(MAE).mean())
        MAE = np.array(MAE).mean()
        
        RSQUARED = []
        for i in range(0,5):
            RSQUARED.append(r2_score(testTargets[:,:,i], new_predictions[:,:,i]))
        print('RSQUARED = ',np.array(RSQUARED).mean())
        RSQUARED = np.array(RSQUARED).mean()

        MAPE = []
        for i in range(0,5):
            MAPE.append(mean_absolute_percentage_error(testTargets[:,:,i], new_predictions[:,:,i]))
        MAPE = np.array(MAPE).mean()


        MAX_ERROR = []
        for i in range(0,5):
            MAX_ERROR.append(max_error(testTargets[:,:,i].flatten(), new_predictions[:,:,i].flatten()))
        MAX_ERROR = np.array(MAX_ERROR).mean()

        def dist(x,y):   
            return np.sqrt(np.sum((x-y)**2))

        EUCLIDEAN = []
        for i in range(0,5):
            EUCLIDEAN.append(dist(testTargets[:,:,i], new_predictions[:,:,i]))
        print('euclidean = ',np.array(EUCLIDEAN).mean())
        EUCLIDEAN = np.array(EUCLIDEAN).mean()

        mse_list.append(MSE)
        rmse_list.append(RMSE)
        mae_list.append(MAE)
        rsquared_list.append(RSQUARED)

        maxerror_list.append(MAX_ERROR)
        euclideanerror_list.append(EUCLIDEAN)
        mape_list.append(MAPE)

        print('MSE = ',np.square(np.subtract(predictions[1], testTargets[:,:,1])).mean())
    

        keras.backend.clear_session()
        tf.keras.backend.clear_session()

    print('all averages = ')
    print('mse score = ',np.array(mse_list).mean())
    print('rmse score = ',np.array(rmse_list).mean())
    print('mae score = ',np.array(mae_list).mean())
    print('rsquared score = ',np.array(rsquared_list).mean())
    print('max error score = ',np.array(maxerror_list).mean())
    print('mape error score = ',np.array(mape_list).mean())
    print('euclidean score = ',np.array(euclideanerror_list).mean())



if __name__== "__main__" :
    main()

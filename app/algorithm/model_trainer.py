#!/usr/bin/env python

import os, warnings, sys
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.utils import shuffle
import pprint

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
from algorithm.model.random_forest import RandomForestClassif
from algorithm.utils import get_model_config




# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()
    
    # perform train/valid split 
    train_data = data
    # print('train_data shape:',  train_data.shape) 
    
    # preprocess data
    print("Pre-processing data...")
    train_data, _, preprocess_pipe = preprocess_data(train_data, None, data_schema)  
    train_X, train_y = train_data['X'].astype(np.float), train_data['y'].astype(np.float)  
    # print('train_X/y shape:',  train_X.shape, train_y.shape)   
              
    # Create and train model     
    print('Fitting model ...')  
    model = train_model(train_X, train_y, hyper_params)    
    
    return preprocess_pipe, model


def train_model(train_X, train_y, hyper_params):    
    # get model hyper-paameters parameters 
    #data_based_params = get_data_based_model_params(train_X)
    model_params = {**hyper_params }
    
    # Create and train model   
    model = RandomForestClassif(  **model_params )  
    # model.summary()  
    model.fit(train_X, train_y)
    #print("last_loss:", history.history['loss'][-1])
    return model


def preprocess_data(train_data, valid_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, model_cfg) 
    # pprint.pprint(pp_params)  
    # sys.exit()
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    # print("Processed train X/y data shape", train_data['X'].shape, train_data['y'].shape)
      
    if valid_data is not None:
        valid_data = preprocess_pipe.transform(valid_data)
    # print("Processed valid X/y data shape", valid_data['X'].shape, valid_data['y'].shape)
    return train_data, valid_data, preprocess_pipe 
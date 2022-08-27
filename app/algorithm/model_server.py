import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import os, sys
from interpret.blackbox import LimeTabular

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.classifier as classifier


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema): 
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
        self.data_schema = data_schema
        self.id_field_name = self.data_schema["inputDatasets"]["binaryClassificationBaseMainInput"]["idField"]  
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 3
    
    
    def _get_preprocessor(self): 
        if self.preprocessor is None: 
            try: 
                self.preprocessor = pipeline.load_preprocessor(self.model_path)
                return self.preprocessor
            except: 
                print(f'Could not load preprocessor from {self.model_path}. Did you train the model first?')
                return None
        else: return self.preprocessor
    
    def _get_model(self): 
        if self.model is None: 
            try: 
                self.model = classifier.load_model(self.model_path)
                return self.model
            except: 
                print(f'Could not load model from {self.model_path}. Did you train the model first?')
                return None
        else: return self.model
        
    
    def _get_predictions(self, data):  
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
                    
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)          
        # Grab input features for prediction
        pred_X = proc_data['X'].astype(np.float)        
        # make predictions
        preds = model.predict( pred_X )
        return preds    
    
    
    def predict_proba(self, data):  
        
        preds = self._get_predictions(data)
        # get class names (labels)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)       
        # get the name for the id field
        
        # return te prediction df with the id and class probability fields
        preds_df = data[[self.id_field_name]].copy()
        preds_df[class_names[0]] = 1 - preds   
        preds_df[class_names[1]] = preds  
        #print(preds_df) 
        
        return preds_df    
    
    
    def predict(self, data):
        preds = self._get_predictions(data)        
        
        # inverse transform the prediction probabilities to class labels
        pred_classes = pipeline.get_inverse_transform_on_preds(self.preprocessor, model_cfg, preds) 
        # return te prediction df with the id and prediction fields
        preds_df = data[[self.id_field_name]].copy()
        preds_df['prediction'] = pred_classes
        
        return preds_df
    
    
    def _get_preds_array(self, X): 
        model = self._get_model()
        preds= model.predict_proba(X)
        preds_arr = np.concatenate([1 - preds, preds], axis=1)
        return preds_arr
    
    
    def explain_local(self, data): 
        
        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f'''Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations.'''
            print(msg)
        
        preprocessor = self._get_preprocessor()        
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data.head(self.MAX_LOCAL_EXPLANATIONS))    
        pred_X = proc_data['X'].astype(np.float)   
        
        
        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")            
        lime = LimeTabular(predict_fn=self._get_preds_array, 
                   data=pred_X, 
                   random_state=1)
        
        # Get local explanations
        lime_local = lime.explain_local(X=pred_X,  y=None, name=f"{classifier.MODEL_NAME} local explanations")
        
        # create the dataframe of local explanations to return
        ids =  proc_data['ids']
        dfs = []
        for i, sample_exp in enumerate(lime_local._internal_obj['specific']): 
            df = df = pd.DataFrame() 
            df['feature'] = sample_exp['names'] + [ sample_exp['extra']['names'][0] ]
            df['score'] = sample_exp['scores'] + [ sample_exp['extra']['scores'][0] ]  
            df.sort_values(by=['score'], inplace=True, ascending=False)
            df.insert(0, self.id_field_name, ids[i])
            dfs.append(df)
        
        local_exp_df = pd.concat(dfs, ignore_index=True, axis=0)
        return local_exp_df
        


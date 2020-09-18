"""
This file contains the models used to generate the results in the paper.
It gives a an example of how to use the modules we implemented for this paper.
"""
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from . import models
from sklearn import linear_model
import sys
import numpy as np
from sklearn.model_selection import KFold



class Ols : 
    def __init__(self, train_data,test_data,train_intercept=0,test_intercept=0):
            self.train_data = train_data
            self.train_intercept = train_intercept
            self.test_data = test_data
            # test_intercept is the same as train intercept with different time index only
            self.test_intercept = test_intercept
            self.model=None

            
            
    def train(self):
        nSegments = len(self.train_data)
        valid_split=0.67
        input_lag, output_lag, sequence_length = 1, 1, len(set(pd.to_datetime(self.train_data.columns).time)) # speedDF.columns.size
        params        = { "scale_output" : True,"name":"OLS" }
        data_model    = models.DataModel( self.train_data,    input_lag, output_lag, sequence_length, valid_split = valid_split, **params)
        data_model.preprocessData()
        x_train_00, y_train_00, x_validation_00, y_validation_00 = data_model.trainSplit()
        x_validation=x_validation_00.reshape(x_validation_00.shape[0:3:2])
        y_validation=y_validation_00
        x=x_train_00.reshape(x_train_00.shape[0:3:2])
        y=y_train_00
        
        
        ols_model = [linear_model.LinearRegression(normalize=True) for i in range(nSegments)]
        
        for i in range(nSegments):
            ols_model[i].fit(x, y[:, i])
            # progressbar
            sys.stdout.write("\r[%-100s] %d%%" % ('='*(100*(i+1)//nSegments), 100*(i+1)//nSegments))

        print()

        preds = []

        for i in range(nSegments):
            preds.append(ols_model[i].predict(x_validation))
        preds = np.array(preds)    

        print("Validation MSE:", mean_squared_error(preds.T.flatten(), y_validation.flatten()))
        print("Validation MAE:", mean_absolute_error(preds.T.flatten(), y_validation.flatten()))
        self.model=ols_model

    def predict(self):        
        nSegments = len(self.test_data)
        input_lag, output_lag, sequence_length = 1, 1, len(set(pd.to_datetime(self.test_data.columns).time)) # speedDF.columns.size
        params        = { "scale_output" : True,"name":"OLS" }
        test_data_model    = models.DataModel( self.test_data,    input_lag, output_lag, sequence_length, valid_split = 1, **params)
        test_data_model.preprocessData()
        test_data_x, test_data_y, _, _ = test_data_model.trainSplit()

        x_test=test_data_x.reshape(test_data_x.shape[0:3:2])
        y_test=test_data_y
        
        
        preds=[]
        for i in range(nSegments):
            preds.append(self.model[i].predict(x_test))

        preds = np.array(preds)    

        self.test_prediction = test_data_model.restorePredictionsAsDF(preds.T)

        print("Test MSE:", mean_squared_error(preds.T.flatten(), y_test.flatten()))
        print("Test MAE:", mean_absolute_error(preds.T.flatten(), y_test.flatten()))
        return ( mean_squared_error(preds.T.flatten(), y_test.flatten()),mean_absolute_error(preds.T.flatten(), y_test.flatten()))


class Lasso : 

    def __init__(self, train_data,test_data,train_intercept=0,test_intercept=0):
            self.train_data = train_data
            self.train_intercept = train_intercept
            self.test_data = test_data
            self.test_intercept = test_intercept
            self.model=None

    def train(self):
        valid_split=0.67

        nSegments = len(self.train_data)
        input_lag, output_lag, sequence_length = 1, 1, len(set(pd.to_datetime(self.train_data.columns).time)) # speedDF.columns.size
        params        = { "scale_output" : True,"name":"Lasso" }
        data_model    = models.DataModel( self.train_data,    input_lag, output_lag, sequence_length, valid_split = valid_split, **params)
        data_model.preprocessData()
        x_train_00, y_train_00, x_validation_00, y_validation_00 = data_model.trainSplit()
        x_validation=x_validation_00.reshape(x_validation_00.shape[0:3:2])
        y_validation=y_validation_00

        x=x_train_00.reshape(x_train_00.shape[0:3:2])
        y=y_train_00
        lasso_model = [linear_model.LassoCV(n_jobs=5, cv=5, max_iter=10000,normalize=True) for i in range(nSegments)]

        

        for i in range(nSegments):
            lasso_model[i].fit(x, y[:, i])
            # progressbar
            sys.stdout.write("\r[%-100s] %d%%" % ('='*(100*(i+1)//nSegments), 100*(i+1)//nSegments))
        print()

        
        preds = []

        for i in range(nSegments):
            preds.append(lasso_model[i].predict(x_validation))
        preds = np.array(preds)    

        print("Validation MSE:", mean_squared_error(preds.T.flatten(), y_validation.flatten()))
        print("Validation MAE:", mean_absolute_error(preds.T.flatten(), y_validation.flatten()))
        self.model=lasso_model

    def predict(self):
        
        nSegments = len(self.test_data)
        
        input_lag, output_lag, sequence_length = 1, 1, len(set(pd.to_datetime(self.test_data.columns).time)) # speedDF.columns.size
        params        = { "scale_output" : True,"name":"OLS" }
        test_data_model    = models.DataModel( self.test_data,    input_lag, output_lag, sequence_length, valid_split = 1, **params)
        test_data_model.preprocessData()
        test_data_x, test_data_y, _, _ = test_data_model.trainSplit()

        x_test=test_data_x.reshape(test_data_x.shape[0:3:2])
        y_test=test_data_y
        
        
        preds=[]
        for i in range(nSegments):
            preds.append(self.model[i].predict(x_test))

        preds = np.array(preds)    

        self.test_prediction = test_data_model.restorePredictionsAsDF(preds.T)

        print("Test MSE:", mean_squared_error(preds.T.flatten(), y_test.flatten()))
        print("Test MAE:", mean_absolute_error(preds.T.flatten(), y_test.flatten()))
        return ( mean_squared_error(preds.T.flatten(), y_test.flatten()),mean_absolute_error(preds.T.flatten(), y_test.flatten()))

class TSLasso : 

    def __init__(self, train_data,test_data,train_intercept=0,test_intercept=0):
            self.train_data = train_data
            self.train_intercept = train_intercept
            self.test_data = test_data
            self.test_intercept = test_intercept
            self.model=None
            
    def split_time(df,t_interval_length = 2,overlapping = True):
        t_index  = np.unique(pd.to_datetime(df.columns).time)
        index_subsets = np.split(t_index,range(t_interval_length,len(t_index),t_interval_length)) if not overlapping else np.array([np.array([t_index[i]  for i in range(j,t_interval_length+j)]) for j in range(len(t_index)-t_interval_length+1)])
        index_subsets = [ pd.Index(pd.to_datetime(df.columns).time).isin(subset) for subset in index_subsets]
        return index_subsets
    
    def train(self):
        valid_split=0.67

        t_interval_length = 2
        overlapping = True
        t_index  = np.unique(pd.to_datetime(self.train_data.columns).time)

        index_subsets =  TSLasso.split_time(self.train_data,t_interval_length,overlapping )
        all_models=[]

        A_t = []
        losses = []
        maes=[]
        full_preds,full_y = [],[]

        for i,subset in enumerate(index_subsets) : 
            print("subset "+ str(i))
            X = self.train_data[self.train_data.columns[subset]].copy()
            nSegments = len(X)
            input_lag, output_lag, sequence_length = 1, 1, t_interval_length

            params        = { "scale_output" : True}


            A_t_model    = models.DataModel( X,    input_lag, output_lag, sequence_length, valid_split = valid_split, **params)                                                

            A_t_model.preprocessData()

            A_t_x_train_00, A_t_y_train_00, A_t_x_test_00, A_t_y_test_00 = A_t_model.trainSplit()

            x_test=A_t_x_test_00.reshape(A_t_x_test_00.shape[0:3:2])
            y_test=A_t_y_test_00

            x=A_t_x_train_00.reshape(A_t_x_train_00.shape[0:3:2])
            y=A_t_y_train_00


            A_lasso = [linear_model.LassoCV(n_jobs=5, cv=5, max_iter=10000) for i in range(nSegments)]


            for i in range(nSegments):
                A_lasso[i].fit(x, y[:, i])
                sys.stdout.write("\r[%-100s] %d%%" % ('='*(100*(i+1)//nSegments), 100*(i+1)//nSegments))
            print()
            all_models.append(A_lasso)
        self.model = all_models

    def predict(self):
        t_interval_length = 2
        overlapping = True
        t_index  = np.unique(pd.to_datetime(self.test_data.columns).time)

        index_subsets =  TSLasso.split_time(self.test_data,t_interval_length,overlapping )
        
        A_t = []
        losses = []
        maes=[]
        full_preds,full_y = [],[]

        for i,subset in enumerate(index_subsets) : 
            print("subset "+ str(i))
            X = self.test_data[self.test_data.columns[subset]].copy()
            nSegments = len(X)
            input_lag, output_lag, sequence_length = 1, 1, t_interval_length

            params        = { "scale_output" : True}


            A_t_model    = models.DataModel( X,    input_lag, output_lag, sequence_length, valid_split = 1, **params)                                                

            A_t_model.preprocessData()

            A_t_x_train_00, A_t_y_train_00, A_t_x_test_00, A_t_y_test_00 = A_t_model.trainSplit()

            x_test=A_t_x_train_00.reshape(A_t_x_train_00.shape[0:3:2])
            y_test=A_t_y_train_00

            preds=[]
            A_lasso = self.model[i]
            for j in range(nSegments):
                preds.append(A_lasso[j].predict(x_test))
            preds = np.array(preds)    
            full_preds.append(A_t_model.restorePredictionsAsDF(preds.T))

        self.test_prediction = pd.concat(full_preds,axis=1).sort_index(axis=1)
        print("MSE:", mean_squared_error(self.test_prediction.values.flatten(), self.test_data[self.test_prediction.columns].values.flatten()))
        print("MAE:", mean_absolute_error(self.test_prediction.values.flatten(), self.test_data[self.test_prediction.columns].values.flatten()))



class RSLasso : 

    def __init__(self, train_data,test_data,train_intercept=0,test_intercept=0):
            self.train_data = train_data
            self.train_intercept = train_intercept
            self.test_data = test_data
            self.test_intercept = test_intercept
            self.model=None
            
    def split_data(df,t=2):
        quarters = np.unique(pd.to_datetime(df.columns).time)
        quarters.sort()
        quarter_split=quarters[t]
        return{ True : df.groupby(pd.to_datetime(df.columns).time<=quarter_split,axis=1).get_group(True),False : df.groupby(pd.to_datetime(df.columns).time>=quarter_split,axis=1).get_group(True)}
    
    def find_best_time_switch(self):
        avg_mse=[]
        for t_split in np.arange(1,len(set(self.train_data.columns.time))-1):
            preds=[]
            kf=KFold(n_splits=5, shuffle=True, random_state=None)
            for fold in kf.split(np.unique(self.train_data.columns.date)):
                train_idx,test_idx= fold
                kf_train = self.train_data[self.train_data.columns[np.isin(self.train_data.columns.date,np.unique(self.train_data.columns.date)[train_idx])]]
                kf_test = self.train_data[self.train_data.columns[np.isin(self.train_data.columns.date,np.unique(self.train_data.columns.date)[test_idx])]]
                kf_train_intercept = self.train_intercept[self.train_data.columns[np.isin(self.train_data.columns.date,np.unique(self.train_data.columns.date)[train_idx])]]
                kf_test_intercept=self.train_intercept[self.train_data.columns[np.isin(self.train_data.columns.date,np.unique(self.train_data.columns.date)[test_idx])]]
                RSlasso = RSLasso(kf_train,kf_test,train_intercept=kf_train_intercept,test_intercept=kf_test_intercept)
                RSlasso.train(split_t=t_split)
                preds.append(RSlasso.predict())
            avg_mse.append(np.array(preds).mean(axis=0)[0])
        return np.array(avg_mse).argmin()
    
    def train(self,split_t=None):
        if split_t is None:
            print("No time split provided. looking for best time split ...")
            self.split=self.find_best_time_switch()
            print('best split is : '+str(self.split))
        else :
            self.split=split_t

        left_speed_df,right_speed_df =  RSLasso.split_data(self.train_data,self.split)[True],RSLasso.split_data(self.train_data,self.split)[False]
        left_speed_test_df,right_speed_test_df = RSLasso.split_data(self.test_data,self.split)[True], RSLasso.split_data(self.test_data,self.split)[False]
        
        left_model= Lasso(left_speed_df,left_speed_test_df)
        print("training early model")
        left_model.train()
        
        right_model = Lasso(right_speed_df, right_speed_test_df)
        print("training late model")
        right_model.train()
        
        self.model=[left_model,right_model]

    def predict(self):
        print("prediction early model")

        self.model[0].predict()
        
        print("training late model")

        self.model[1].predict()
        print("training regime switching model")

        self.test_prediction = pd.concat([self.model[0].test_prediction, self.model[1].test_prediction],axis=1).sort_index(axis=1)

        print("MSE:", mean_squared_error(self.test_prediction.values.flatten(), self.test_data[self.test_prediction.columns].values.flatten()))
        print("MAE:", mean_absolute_error(self.test_prediction.values.flatten(), self.test_data[self.test_prediction.columns].values.flatten()))
        return ( mean_squared_error(self.test_prediction.values.flatten(), self.test_data[self.test_prediction.columns].values.flatten()),mean_absolute_error(self.test_prediction.values.flatten(), self.test_data[self.test_prediction.columns].values.flatten()))

        
        
        
        
        
        
        
        
        
        
        
class ElasticNet : 

    def __init__(self, train_data,test_data,train_intercept=0,test_intercept=0):
            self.train_data = train_data
            self.train_intercept = train_intercept
            self.test_data = test_data
            self.test_intercept = test_intercept
            self.model=None

    def train(self):
        valid_split=0.67

        nSegments = len(self.train_data)
        input_lag, output_lag, sequence_length = 1, 1, len(set(pd.to_datetime(self.train_data.columns).time)) # speedDF.columns.size
        params        = { "scale_output" : True,"name":"Lasso" }
        data_model    = models.DataModel( self.train_data,    input_lag, output_lag, sequence_length, valid_split = valid_split, **params)
        data_model.preprocessData()
        x_train_00, y_train_00, x_validation_00, y_validation_00 = data_model.trainSplit()
        x_validation=x_validation_00.reshape(x_validation_00.shape[0:3:2])
        y_validation=y_validation_00

        x=x_train_00.reshape(x_train_00.shape[0:3:2])
        y=y_train_00
        lasso_model = [linear_model.ElasticNetCV(l1_ratio = np.arange(0.1,0.9,0.1),n_jobs=5, cv=5, max_iter=10000,normalize=True) for i in range(nSegments)]

        

        for i in range(nSegments):
            lasso_model[i].fit(x, y[:, i])
            # progressbar
            sys.stdout.write("\r[%-100s] %d%%" % ('='*(100*(i+1)//nSegments), 100*(i+1)//nSegments))
        print()

        
        preds = []

        for i in range(nSegments):
            preds.append(lasso_model[i].predict(x_validation))
        preds = np.array(preds)    

        print("Validation MSE:", mean_squared_error(preds.T.flatten(), y_validation.flatten()))
        print("Validation MAE:", mean_absolute_error(preds.T.flatten(), y_validation.flatten()))
        self.model=lasso_model

    def predict(self):
        
        nSegments = len(self.test_data)
        
        input_lag, output_lag, sequence_length = 1, 1, len(set(pd.to_datetime(self.test_data.columns).time)) # speedDF.columns.size
        params        = { "scale_output" : True,"name":"OLS" }
        test_data_model    = models.DataModel( self.test_data,    input_lag, output_lag, sequence_length, valid_split = 1, **params)
        test_data_model.preprocessData()
        test_data_x, test_data_y, _, _ = test_data_model.trainSplit()

        x_test=test_data_x.reshape(test_data_x.shape[0:3:2])
        y_test=test_data_y
        
        
        preds=[]
        for i in range(nSegments):
            preds.append(self.model[i].predict(x_test))

        preds = np.array(preds)    

        self.test_prediction = test_data_model.restorePredictionsAsDF(preds.T)

        print("Test MSE:", mean_squared_error(preds.T.flatten(), y_test.flatten()))
        print("Test MAE:", mean_absolute_error(preds.T.flatten(), y_test.flatten()))

class RidgeCV : 

    def __init__(self, train_data,test_data,train_intercept=0,test_intercept=0):
            self.train_data = train_data
            self.train_intercept = train_intercept
            self.test_data = test_data
            self.test_intercept = test_intercept
            self.model=None

    def train(self):
        valid_split=0.67

        nSegments = len(self.train_data)
        input_lag, output_lag, sequence_length = 1, 1, len(set(pd.to_datetime(self.train_data.columns).time)) # speedDF.columns.size
        params        = { "scale_output" : True,"name":"Lasso" }
        data_model    = models.DataModel( self.train_data,    input_lag, output_lag, sequence_length, valid_split = valid_split, **params)
        data_model.preprocessData()
        x_train_00, y_train_00, x_validation_00, y_validation_00 = data_model.trainSplit()
        x_validation=x_validation_00.reshape(x_validation_00.shape[0:3:2])
        y_validation=y_validation_00

        x=x_train_00.reshape(x_train_00.shape[0:3:2])
        y=y_train_00
        lasso_model = [linear_model.RidgeCV( cv=5,normalize=True) for i in range(nSegments)]

        

        for i in range(nSegments):
            lasso_model[i].fit(x, y[:, i])
            # progressbar
            sys.stdout.write("\r[%-100s] %d%%" % ('='*(100*(i+1)//nSegments), 100*(i+1)//nSegments))
        print()

        
        preds = []

        for i in range(nSegments):
            preds.append(lasso_model[i].predict(x_validation))
        preds = np.array(preds)    

        print("Validation MSE:", mean_squared_error(preds.T.flatten(), y_validation.flatten()))
        print("Validation MAE:", mean_absolute_error(preds.T.flatten(), y_validation.flatten()))
        self.model=lasso_model

    def predict(self):
        
        nSegments = len(self.test_data)
        
        input_lag, output_lag, sequence_length = 1, 1, len(set(pd.to_datetime(self.test_data.columns).time)) # speedDF.columns.size
        params        = { "scale_output" : True,"name":"OLS" }
        test_data_model    = models.DataModel( self.test_data,    input_lag, output_lag, sequence_length, valid_split = 1, **params)
        test_data_model.preprocessData()
        test_data_x, test_data_y, _, _ = test_data_model.trainSplit()

        x_test=test_data_x.reshape(test_data_x.shape[0:3:2])
        y_test=test_data_y
        
        
        preds=[]
        for i in range(nSegments):
            preds.append(self.model[i].predict(x_test))

        preds = np.array(preds)    

        self.test_prediction = test_data_model.restorePredictionsAsDF(preds.T)

        print("Test MSE:", mean_squared_error(preds.T.flatten(), y_test.flatten()))
        print("Test MAE:", mean_absolute_error(preds.T.flatten(), y_test.flatten()))





import datetime

import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from . import CustomUtils
from functools import reduce
import matplotlib
import folium
from . import Plotting
import warnings
from datetime import timedelta


class BaseModels:
    '''This class contains the base models that we aim to surpass.
    prédicteur historique,
    prédicteur historique sur le temps d’interet,
    dernière valeur observée,
    AR(5).
    '''

    def __init__(self, model, historic_data=None,lag=5):
        '''Base models class initialisation
        
        Arguments:
            model {string} -- The model to create ['lastValue', 'historic', 'timehistoric', 'AR5']
            historic_data {ps.Series} -- The historical data, typically updatedSpeed.iloc[:,:200]
        '''

        self.history = historic_data
        self.type = model
        
        if model == 'AR5':
            self.lag=lag

            self.models = self.AR5_train(historic_data)


    def predict(self, x, time=datetime.time(14,0)):
        '''Method to make predictions, depending on the model used.
        
        Arguments:
            x {pandas DataFRame} -- The input, the last data, just before the what we want to predict.
        
        Keyword Arguments:
            time {datetime.time} -- Only for timeHistoric model. The time for which we want the prediction. (default: {datetime.time(14,0)})
        
        Returns:
            numpy.array -- The array of values predicted.
        '''

        y=[]
        if self.type == 'lastValue':
            y = x.iloc[:, -1]
        elif self.type == 'historic':
            y = self.history.mean(axis=1).values
        elif self.type == 'timeHistoric':
            columns = [d for d in self.history.columns if d.time()==time]
            y = self.history[columns].mean(axis=1).values
        elif self.type == 'AR5':
            y = self.AR5(x)
        return y



    def AR5_train(self, updatedSpeeds):
        '''The method to train teh AR(5) model
        
        Arguments:
            updatedSpeeds {pandas.DataFrame} -- The historic data for training
        
        Returns:
            list -- A list of AR5 models. One per section.
        '''

        print('Training the AR(5) model')
        train = updatedSpeeds.values
        train_dates = updatedSpeeds.columns
        print('Train data shape:', train.shape)
        
        print('\nFilling the voids...')
        minutes_interval=str(int((updatedSpeeds.columns[1]-updatedSpeeds.columns[0]).total_seconds()/60))

        delta = datetime.timedelta(microseconds=(updatedSpeeds.columns[1]-updatedSpeeds.columns[0]).value//1000)
        
        stretchedDF=pd.DataFrame(index = updatedSpeeds.index,columns=pd.to_datetime(pd.date_range(updatedSpeeds.columns[0],updatedSpeeds.columns[-1],freq="0.25H")))
        stretchedDF.update(updatedSpeeds)
        stretchedDF=stretchedDF.fillna(0)
        print('\nTraining the models...')
        
        print('Params: max_lag:', self.lag)
        

        models = [smt.AR(stretchedDF.values[i], dates=stretchedDF.columns, freq=minutes_interval+'min').fit(maxlag=self.lag, trend='c') for i in range(stretchedDF.shape[0]) ]
        print('\nTraining finished !')

        return models


    def AR5_single_pred(self, data, section):
        '''Method to predict the foloowing value for a single section
        
        Arguments:
            data {pandas.DataFrame} -- The input data
            section {int} -- The id of the section considered
        
        Returns:
            float -- The speed prediction for this one section
        '''

        coefs = self.models[section].params
        pred = coefs[0]
        
        for i in range(1,self.lag+1):
            pred += coefs[i] * data.iloc[section, self.lag-i]
        
        return pred


    def AR5(self, x):
        '''
        The method for AR5 predictions
        
        Arguments:
            x {panda.DataFrame} -- The input data. (data of last lags)
        
        Returns:
            numpy.array -- The predictions
        '''

        predictions = np.array([self.AR5_single_pred(x, i) for i in range(x.shape[0])])
        
        return predictions


    
class DataModel:
    
    def __init__(self,data, input_lag, output_lag, sequence_length,scale_max=False,scale_log=False,shift_mean=False,y_only=False,add_time=False,max_value=130,valid_split=0.7,min_max_scale=False,differentiate_y=False,scale_output = True,segmentWiseNormalization=False,name="model_name"):
        self.name=name
        self.data = data
        self.input_lag = input_lag
        self.output_lag = output_lag
        self.sequence_length = sequence_length
        self.scale_max = scale_max
        self.scale_log = scale_log
        self.shift_mean = shift_mean
        self.y_only=y_only
        self.add_time = add_time
        self.max_value = max_value
        self.min_max_scale = min_max_scale
        self.differentiate_y = differentiate_y
        self.segmentWiseNormalization = segmentWiseNormalization
        self.models=None
        self.count_data =None
        self.time_data  =None
        self.valid_split=valid_split
        if (len(data.columns)/sequence_length)<10 : 
            self.split_idx=(int((valid_split)*(len(data.columns)-input_lag)))
        else : 
            self.split_idx= (int((valid_split)*(len(data.columns)/sequence_length)))*(sequence_length-input_lag)
            
            
        
        self.x,self.y,self.t = self.getXY()
        self.n_segments = len(data)
        self.scale_output = scale_output
        self.__reversed_process=[]

    def __onehot(x,size=19):
        idx=(x-14)*4
        ar =np.zeros((size,))
        ar[int(idx)-1] = 1
        return ar    
    
    def getDaysTypes(self,onehot=False):
        """
        returns the types of day (monday to friday), and real value representing the time of day for each example (number of seconds/ 60*60)
        """
        day_types = pd.DatetimeIndex(self.t.reshape(-1)).weekday.values.reshape(self.t.shape)
        time_fraction = (CustomUtils.timeToSeconds(pd.DatetimeIndex(self.t.reshape(-1)))/(60*60)).values.reshape(self.t.shape)
        time_input = np.concatenate([day_types,time_fraction],1)
        train_days = time_input[:self.split_idx]
        test_days = time_input[self.split_idx:]
        
        if onehot : 
            return np.array(list(DataModel.__onehot(x,self.sequence_length) for x in train_days[:,1])),np.array(list(DataModel.__onehot(x,self.sequence_length) for x in test_days[:,1]))
        return train_days,test_days
    
    def getExamples(self,sequence,hours):
        """
        create examples (inputlag,outputlag) for one day by shifting time by one step 
        
        """
        
        sequence_length=len(sequence)
        sub_sequence_length = self.input_lag+self.output_lag
        if sub_sequence_length > sequence_length :
            raise ValueError("sequence length {} too small for lags : {},{}".format(sequence_length,self.input_lag,self.output_lag))
        return [sequence[i:i+self.input_lag] for i in range(0,sequence_length-sub_sequence_length+1,1)],\
               [sequence[i+self.input_lag:i+self.input_lag+self.output_lag] for i in range(0,sequence_length-sub_sequence_length+1,1)],\
               [hours[i+self.input_lag:i+self.input_lag+self.output_lag] for i in range(0,sequence_length-sub_sequence_length+1,1)]  
    
    def getXY(self):
        """
        create X and Y matricies out of the original speed dataFrame
        X shape : n_samples, inputLag, n_segments
        Y shape : n_samples, outputLag, n_segments (middle dimension is dropped if output lag =1)

        """
        
        
        nsegs,ntime=self.data.shape
        if(ntime%self.sequence_length)!= 0 :
            raise ValueError("sequence length {} not compatible with number of time features {}".format(self.sequence_length,ntime))

        shapedData = self.data.values.T.reshape(int(ntime/self.sequence_length),self.sequence_length,nsegs)
        timestamps = pd.Series(self.data.columns).values.reshape(int(ntime/self.sequence_length),self.sequence_length)
        
        examples=[self.getExamples(x,hours) for x,hours in zip(shapedData,timestamps)]

        x,y,t = list(zip(*examples))
        return np.concatenate(x), np.concatenate(y), np.concatenate(t)
    
    
    def getIndexes(self,idx):
        """
        restore the time index of all lags used given a sample position on the X matrix
        """
        cx,cy= (idx +(self.input_lag+self.output_lag-1)*(idx//(self.sequence_length - self.input_lag-self.output_lag+1)),\
                idx +(self.input_lag+self.output_lag-1)*(idx//(self.sequence_length - self.input_lag-self.output_lag+1))+self.input_lag )
        return (self.data.columns[cx:cy],self.data.columns[cy:cy+self.output_lag])
    
    def scaleMax(self):
        """
        divide all values by a max_value (default 130)
        """
        self.__reversed_process.append(self.reverseScaleMax)
        self.x/=self.max_value
        if self.scale_output:
            self.y/=self.max_value
        
    def scaleMinMax(self):
        """
        normalize data to 0-1 scale
        """
        
        self.__reversed_process.append(self.reverseMinMaxScale)
        self.min =self.x[:self.split_idx].min()
        self.max =self.x[:self.split_idx].max()
        diff = self.max - self.min
        self.x = (self.x-self.min)/diff
        if self.scale_output:
            self.y = (self.y-self.min)/diff
    def segmentWiseNormalisation(self):
        
        self.__reversed_process.append(self.reverseSegmentWiseNormalisation)
        
        self.segmin =self.x[:self.split_idx].min(axis=0).min(axis=0)
        self.segmax =self.x[:self.split_idx].max(axis=0).max(axis=0)
        
        diff = self.segmax - self.segmin
        
        self.x = (self.x-self.segmin)/diff
        if self.scale_output:
            self.y = (self.y-self.segmin)/diff
    
    def reverseSegmentWiseNormalisation(self,x):

        return x*(self.segmax-self.segmin)+self.segmin
    
    def reverseMinMaxScale(self,x):
        """
        reverse normalisation
        """
        return x*(self.max-self.min)+self.min
    
    def reverseScaleMax(self,y):
        """
        reverse max scaling
        """
        return y*self.max_value

        
    def scaleLog(self):
        """
        apply log(x+1) on all values
        
        """
        self.__reversed_process.append(self.reverseScaleLog)
        self.__lastx=self.x[:,-1,:]
        if self.scale_output:
            self.y=np.log1p(self.y) - np.log1p(self.x[:,-1,:])
        self.x=np.log1p(self.x)
        self.x= self.x[:,1:,:] -self.x[:,:self.input_lag-1,:]
        
        
    def reverseScaleLog(self,y):
        """
        reverse the log scale
        """
        x_len = y.shape[0]
        if( x_len == self.split_idx or x_len == self.__lastx.shape[0]):
            
            return np.expm1(y) *self.__lastx[:x_len]
        else :
            return np.expm1(y) *self.__lastx[-x_len:]
    def addTime(self):
        """
        add time represntation of all input lags
        """
        self.__reversed_process.append(self.removeTime)
        self.x=np.concatenate((self.x,self.t.reshape(-1,self.t.shape[1],1)),2)
        
        
    def removeTime(self,y):
        """
        remove time for input
        """
        if y.shape == self.x.shape :
            return np.delete(self.x,self.x.shape[2]-1,axis=2)
        return y
        
    def shiftMean( self, quarterwise=False ) :
        """
        Compute local time mean on train data and substract it from all data
        """
        self.__reversed_process.append(self.resetMean)
        self.means  =  self.data[self.data.columns[:(int((valid_split)*(len(data.columns)/sequence_length)))*sequence_length]].mean(axis=1).values
        
        self.x-=self.means
        
        if self.scale_output:
            self.y-=self.means
        
    def resetMean(self,y):
        """
        add local time mean to input
        """
        return y+self.means
        
        

    
    def differentiateY(self):
        """
        compute the difference between output y and the last x lag (transforming the problem to prediction of change from last value)
        """
        self.__reversed_process.append(self.reverseDifferentiatedY)
        if self.scale_output:
            self.y = self.y-self.x[:,-1:,:]
    
    def reverseDifferentiatedY(self,y):
        """
        reverse y differntiation
        """
        if len(y.shape)>2  : return y
        if self.output_lag >1 :
            return y+self.x[:,-1:,:]
        return y+self.x[:,-1,:]



    def preprocessData(self):
        """
        apply different preprocessings if requested
        
        """


        if self.differentiate_y :
            self.differentiateY()
        if self.segmentWiseNormalization:
            self.segmentWiseNormalisation()
        if self.shift_mean :
            self.shiftMean()
            
        if self.scale_max :
            self.scaleMax()
            

        if self.min_max_scale : 
            self.scaleMinMax()
        if self.add_time :
            self.addTime()
        
        self.y=self.y.reshape(self.y.shape[0],-1)
        if self.scale_log :
            self.scaleLog()

    def getRawYData(self,y):
        """
        reverse all preprocessings done on data
        """
        return reduce(lambda res, f:f(res), self.__reversed_process[::-1], y)

    def mse(self,p,y=None,y_step=0):
        """
        Compute mse between predictions and true values on the original scale
        """

        
        y=y.reshape((-1,self.output_lag,self.x.shape[-1]))[:,y_step,:]
        if y is not None :
            if not self.scale_output:
                return np.mean((p-y)**2)
            raw_y = self.getRawYData(y)
        else :
            if not self.scale_output:
                return np.mean((p-self.y)**2)
            
            raw_y = self.getRawYData(self.y)
            
        pred = self.getRawYData(p)
   
        return np.mean((pred-raw_y)**2)
    
    def mae(self,p,y=None,y_step=0):
        """
        Compute mae between predictions and true values on the original scale
        """
        y=y.reshape((-1,self.output_lag,self.x.shape[-1]))[:,y_step,:]

        if y is not None :
            if not self.scale_output:
                return np.mean(abs(p-y))
            raw_y = self.getRawYData(y)
        else :
            if not self.scale_output:
                return np.mean(abs(p-self.y))
            
            raw_y = self.getRawYData(self.y)
            
        pred = self.getRawYData(p)
   
        return np.mean(abs(pred-raw_y))
    
    def trainSplit(self):
        """
        split data into train, validation sets (using valid_split attribute)
        """
        
        x_train = self.x[:self.split_idx]
        x_test = self.x[self.split_idx:]
        y_train = self.y[:self.split_idx]
        y_test = self.y[self.split_idx:]
        return x_train,y_train,x_test,y_test
    
    def getSplitSequences(self,values,sequence_length,skip=0):
        """
        add nans where prediction is not possible (instead of linking non sequential data)
        
        """
        
        def addNans(values, sequence_length, skip):
            try :
                values=values.reshape(-1,sequence_length)
            except ValueError:
                warnings.warn("cannont reshape the data to sequence length (this is probably due to trying to plot train/validation data only (not implemented)) we rollback to linking sequences plot is going to be ugly", None)
                raise
            nans=np.array([np.nan]*(values.shape[0]*(skip+1))).reshape(values.shape[0],-1)
            values = np.concatenate((values,nans),axis=1).reshape(-1)
            return values
        return addNans(np.arange(len(values)),sequence_length,skip), addNans(values,sequence_length,skip)
    
    def restorePredictionsAsDF(self,preds,split="full"):
        """
        create a data frame from predictions with time index
        Note : input is supposed to be the full (no train validation split) otherwise the time indexe will be wrong (TODO)
        """
        if split.lower()=="test":
            test_start=self.split_idx
            index = [self.getIndexes(i+test_start)[1][0] for i in range(len( preds ))]
        else:
            
            index = [self.getIndexes(i)[1][0] for i in range(len( preds ))]
        
        if self.scale_output :
            
            df = pd.DataFrame(self.getRawYData(preds),index=index,columns=self.data.index)
            
        else :
            df = pd.DataFrame(preds,index=index,columns=self.data.index)
            
        return df.T
    
    def restoreXAsDF(self,x):
        """
        create data frame from X matrix
        """
        
        index = [self.getIndexes(i)[1][0] for i in range(len(x))]
        
        df = pd.DataFrame(self.getRawYData(x).swapaxes(1,2).tolist(),index=index,columns=self.data.index)
        
        return df.T
    
    def predict(self,split="full",y_step=0,x=None):
        
        """
        make prediction on the data using the stored model (baseline or lstm for now)
        """
            
        time_index = [self.getIndexes(i)[1][0] for i in range(len(self.x))]
                            
        if split.lower() == "custom":
            if not x is None :
                main_input = x
                
        if split.lower() == "full":
            main_input = self.x
            
            if not self.count_data is None :
                count_input = self.count_data.x
            
            if not self.time_data is None :
                secondary_input = np.concatenate(self.time_data)
            
            
        if split.lower() == "train":
            main_input,*_ = self.trainSplit()
            
            if not self.count_data is None :
                count_input,*_  = self.count_data.trainSplit()
            
            
            if not self.time_data is None :
                secondary_input=self.time_data[0]
            
            time_index = time_index[:self.split_idx]
            
        if split.lower() == "test":
            

            *_,main_input,_ = self.trainSplit()
            
            if not self.count_data is None :
                *_,count_input,_  = self.count_data.trainSplit()
            
            time_index = time_index[self.split_idx:]
            
            if not self.time_data is None :
                secondary_input=self.time_data[1]
            
            
            
        if isinstance(self.model,BaseModels):
            return np.array([self.model.predict(pd.DataFrame(x_i).T,time_index[i].time()) for  i,x_i in enumerate(main_input)])
            
        inputs = [main_input]
        if not self.count_data is None :
            inputs.append(count_input)
                    
        if not self.time_data is None :
            inputs.append(secondary_input)
                
        if self.output_lag>1:    
            return self.getYAtStep(self.model.predict(inputs),y_step)
        return self.model.predict(inputs)
        
    def getYAtStep(self,y,y_step=0):
        return y.reshape((-1,self.output_lag,self.x.shape[-1]))[:,y_step,:]
    
    
class DataCleaner:
    """
    this class is used to clean data:
    reindexing new roads
    merging roads data
    dropping weekends
    imputing missing values
    droping unwanted erroneous data
    ...
    """
    def __init__(self,data,segmentsMeta,mergeResults,counts=None,thresh=0.8,merge_segments=True):
        self.rawData = data.copy()
        if not counts is None :
            self.rawCounts =counts.copy()
        
        self.data = data
        self.counts =counts
        self.segmentsMeta=segmentsMeta
        self.mergeResults=mergeResults
        self.mergedIndex=None
        
        self.dropWeekends()
        if self.countsAvailable : 
            self.dropErroneousData()
        if merge_segments :
            self.computeMergeData(thresh)
        self.fillNaWithHistoricalValues()
        self.segments_tags = segmentsMeta[segmentsMeta.segmentID.isin( self.data.index)].set_index('segmentID').reindex(self.data.index).tag.apply(lambda x :x['highway'])
        

    def countsAvailable(self):
        return not self.counts is None
    
    def dropWeekends(self):
        """
        drop weekends from data
        """
        self.data.drop(
            self.data.columns[
                [ x.date().weekday()>=5 for x  in self.data.columns]
            ],
            axis=1,
            inplace=True)
        if self.countsAvailable():
            self.counts.drop(
                self.counts.columns[
                    [ x.date().weekday()>=5 for x  in self.counts.columns]
                ],
                axis=1,
                inplace=True
            )            
        
    def fillNaWithHistoricalValues(self):
        """
        replacing missing data with local time mean values
        """
        oldIdx = self.data.columns
        # splitting index form datetime to multi index (date,time)
        idx=[pd.to_datetime(self.data.columns.values).date,pd.to_datetime(self.data.columns.values).time]
        mIdx=pd.MultiIndex.from_arrays(idx,names=['day','time'])
        self.data.set_axis(mIdx,axis=1,inplace=True)
        # computing local time means
        self.data = self.data.add(
            self.data.isna()*self.data.groupby(by=self.data.columns.get_level_values(1),axis=1).mean(),
            fill_value=0)
        
        # resetting old index
        self.data.set_axis(oldIdx,axis=1,inplace=True)        
        
    def computeMergeData(self, thresh=0.8):
        """
        setting mean speed and sum counts as values for merged segments
        """
        self.mergedIndex=pd.Series(data=self.segmentsMeta.loc[self.mergeResults]['segmentID'].values,index = self.segmentsMeta['segmentID'].values)
        self.data =self.data * self.counts
        self.data = self.data.assign(newIndex =self.mergedIndex.reindex(self.data.index).values)
        self.data = self.data[~self.data.newIndex.isna()]
        self.data=(self.data.groupby('newIndex').mean()*self.data.groupby('newIndex').count()).dropna(thresh = int(thresh*len(self.data.columns)))
        self.counts = self.counts.assign(newIndex =self.mergedIndex.reindex(self.counts.index).values)
        self.counts = self.counts[~self.counts.newIndex.isna()]
        self.counts = self.counts.groupby('newIndex').sum().loc[self.data.index]
        self.data =self.data/self.counts
            
    def dropErroneousData(self):
        """
        drop some erroneous data (will probably change should be more dynamic)
        """
        days_count =self.counts.groupby(pd.DatetimeIndex(self.data.columns).date,axis=1).sum().sum()
        days_quarter_count = pd.Series(self.data.columns.date).value_counts()
        days_index=np.intersect1d(days_count[days_count>0.75*days_count.median()].index,days_quarter_count[days_quarter_count==days_quarter_count.mode().iloc[0]].index)
        self.data=self.data[self.data.columns[[ x.date() in days_index for x  in self.data.columns]]]
        self.counts = self.counts[self.data.columns[[ x.date() in days_index for x  in self.data.columns]]]
        
    def plotSegmentComponents(self,idx):
        self.rawData.reindex(self.mergedIndex[self.mergedIndex==self.data.iloc[idx].name].index.values).T.plot(use_index=False,figsize=(36,5),title="speeds")
        self.rawCounts.reindex(self.mergedIndex[self.mergedIndex==self.data.iloc[idx].name].index.values).T.plot(use_index=False,figsize=(36,5),title="counts")
        
class ModelPlots:
    
    """
    functions used to plot results , losses of the model
    
    """
    def __init__(self,data_model, data_cleaner,split="full",y=None,preds= None, y_step=0,intercept_data_model=None):
        
        if not y is None and not preds is None :
            self.y=y
            self.preds =preds
            return
        self.data_model = data_model
        self.data_cleaner = data_cleaner
        self.intercept_data_model=intercept_data_model
        if self.data_model.scale_output :
            self.preds = data_model.getRawYData(data_model.predict(split,y_step))
            self.y = data_model.getRawYData(data_model.getYAtStep(data_model.y,y_step))
        else :
            self.preds = data_model.predict(split,y_step)
            self.y = data_model.getYAtStep(data_model.y,y_step)
        if not y is None:
            self.y=y
        if not self.intercept_data_model is None:
            if split.lower()=="test": 
                self.y += self.intercept_data_model.trainSplit()[3]
                self.preds += self.intercept_data_model.trainSplit()[3]
            
    def createSubPlots(self,data, pltFunc=plt.plot, figsize=(12,12),titles=None):
        """
        create sub plots using data and pltFunc
        """
        nCols= int(np.sqrt(len(data)))+1
        plt.figure(figsize=figsize)
        for i,vals in enumerate(data):
            plt.subplot(nCols,nCols,i+1)
            plt.plot(vals)
            plt.xlabel("epochs")
            plt.ylabel("MSE")
            if type(titles)!=type(None):
                plt.title(titles[i])
        plt.tight_layout()
    
    def plotDiscreteSpeedError(self,ax,name=""):
        """
        plots average absolute error per discrete speed 
        """
        error = abs((self.y -self.preds).flatten().round())
        yys=self.y.flatten().round()
        arsort=yys.argsort()
        error = error[arsort]
        yys = yys[arsort]
        y_idx=np.unique(yys,return_index=True)[0]
        split_idx = np.unique(yys,return_index=True)[1][1:]
        y_mean_error=np.fromiter([np.mean(x) for x in np.split(error ,split_idx)],dtype=float)
        plt.plot(y_idx,y_mean_error,label=name)
        plt.xlabel("discrete speed")
        plt.ylabel("mean absolute error")
        plt.legend()
        
    def __plotDiscreteSpeedError(self,ax,name=""):
        """
        (deprecated) dropped in favor of faster numpy based function
        plots average absolute error per discrete speed 
        """
        error = self.y -self.preds
        y_error_df=pd.DataFrame([self.y.flatten(),error.flatten()],index=["y","error"+"_"+name]).T
        y_error_df.abs().round().groupby("y").mean().plot(ax=ax)

    def plotSegmentSeries(self,idx,subplot=False,plot_error=False,plot_surface=False):
        """
        plot the series of a segment (both true and predicted values)
        """
        
        
        if not subplot :
            plt.figure(figsize=(30,4))
            
        try :
            ys = self.data_model.getSplitSequences(
                                                    self.y[:,idx],
                                                    self.data_model.sequence_length-self.data_model.input_lag-self.data_model.output_lag+1,
                                                    skip=self.data_model.input_lag)
        except (ValueError ,AttributeError):
            plt.plot(self.y[:,idx] )
            plt.plot(self.preds[:,idx])
            return
            


        
        preds = self.data_model.getSplitSequences(
                                                self.preds[:,idx],
                                                self.data_model.sequence_length-self.data_model.input_lag-self.data_model.output_lag+1,
                                                skip=self.data_model.input_lag)
        if plot_error :
            plt.plot(ys[0],ys[1]-preds[1])
        elif plot_surface:
            plt.fill_between(ys[0],ys[1],preds[1])
        else :
            plt.plot(*ys )

            plt.plot(*preds)
        
        dates = np.array([self.data_model.getIndexes(i)[1][0] for i in range(len(self.y))])
        
        if not subplot :
            plt.xticks(ticks  = np.arange(len(self.y))[np.r_[:len(self.y)-1:30j].astype(int)],
                       labels = dates[np.r_[:len(self.y)-1:30j].astype(int)],rotation='vertical');
        plt.ylabel("Speed")
        plt.axvline(self.split_idx,c='r')
        if not plot_error :
            plt.legend(['y','pred','validationSplit'])
        plt.title(" segment : {}, tag : {:}".format(idx,self.data_cleaner.segments_tags.iloc[idx]))

        
    def plotMultipleSegmentsSeries(self,ids=None,plot_error=False,plot_surface=False):
        """
        plots multiple series using index in "ids" if provided else plots 20 series ordered by mean difference in  predictions
        """
        if ids is None : 
            try :
                ids = np.argsort(self.data_model.y.mean(axis=0)[0]-self.data_model.predict('full').mean(axis=0))[np.r_[:self.data_model.n_segments-1:20j].astype(int)]
            except AttributeError :
                ids = np.random.randint(0,len(self.y),10)
        plt.figure(figsize=(24,36))
        for ix, xSample in enumerate(ids):
            plt.subplot(len(ids),1,ix+1)
            self.plotSegmentSeries(xSample,subplot=True,plot_error=plot_error,plot_surface=plot_surface)
        plt.tight_layout()
        
        
        
    def plotPredictionMatchHeatMap(self,split='full',size_rate=4,figsize=(10,10)):
        """
        density plot of rounded values of predictions and true data
        """
        try :
#             train_split = int(len(self.y)*self.data_model.valid_split)
            train_split = self.split_idx

        except AttributeError :
            train_split = len(self.y)
        if split.lower() == 'train':
            prdsVsYDF=pd.DataFrame([self.preds[:train_split].flatten(),self.y[:train_split].flatten()],index=['pred','y'])
        
        if split.lower()[:5]=='valid':
            prdsVsYDF=pd.DataFrame([self.preds[train_split:].flatten(),self.y[train_split:].flatten()],index=['pred','y'])
        
        if split.lower() =='full':
            prdsVsYDF=pd.DataFrame([self.preds.flatten(),self.y.flatten()],index=['pred','y'])
        prdsVsYDF=prdsVsYDF.T.astype(int)
        
        fig= plt.figure(figsize=figsize)
        gs = matplotlib.gridspec.GridSpec(size_rate, size_rate)
        ax_main = plt.subplot(gs[1:, :-1])
        ax_x_hist = plt.subplot(gs[0, :-1],sharex=ax_main)
        ax_y_hist = plt.subplot(gs[1:, -1],sharey=ax_main)     
        ax_x_hist.hist(prdsVsYDF.pred.values,bins=len(set(prdsVsYDF.pred.values)),align='mid')
        ax_y_hist.hist(prdsVsYDF.y.values,bins=len(set(prdsVsYDF.y.values)),align='mid',orientation='horizontal')  


        prdsVsYDF=prdsVsYDF.groupby(['pred','y']).size().unstack().fillna(0).T

        prdmin  = -prdsVsYDF.columns.values.min()


        heat_map=ax_main.imshow(prdsVsYDF,aspect='auto',origin='bottom-left',cmap =plt.cm.gist_ncar,interpolation='spline16')
        ax_main.set_xticks(np.arange(len(prdsVsYDF.columns.values))[::12])
        ax_main.set_xticklabels(labels=prdsVsYDF.columns.values[::12])
        ax_main.plot([prdmin,130+prdmin],[0,130],c='red',linewidth=3)
        ax_main.set(xlabel="x Prediction", ylabel="y True")
        plt.colorbar(heat_map,ax=ax_y_hist)
        plt.tight_layout()
        
        
    def plotPredictions(self, yDF,predDF, timesteps,folium_map=None):
        """
        creates a map representing the error in prediction
        """
        
        if folium_map == None :
            folium_map = Plotting.getFoliumMap()
        layers=[]
        colors = ((np.abs(yDF.clip(lower=15) - predDF.clip(lower=15))+1)/(yDF.clip(lower=15)+1)).clip(upper=1)
        laggedX = self.data_model.restoreXAsDF(self.data_model.x)
        predSegs = self.data_cleaner.segmentsMeta[self.data_cleaner.segmentsMeta.segmentID.isin(self.data_cleaner.mergedIndex[self.data_cleaner.mergedIndex.isin(yDF.index)].index)]
        segment_tag=predSegs.tag.apply(lambda x:x['highway']).values
        segment_overall_mean = [self.data_model.data.mean(axis=1).loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
        segment_timestamp_mean = self.data_model.data.groupby(pd.DatetimeIndex(self.data_model.data.columns).time,axis=1).mean()
        
        for t in timesteps :
            colorList=[colors[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
            y= [yDF[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
            preds = [predDF[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
            segCounts=[self.data_cleaner.counts[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
            timestampLaggedX= [laggedX[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]

            current_segment_timestamp_mean = [segment_timestamp_mean[t.time()].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]

            popups = ["segment : {:},<br>tag : {:},<br> y : {:.2f},<br> pred : {:.2f},<br> %error : {:.0f}%,<br> count : {:}<br>mean: {:}<br>timestamp_mean: {:}<br>x: {:} "\
                      .format(seg,seg_tag,yi,predi,props*100,count,mean,timestamp_mean,np.array(x).astype(int)) 
                      for seg,seg_tag,yi,predi,props,count,mean,timestamp_mean,x 
                      in zip(predSegs.segmentID,segment_tag,y,preds,colorList,segCounts,segment_overall_mean,current_segment_timestamp_mean,timestampLaggedX)]
            pos = yDF.columns.get_loc(t)
            layer = self.getPredictionLayer(predSegs,colorList,segCounts,folium_map,str(t),popups)
            layers.append(layer)

        return Plotting.stackHistotyLayers([*layers,folium.TileLayer()],folium_map)

    def getPredictionLayer(self,segments,colors,counts,folium_map,name='layer',popups=[]):
        """
        create folium layer for one timestamp prediction
        """
        layer = folium.plugins.FeatureGroupSubGroup(folium_map,name=name,show=False, overlay=False)
        [folium.PolyLine(locations=[lo[::-1] for lo in x['coordinates']],weight = count//5+1, color=matplotlib.colors.rgb2hex(plt.cm.brg_r(color/2)),popup=pop).add_to(layer) for x,color,pop,count in zip(segments['loc'],colors,popups,counts)]
        return layer


    def cdfPlot(self,lower_clip_value=15,upper_clip_value=130,error_type ="mape",label="model",plot_lines=True):
        """
        cumulative distribution plot of the given *error_type* for the current model
        
        TODO : add split params for train/validation output
        """
        if error_type.lower() == "mape":
            error = (abs(self.y.clip(lower_clip_value,upper_clip_value) - self.preds.clip(lower_clip_value,upper_clip_value))/(self.y.clip(lower_clip_value,upper_clip_value))).flatten()
        if error_type.lower() =="mae":
            error = abs(self.y.clip(lower_clip_value,upper_clip_value) - self.preds.clip(lower_clip_value,upper_clip_value)).flatten()
        if error_type.lower() =="mse":
            error = ((self.y.clip(lower_clip_value,upper_clip_value) - self.preds.clip(lower_clip_value,upper_clip_value))**2).flatten()

        error.sort()
        idx_error=np.cumsum(np.arange(len(error)))
        plt.plot(error,idx_error/idx_error.max(),label=label)
        if plot_lines :
            plt.axvline(0.5,c="red",label="0.5")
            plt.axvline(0.25,c="green",label="0.25")
            plt.axvline(0.15,c="pink",label="0.15")
        plt.ylabel("cumulative probability")
        plt.xlabel(error_type)
        plt.title("CDF")
        plt.legend()
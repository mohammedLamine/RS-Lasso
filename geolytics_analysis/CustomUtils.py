import json
import pandas as pd
import numpy as np
from vincenty import vincenty
import matplotlib.pyplot as plt
import datetime
from matplotlib.colors import rgb2hex

import time

global oldTime
oldTime=-1

def getTimeSpent(reset=False):
    global oldTime

    if oldTime == -1 or reset:
        oldTime =time.time()
        return "first time call or reset : starting counter"
    currentTime = time.time()
    diff = currentTime - oldTime
    oldTime = currentTime
    return 'time spent {:.1f} s'.format(diff)

def createIrisPopulationCollection(path,db, name=None):
    """
    Create collection from excel file
    
    path : str
        the path to excel file (must be of the same format as Population en 2015 - IRIS https://www.insee.fr/fr/statistiques/fichier/3627376/base-ic-evol-struct-pop-2015.zip)
    
    db: MongoClient.database
        data base where to create the collection
    
    name : str or None
        the name of the collection, if None the name is infered from the path
    
    """
    if not name :
        name=path.split('/')[-1].split('.')[-2]
    irisPop = db[name]
    records= pd.read_excel(path, encoding="utf8", skiprows=5, header=0).iloc[:,:13].apply(lambda x : json.loads(x.to_json(force_ascii=False)),axis=1).tolist()
    irisPop.insert_many(records)
    
def timeToTimeDelta(timeSerie):
    """
    extract time from timestamp and return it as a delta time
    
    timeSeries : pandas Series
        series of timestamps
    """
    return pd.to_timedelta(pd.to_timedelta(timeSerie).dt.seconds,unit='s')
def timeToSeconds(ts):
    """
    extract seconds out of time
    
    ts : datetime.time
        time
    """
    return ts.hour*3600+ts.minute*60+ts.second
def secondsToTime(secs):
    """
    create time out of seconds
    
    secs : int
        number of seconds
    """
    return datetime.time(int(secs/3600),int(secs%3600/60),int(secs%60))

def getClustersColors(clusters,cmap=plt.cm.hot):
    """
    return colors for clusters
    clusters : array of int
        clusters labels
    cmap : matplotlib cmap
        color map used to generate colors
    """
    clusters+=1
    return np.array([rgb2hex(color) for color in cmap(clusters/(max(clusters)+1))])

def reverseVincenty(a,b):
    """ 
    Vincenty distance adapted to the order of lon/lat in mongo db
    
    a,b : array on length 2 (longitude, latitude)
 
    returns the distance between the two points     
    """
    return vincenty(a[::-1],b[::-1]) 


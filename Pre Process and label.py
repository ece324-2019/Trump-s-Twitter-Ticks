import pandas as pd
import datetime

data = pd.read_csv('dowdata.csv')
for i in range(len(data)):
    if(data['Time'][i][0:2]=='20'):
        data.at[i,'Time'] = (data['Time'][i][5:7]+'/'+data['Time'][i][2:4]+'/'+data['Time'][i][8:])
    else:
        data.at[i, 'Time'] = (data['Time'][i][3:5] + '/' + data['Time'][i][0:2] + data['Time'][i][5:])
    data.at[i,'Time'] = (data['Time'][i][6:8]+data['Time'][i][2:6]+data['Time'][i][:2]+data['Time'][i][8:])
    data.at[i,'Time'] = '20'+data['Time'][i].replace('/','-')
changes=[]
changesBinary=[]
for i in range(len(data)):
    if(i<2 or i>(len(data)-5)):
        changes.append(0)
    else:
        changes.append(sum(data['Change'][i+1:i+4])-sum(data['Change'][i-2:i+1]))
    changesBinary.append(1 if changes[i-1] > 0 else 0)
data['3 before/3 after change'] = changes
data['Up or Down'] = changesBinary
dates=[]

wallSt = data
data = pd.read_csv('tweets_datset.csv')
from datetime import timedelta
BST = timedelta(hours=5)
for i in range(len(data)):
    year = int(data['created_at'][i][:4])
    month = int(data['created_at'][i][5:7])
    day = int(data['created_at'][i][8:10])
    hours = int(data['created_at'][i][11:13])
    minutes = int(data['created_at'][i][14:16])
    seconds = int(data['created_at'][i][17:19])
    dates.append(datetime.datetime(year=year,month=month,day=day,hour=hours,minute=minutes,second=seconds)-BST)
data['datetime']=dates

datesW=[]
for i in range(len(wallSt)-1):
    year = int(wallSt['Time'][i][:4])
    month = int(wallSt['Time'][i][5:7])
    day = int(wallSt['Time'][i][8:10])
    hours = int(wallSt['Time'][i][11:-3])
    minutes = 0
    seconds = 0
    datesW.append(datetime.datetime(year=year, month=month, day=day, hour=hours, minute=minutes, second=seconds))
datesW.append(datetime.datetime(year=1000, month=1, day=1, hour=1, minute=1, second=1))
wallSt['Time']=datesW

def findLabel(dates,date):
    for i in range(len(dates)-1):
        if(date>dates[i+1]):
            return i
    return 0
labels=[]
for i in range(len(data)):
    value = findLabel(wallSt['Time'].values,data['datetime'][i])
    labels.append(wallSt['Up or Down'][value])
data['labels']=labels

data.to_csv('labeledTweets.csv')


import pandas as pd
import datetime
from datetime import timedelta

def genLabels(tweetData):
    data = pd.read_csv('sANDp.csv')
    for i in range(len(data)):
        if(data['Time'][i][0:2]=='20'):
            data.at[i,'Time'] = (data['Time'][i][5:7]+'/'+data['Time'][i][2:4]+'/'+data['Time'][i][8:])
        else:
            data.at[i, 'Time'] = (data['Time'][i][3:5] + '/' + data['Time'][i][0:2] + data['Time'][i][5:])
        data.at[i,'Time'] = (data['Time'][i][6:8]+data['Time'][i][2:6]+data['Time'][i][:2]+data['Time'][i][8:])
    data.at[i,'Time'] = '20'+data['Time'][i].replace('/','-')

    dates=[]
    wallSt = data
    BST = timedelta(hours=5)
    for i in range(len(tweetData)):
        year = int(str(tweetData['created_at'][i])[:4])
        month = int(str(tweetData['created_at'][i])[5:7])
        day = int(str(tweetData['created_at'][i])[8:10])
        hours = int(str(tweetData['created_at'][i])[11:13])
        minutes = int(str(tweetData['created_at'][i])[14:16])
        seconds = int(str(tweetData['created_at'][i])[17:19])
        dates.append(datetime.datetime(year=year,month=month,day=day,hour=hours,minute=minutes,second=seconds)-BST)
    tweetData['datetime']=dates

    datesW=[]
    for i in range(len(wallSt)-1):
        year = 2000 + int(wallSt['Time'][i][:2])
        month = int(wallSt['Time'][i][3:5])
        day = int(wallSt['Time'][i][6:8])
        hours = int(wallSt['Time'][i][9:-3])
        minutes = 0
        seconds = 0
        datesW.append(datetime.datetime(year=year, month=month, day=day, hour=hours, minute=minutes, second=seconds))
    datesW.append(datetime.datetime(year=1000, month=1, day=1, hour=1, minute=1, second=1))
    wallSt['Time']=datesW

    print(wallSt.columns)
    print(tweetData.columns)
    casual = []
    for i in range(len(wallSt)):
        if(i<2 or i>len(wallSt)-3):
            casual.append(0)
        else:
            casual.append(sum(wallSt['Change'][i:i+3])/3-sum(wallSt['Change'][i-2:i+1])/3)
    wallSt['second derivative'] = casual
    print(casual)
    binary=[]
    for i in range(len(casual)):
        if(casual[i]>.2):
            binary.append(1)
        elif(casual[i]<-.2):
            binary.append(-1)
        else:
            binary.append(0)
    wallSt['binary']=binary
    tweeters=[]
    j = 0
    for i in range(len(tweetData)):
        done=0
        while(done==0):
            if(wallSt['Time'][j]<tweetData['datetime'][i] and done==0):
                print(wallSt['Time'][j])
                print(tweetData['datetime'][i])
                done = 1
                tweeters.append(wallSt['binary'][j+1])
                print(wallSt['binary'][j+1])
                j-=1
            if(j==len(wallSt)-1):
                done = 1
                tweeters.append(0)
                j-=1
            j+=1
    tweetData['label']=tweeters
    onehot = []
    for i in range(len(tweetData)):
        if (tweetData['label'][i] == -1):
            onehot.append([1,0,0])
        if (tweetData['label'][i] == 0):
            onehot.append([0,1,0])
        if (tweetData['label'][i] == 1):
            onehot.append([0,0,1])
    tweetData['onehot'] = onehot
    return tweetData
tweets = pd.read_json('trump_tweets_json.json')
tweets = tweets[['created_at', 'text']]
data = genLabels(tweets)
data.to_csv(r'labeledSNP.csv')

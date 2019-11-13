import pandas as pd

def Baseline(tweets):
    output = []
    for tweet in tweets:
        tweet = tweet.lower()
        if 'tax cut' in tweet:
            output.append(1)
        elif 'doing great' in tweet:
            output.append(1)
        elif 'stock market' in tweet:
            output.append(1)
        elif 'trade war' in tweet:
            output.append(0)
        elif 'tariff' in tweet:
            output.append(0)
        elif 'china' in tweet:
            output.append(0)
        elif 'north korea' in tweet:
            output.append(0)
        elif 'deregulation' in tweet:
            output.append(1)
        elif 'trade deal' in tweet:
            output.append(1)
        elif 'kim jong' in tweet:
            output.append(0)
        elif 'unemployment' in tweet:
            output.append(1)
        elif 'democrat' in tweet:
            output.append(0)
        elif 'pelosi' in tweet:
            output.append(0)
        elif 'hillary' in tweet:
            output.append(0)
        elif 'jobs' in tweet:
            output.append(1)
        elif 'back' in tweet:
            output.append(1)
        elif 'economy' in tweet:
            output.append(1)
        elif 'new york times' in tweet:
            output.append(0)
        elif 'fake news' in tweet:
            output.append(0)
        elif 'waashington post' in tweet:
            output.append(0)
        elif 'president xi' in tweet:
            output.append(0)
        elif 'trump' in tweet:
            output.append(1)
        elif 'obama' in tweet:
            output.append(0)
        elif 'great job' in tweet:
            output.append(1)
        elif 'cnn' in tweet:
            output.append(0)
        elif 'fox' in tweet:
            output.append(0)
        elif 'nytimes' in tweet:
            output.append(0)
        elif 'wall' in tweet:
            output.append(0)
        elif 'polls' in tweet:
            output.append(0)
        else:
            output.append(1 if len(tweet)<100 else 0)
    return output
data = pd.read_json('trump_tweets_json.json')
y = Baseline(data['text'])
for i in range(len(y)):
    print(data['text'][i]+'  '+str(y[i]))


# using tweeter api: https://developer.twitter.com/en/docs/tweets/post-and-engage/api-reference/get-statuses-lookup
# data: http://pan.webis.de/clef16/pan16-web/author-profiling.html

import os
import xml.etree.ElementTree
from codecs import open

import pandas as pd
import tweepy
from tqdm import tqdm

consumer_key = 'YOUR_KEY_HERE'
consumer_secret = 'YOUR_KEY_HERE'
access_token = 'YOUR_KEY_HERE'
access_token_secret = 'YOUR_KEY_HERE'
project = 'PATH/TO/PROJECT'
age_dir = project + 'data/raw/pan16-author-profiling-training-dataset-2016-04-25/pan16-author-profiling-training' \
                    '-dataset-english-2016-04-25/'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

with open(age_dir + 'truth.txt', 'r') as f:
    lines = f.readlines()

gender_dic = {'MALE': 'male', 'FEMALE': 'female'}
age_dic = {'18-24': 0, '25-34': 1, '35-49': 2, '50-64': 3, '65-xx': 4}

d = {}
for line in lines:
    x = line.strip().split(':::')
    d[x[0]] = [gender_dic[x[1]], age_dic[x[2]]]

for filename in tqdm(os.listdir(age_dir)):
    try:
        if filename == 'truth.txt':
            continue
        else:
            tweets_id = []
            tweets = []
            e = xml.etree.ElementTree.parse(age_dir + filename).getroot()
            for atype in e.findall('documents')[0].findall('document'):
                tweets_id.append(int(atype.get('id')))
                if len(tweets_id) == 100:
                    ans = api.statuses_lookup(tweets_id)
                    for a in ans:
                        tweets.append(a.text)
                    tweets_id = []
            if len(tweets_id) > 0:
                ans = api.statuses_lookup(tweets_id)
                for a in ans:
                    tweets.append(a.text)
            d[filename.split('.')[0]].append(tweets)
    except:
        pass

ids, genders, ages, texts = [], [], [], []
for k, v in d.items():
    if len(v) < 3:
        print 'no tweets of this user', k
        continue
    for tweet in v[2]:
        ids.append(k)
        genders.append(v[0])
        ages.append(v[1])
        texts.append(tweet)

data = {'id': ids, 'gender': genders, 'age': ages, 'text': texts}
df = pd.DataFrame(data=data)
df.to_csv(project + 'data/interim/author-profiling.tsv', sep='\t', encoding='utf-8')

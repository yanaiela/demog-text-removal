import pandas as pd
import numpy as np

import sklearn.utils

import os

from tqdm import tqdm

from twitter_utils import normalize_text
from twitter_utils import MENTION

CONF_LEVEL = 0.8
SEED = 16


def get_data(input_file):
    """
    reading the DIAL data, removing dups
    :param input_file: path to the DIAL data
    :return: a cleaned dataframe
    """
    df = pd.read_csv(input_file, sep='\t',
                     names=['tid', 'tts', 'uid', 'tll', 'tcb', 'text', 'aa', 'i2', 'i3', 'wh'],
                     usecols=['uid', 'text', 'aa', 'i2', 'i3', 'wh'], encoding='utf8')
    df = df.drop_duplicates(subset=['text'], keep=False)
    # fix of pd file reading
    df = df[pd.to_numeric(df['wh'], errors='coerce').notnull()]
    df['wh'] = df.wh.astype(np.float64)
    df = sklearn.utils.shuffle(df, random_state=SEED)
    return df


def mention_split(data, min_len=1):
    """
    creating pos and neg examples for the `tweet-mention-prediction'
    :param data: words array
    :param min_len: minimum length of sentence to keep
    :return: positive and negative lists
    """
    pos = []
    neg = []
    for sen in tqdm(data):
        clean = [x for x in sen if not x == MENTION]
        if len(clean) < min_len: continue
        if len(set(clean)) == 1 and clean[0] == MENTION: continue
        if len(clean) != len(sen):
            pos.append(clean)
        else:
            neg.append(sen)
    return pos, neg


def get_race(df, min_len=1):
    """
    creating pos and neg examples for the binary race prediction
    :param df: dataframe with the race probabilities
    :param min_len: minimum length of sentence to keep
    :return: positive and negative lists
    """
    wh_data = []
    aa_data = []
    white = df[(df.wh > CONF_LEVEL)]
    white = white.filter(items=['text'])
    for ind in tqdm(range(len(white))):
        try:
            t = normalize_text(white.iloc[ind].text)
            if len(t) < min_len:
                continue
            if len(set(t)) == 1 and t[0] == MENTION: continue
            wh_data.append(t)
        except:
            pass

    aa = df[(df.aa > CONF_LEVEL)]
    aa = aa.filter(items=['text'])
    for ind in tqdm(range(len(aa))):
        try:
            t = normalize_text(aa.iloc[ind].text)
            if len(t) < min_len:
                continue
            if len(set(t)) == 1 and t[0] == MENTION: continue
            aa_data.append(t)
        except:
            pass
    return wh_data, aa_data


def get_sentiment(df, emotions, other_emotions, min_len=1):
    """
    creating pos examples for the binary emoji-based sentiment prediction
    :param df: dataframe
    :param emotions: list of possible emotions for the current class
    :param other_emotions: list of emotions for the other class, of which, upon
                            a shared emoji, the example will be discarded
    :param min_len: minimum length of sentence to keep
    :return: examples of the same class
    """
    data = []
    for sentiment in tqdm(emotions):
        res = df[df['text'].str.contains(sentiment, na=False)]
        for ind in range(len(res)):
            try:
                t = normalize_text(res.iloc[ind].text)
                if not set(t).isdisjoint(other_emotions):  # there's more than one sentiment emoji
                    continue
                if len(t) < min_len:
                    continue
                if len(set(t)) == 1 and t[0] == MENTION: continue
                data.append(t)
            except:
                pass
    return data


# collect data from some type above a certain confidence
def get_attr_sentiments(df, emotions, other_emotions, col_name, min_len=1):
    col = df[(df[col_name] > CONF_LEVEL)]
    return get_sentiment(col, emotions, other_emotions, min_len)


# writing the examples and vocabulary to files
def to_file(output_dir, voc2id, vocab, pos_pos, pos_neg, neg_pos, neg_neg):
    if output_dir[-1] != '/':
        output_dir += '/'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + 'vocab', 'w') as f:
        f.writelines('\n'.join(vocab))

    for data, name in zip([pos_pos, pos_neg, neg_pos, neg_neg], ['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg']):
        with open(output_dir + name, 'w') as f:
            for sen in data:
                ids = map(lambda x: str(voc2id[x]), sen)
                f.write(' '.join(ids) + '\n')


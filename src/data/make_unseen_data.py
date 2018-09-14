import sys

import sklearn.utils
from tqdm import tqdm

project = 'PATH/TO/PROJECT'
sys.path.insert(0, project + '/src/models')

from data_utils import get_data, CONF_LEVEL, normalize_text, to_file
from twitter_utils import happy, sad, MENTION
from data_handler import read_files

SEED = 16


df = get_data(project + '/path/to/downloaded/twitteraae_all')

emotions = happy + sad
cleaned = df[~df.text.str.contains('|'.join(emotions))]

cleaned = sklearn.utils.shuffle(cleaned, random_state=SEED)

with open(project + '/data/processed/sent_race/vocab', 'r') as f:
    vocab = f.readlines()
    vocab = map(lambda s: s.strip(), vocab)


vocab_d = {x: 0 for x in vocab}
pos_wh, pos_aa, neg_wh, neg_aa = read_files(project + '/data/processed/sent_race/')

prev_sent = {}
for s in pos_wh + pos_aa + neg_wh + neg_aa:
    try:
        sen = ' '.join([str(vocab[w]) for w in s])
        prev_sent[sen] = 0
    except:
        pass


def get_race(df, min_len=1):
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
            if not all(x in vocab_d for x in t): continue

            try:
                s = ' '.join([w for w in t])
                if s in prev_sent: continue
            except:
                continue

            wh_data.append(t)
        except:
            pass
        if len(wh_data) >= 100000: break

    print 'reached 100k after {0} tweets'.format(ind)
    aa = df[(df.aa > CONF_LEVEL)]
    aa = aa.filter(items=['text'])
    for ind in tqdm(range(len(aa))):
        try:
            t = normalize_text(aa.iloc[ind].text)
            if len(t) < min_len:
                continue
            if len(set(t)) == 1 and t[0] == MENTION: continue
            if not all(x in vocab_d for x in t): continue

            try:
                s = ' '.join([w for w in t])
                if s in prev_sent: continue
            except:
                continue

            aa_data.append(t)
        except:
            pass
        if len(aa_data) >= 100000: break
    print 'reached 100k after {0} tweets'.format(ind)
    return wh_data, aa_data


wh, aa = get_race(cleaned, 3)

id2voc = dict(enumerate(vocab))
voc2id = {v: k for k, v in id2voc.iteritems()}

pos_pos, neg_pos = aa[:50000], aa[50000:]
pos_neg, neg_neg = wh[:50000], wh[50000:]

to_file(project + '/data/processed/unseen_race/', voc2id, vocab, pos_pos, pos_neg, neg_pos, neg_neg)

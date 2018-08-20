import pandas as pd
import sklearn.utils
from sklearn.utils import shuffle
from tqdm import tqdm

from data_utils import mention_split, normalize_text, MENTION, to_file, SEED

project = 'PATH/TO/PROJECT'
MIN_SENTENCE_LEN = 3

df = pd.read_csv(project + 'data/interim/author-profiling.tsv', sep='\t',
                 names=['index', 'age', 'gender', 'id', 'text'], skiprows=1, index_col=False, lineterminator='\n',
                 encoding='utf8')

df = sklearn.utils.shuffle(df, random_state=SEED)


def tokenize(data, min_len=1):
    arr = []
    ids = []
    for ind in tqdm(range(len(data))):
        try:
            t = normalize_text(data.iloc[ind].text)
            if len(t) < min_len:
                continue
            if len(set(t)) == 1 and t[0] == MENTION: continue
            arr.append(t)
            ids.append(data.iloc[ind].id)
        except:
            pass
    return arr, ids


# Gender
males, m_ids = tokenize(df[df['gender'] == 'male'], MIN_SENTENCE_LEN)
_, males = zip(*sorted(zip(m_ids, males)))

females, f_ids = tokenize(df[df['gender'] == 'female'], MIN_SENTENCE_LEN)
_, females = zip(*sorted(zip(f_ids, females)))

train_pos_m, train_neg_m = mention_split(males[:100000], min_len=MIN_SENTENCE_LEN)
test_pos_m, test_neg_m = mention_split(males[102000:], min_len=MIN_SENTENCE_LEN)
train_pos_m = shuffle(train_pos_m, random_state=SEED)
train_neg_m = shuffle(train_neg_m, random_state=SEED)

train_pos_f, train_neg_f = mention_split(females[:92000], min_len=MIN_SENTENCE_LEN)
test_pos_f, test_neg_f = mention_split(females[94000:], min_len=MIN_SENTENCE_LEN)
train_pos_f = shuffle(train_pos_f, random_state=SEED)
train_neg_f = shuffle(train_neg_f, random_state=SEED)

train_size = 40000
sentences = train_pos_m + train_pos_f + train_neg_m + train_neg_f + test_pos_m + test_pos_f + test_neg_m + test_neg_f
vocab = list(set([item for sublist in sentences for item in sublist]))
id2voc = dict(enumerate(vocab))
voc2id = {v: k for k, v in id2voc.iteritems()}

to_file(project + 'data/processed/author_mention_gender/', voc2id, vocab, train_pos_m[:train_size] + test_pos_m,
        train_pos_f[:train_size] + test_pos_f, train_neg_m[:train_size] + test_neg_m,
        train_neg_f[:train_size] + test_neg_f)


young, y_ids = tokenize(df[(df['age'] == 0) | (df['age'] == 1)], MIN_SENTENCE_LEN)
_, young = zip(*sorted(zip(y_ids, young)))

old, o_ids = tokenize(df[(df['age'] == 2) | (df['age'] == 3) | (df['age'] == 4)], MIN_SENTENCE_LEN)
_, old = zip(*sorted(zip(o_ids, old)))

train_pos_y, train_neg_y = mention_split(young[6500:], min_len=MIN_SENTENCE_LEN)
test_pos_y, test_neg_y = mention_split(young[:6000], min_len=MIN_SENTENCE_LEN)
train_pos_y = shuffle(train_pos_y, random_state=SEED)
train_neg_y = shuffle(train_neg_y, random_state=SEED)

train_pos_o, train_neg_o = mention_split(old[:110000], min_len=MIN_SENTENCE_LEN)
test_pos_o, test_neg_o = mention_split(old[112000:], min_len=MIN_SENTENCE_LEN)
train_pos_o = shuffle(train_pos_o, random_state=SEED)
train_neg_o = shuffle(train_neg_o, random_state=SEED)

sentences = train_pos_y + train_neg_y + train_pos_o + train_neg_o + test_pos_y + test_neg_y + test_pos_o + test_neg_o
vocab = list(set([item for sublist in sentences for item in sublist]))
id2voc = dict(enumerate(vocab))
voc2id = {v: k for k, v in id2voc.iteritems()}

to_file(project + 'data/processed/author_mention_age/', voc2id, vocab, train_pos_y[:train_size] + test_pos_y,
        train_pos_o[:train_size] + test_pos_o, train_neg_y[:train_size] + test_neg_y,
        train_neg_o[:train_size] + test_neg_o)

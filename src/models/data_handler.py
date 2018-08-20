LEN_THRESHOLD = 10


def read_files(data_dir):
    with open(data_dir + 'pos_pos', 'r') as f:
        pos_pos = f.readlines()
        pos_pos = [map(int, sen.split(' ')) for sen in pos_pos]
    with open(data_dir + 'pos_neg', 'r') as f:
        pos_neg = f.readlines()
        pos_neg = [map(int, sen.split(' ')) for sen in pos_neg]
    with open(data_dir + 'neg_pos', 'r') as f:
        neg_pos = f.readlines()
        neg_pos = [map(int, sen.split(' ')) for sen in neg_pos]
    with open(data_dir + 'neg_neg', 'r') as f:
        neg_neg = f.readlines()
        neg_neg = [map(int, sen.split(' ')) for sen in neg_neg]
    return pos_pos, pos_neg, neg_pos, neg_neg


# positive: 1, negative: 0
# afro-american: 1, white: 0
def get_labeled_data(pos_pos, pos_neg, neg_pos, neg_neg, total, train_s):
    x_train = []
    x_test = []

    for x in pos_pos[:train_s]:
        x_train.append((x, 1, 1))
    for x in pos_pos[train_s:total]:
        x_test.append((x, 1, 1))

    for x in pos_neg[:train_s]:
        x_train.append((x, 1, 0))
    for x in pos_neg[train_s:total]:
        x_test.append((x, 1, 0))

    for x in neg_pos[:train_s]:
        x_train.append((x, 0, 1))
    for x in neg_pos[train_s:total]:
        x_test.append((x, 0, 1))

    for x in neg_neg[:train_s]:
        x_train.append((x, 0, 0))
    for x in neg_neg[train_s:total]:
        x_test.append((x, 0, 0))

    return x_train, x_test


def get_unbalanced(task, pos_pos, pos_neg, neg_pos, neg_neg):
    """
    creating dataset for the unbalanced setup.
    hard-coded numbers were selected to maximize the examples,
    while keeping `pretty' numbers
    """
    x_train = []
    x_test = []

    if 'age' in task or 'mention2' in task:
        lim_1_train = 40000
        lim_1_test = 42000
        lim_2_train = 10000
        lim_2_test = 11000
    else:
        lim_1_train = 66400
        lim_1_test = 70400
        lim_2_train = 16600
        lim_2_test = 17600

    for x in pos_pos[:lim_1_train]:
        x_train.append((x, 1, 0))
    for x in pos_pos[lim_1_train:lim_1_test]:
        x_test.append((x, 1, 0))

    for x in pos_neg[:lim_2_train]:
        x_train.append((x, 1, 1))
    for x in pos_neg[lim_2_train:lim_2_test]:
        x_test.append((x, 1, 1))

    for x in neg_pos[:lim_2_train]:
        x_train.append((x, 0, 0))
    for x in neg_pos[lim_2_train:lim_2_test]:
        x_test.append((x, 0, 0))

    for x in neg_neg[:lim_1_train]:
        x_train.append((x, 0, 1))
    for x in neg_neg[lim_1_train:lim_1_test]:
        x_test.append((x, 0, 1))

    return x_train, x_test


def get_data(main_task, data_dir):
    """
    routing the task and dir to the corresponding function.
    returning train and test data
    """
    train_s = 41500
    if 'unbalanced' in main_task:
        total = 72000
    else:
        total = 44000

    pos_pos, pos_neg, neg_pos, neg_neg = read_files(data_dir)

    if 'unbalanced' in main_task:
        return get_unbalanced(main_task, pos_pos, pos_neg, neg_pos, neg_neg)
    elif 'age' in main_task or 'mention2' in main_task:
        return get_labeled_data(pos_pos, pos_neg, neg_pos, neg_neg, 42500, 40000)
    else:
        return get_labeled_data(pos_pos, pos_neg, neg_pos, neg_neg, total, train_s)





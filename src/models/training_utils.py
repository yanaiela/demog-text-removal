# The 2 in the gender dictionary keys are due to legacy usage of gender

import logging


task_dic = {'sentiment': 'sent_race', 'mention': 'mention_race', 'race': 'sent_race',

            'race_sent': 'race_sent',

            'sent_race': 'sent_race',
            'unbalanced_race': 'sent_race',
            'unbalanced_mention_race': 'mention_race', 'unbalanced_mention_gender': 'mention_gender',
            'mention_race': 'mention_race',
            'ment_race': 'mention_race',

            'unseen_race': 'unseen_race',

            'mention2_gender': 'author_mention_gender', 'mention_age': 'author_mention_age',
            'unbalanced_mention2_gender': 'author_mention_gender', 'unbalanced_mention_age': 'author_mention_age',
            'ment2_gender': 'author_mention_gender', 'ment_age': 'author_mention_age',

            }


def get_logger(task, model_dir):
    logger = logging.getLogger(task)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(model_dir + '/' + task + '.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

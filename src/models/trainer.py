"""
Usage:
  sent_demog_train.py [--dynet-mem=MEM] [--dynet-seed=SEED] [--dynet-autobatch=NUM] [--dynet-gpus=NUM] [--dynet-devices=DEVICE]
  [--epochs=EPOCHS] [--ro=RO] [--task=TASK] [--type=TYPE] [--num_adv=NUM_ADV] [--adv_depth=NUM_HID_LAY]
  [--lstm_size=LSTM_SIZE] [--dropout=DROPOUT] [--rnn_dropout=RNN_DROPOUT] [--rnn_type=RNN_TYPE]
  [--enc_size=ENC_SIZE] [--adv_size=ADV_SIZE] [--vec_drop=DROPOUT] [--init=INIT]


Options:
  -h --help                     show this help message and exit
  --dynet-mem=MEM               allocates MEM bytes for dynet
  --dynet-seed=SEED             dynet random seed
  --dynet-autobatch=NUM         autobatch for dynet
  --dynet-gpus=NUM              use gpu
  --dynet-devices=DEVICE        device to use
  --epochs=EPOCHS               amount of training epochs [default: 100]
  --ro=RO                       amount of power to the adversarial
  --task=TASK                   single task to train [default: sentiment]
  --type=TYPE                   type of the task (1/2) [default: 1]
  --num_adv=NUM_ADV             number of simultaneous adversarials trainer [default: 1]
  --adv_depth=NUM_HID_LAY       number of hidden layers of adversarials [default: 1]
  --lstm_size=LSTM_SIZE         size of lstm layer [default: 1]
  --dropout=DROPOUT             dropout probability [default: 0.2]
  --rnn_dropout=RNN_DROPOUT     rnn dropout [default: 0.0]
  --rnn_type=RNN_TYPE           type of rnn - lstm/gru [default: lstm]
  --enc_size=ENC_SIZE           size of lstm hidden layer [default: 300]
  --adv_size=ADV_SIZE           size of adversarial hidden layer [default: 300]
  --vec_drop=DROPOUT            dropout on the representation vector [default: 0]
  --init=INIT                   init value [default: 0]

"""

import os

import dynet as dy
import numpy as np
from docopt import docopt
from sklearn.metrics import accuracy_score
from tensorboard_logger import configure, log_value
from os.path import expanduser

from AdvNN import AdvNN
from consts import SEED, data_dir, models_dir, tensorboard_dir
from data_handler import get_data
from training_utils import get_logger, task_dic

np.random.seed(SEED)


def epoch_pass(data, model, trainer, training, ro, vec_drop, batch_size, logger, print_every=20000):
    """
    run a single epoch pass on the data
    :param data: data to train/predict on
    :param model: .
    :param trainer: optimizer
    :param training: boolean - for training/testing purposes
    :param ro: lambda used in paper. determines the adversarial power
    :param vec_drop: dropout on the representation vector
    :param batch_size: size of batch
    :param logger: .
    :param print_every: print the accumulative stats
    :return: accuracy score of the main task, the adversary task and the total loss.
    """
    task_preds, adv_preds = [], []
    task_truth, adv_truth = [], []
    losses = []
    t_loss = 0.0
    for i in range(num_adv):
        adv_preds.append([])

    for ind, data in enumerate(data):
        loss, task_pred, adv_pred = model.calc_loss(data[0], data[1], data[2], training,
                                                    ro, vec_drop)
        task_preds.append(task_pred)
        task_truth.append(data[1])

        for i in range(len(adv_pred)):
            adv_preds[i].append(adv_pred[i])

        adv_truth.append(data[2])
        losses.append(loss)

        if len(losses) == batch_size or ind == len(data) - 1:
            loss = dy.esum(losses)

            if training:
                loss.forward()
                loss.backward()
                trainer.update()

            t_loss += loss.npvalue()
            losses = []
            dy.renew_cg()

        if (ind + 1) % print_every == 0:
            mean = 0.0
            for i in range(num_adv):
                mean += accuracy_score(adv_truth, adv_preds[i])
            mean /= len(adv_pred)
            logger.debug('{0}: task loss: {1}, task acc: {2}, mean adv acc: {3}'
                         .format((ind + 1) / print_every, t_loss[0] / (ind + 1),
                                 accuracy_score(task_truth, task_preds), mean))
    adv_res = []
    for i in range(num_adv):
        adv_res.append(accuracy_score(adv_truth, adv_preds[i]))
    return accuracy_score(task_truth, task_preds), adv_res, t_loss[0] / len(data)


def train(model, train, dev, trainer, epochs, batch_size,
          vec_drop, logger, print_every=20000):
    """
    the training function with the adversarial usage
    """
    train_task_acc_arr, train_adv_acc_arr, train_loss_arr = [], [], []
    dev_task_acc_arr, dev_adv_acc_arr, dev_loss_arr = [], [], []
    best_model_epoch = 1
    best_score = 0.0

    ro = float(arguments['--ro'])
    logger.debug('training started')
    for epoch in xrange(1, epochs + 1):
        dy.renew_cg()

        # train
        epoch_pass(train, model, trainer, True, ro, vec_drop, batch_size,
                   logger, print_every)
        train_task_acc, train_adv_acc, loss = epoch_pass(train, model, trainer, False, ro,
                                                         vec_drop, batch_size, logger, print_every)

        train_task_acc_arr.append(train_task_acc)
        train_adv_acc_arr.append(train_adv_acc)
        train_loss_arr.append(loss)
        logger.debug('train, {0}, {1}, {2}'.format(epoch, train_task_acc, train_adv_acc))

        # dev
        dev_task_acc, dev_adv_acc, loss = epoch_pass(dev, model, trainer, False, ro,
                                                     0, batch_size, logger, print_every)
        dev_task_acc_arr.append(dev_task_acc)
        dev_adv_acc_arr.append(dev_adv_acc)
        dev_loss_arr.append(loss)
        logger.debug('dev, {0}, {1}, {2}'.format(epoch, dev_task_acc, np.mean(dev_adv_acc)))
        log_value('dev-task-acc', dev_task_acc, epoch)
        log_value('dev-mean-adv-acc', np.mean(dev_adv_acc), epoch)

        if dev_task_acc > best_score:
            best_score = dev_task_acc
            model.save(models_dir + task + '/best_model')
            best_model_epoch = epoch
        if epoch % 10 == 0:
            model.save(models_dir + task + '/epoch_' + str(epoch))

    logger.info('best_score:' + str(best_score))
    logger.info('best_epoch:' + str(best_model_epoch))
    logger.info('train_task_acc:' + str(train_task_acc_arr))
    logger.info('train_adv_acc:' + str(train_adv_acc_arr))
    logger.info('train_loss:' + str(train_loss_arr))
    logger.info('dev_task_acc:' + str(dev_task_acc_arr))
    logger.info('dev_adv_acc:' + str(dev_adv_acc_arr))
    logger.info('dev_loss:' + str(dev_loss_arr))


def train_task(model, train, dev, trainer, epochs, batch_size, task_type, logger, print_every=20000):
    """
    the training function for a single task. used for the baseline experiments
    """
    # train_task_acc_arr, train_loss_arr = [], []
    dev_task_acc_arr, dev_loss_arr = [], []
    best_model_epoch = 1
    best_score = 0.0

    logger.debug('training started')
    for epoch in xrange(1, epochs + 1):
        dy.renew_cg()
        t_loss = 0.0
        adv_preds, adv_truth, losses = [], [], []
        # train
        for ind, data in enumerate(train):
            loss, adv_pred = model.adv_loss(data[0], data[task_type], True)
            adv_preds.append(adv_pred)
            adv_truth.append(data[task_type])

            losses.append(loss)

            if len(losses) == batch_size or ind == len(train) - 1:
                loss = dy.esum(losses)
                loss.forward()
                loss.backward()
                trainer.update()

                t_loss += loss.npvalue()
                losses = []
                dy.renew_cg()

            if (ind + 1) % print_every == 0:
                logger.debug('{0}: task loss: {1}, adv acc: {2}'.format((ind + 1) / print_every, t_loss / (ind + 1),
                                                                        accuracy_score(adv_truth, adv_preds)))
        # dev
        t_loss = 0.0
        adv_preds = []
        adv_truth = []
        for ind, data in enumerate(dev):
            dy.renew_cg()
            loss, adv_pred = model.adv_loss(data[0], data[task_type], False)
            adv_preds.append(adv_pred)
            adv_truth.append(data[task_type])

            t_loss += loss.npvalue()
        adv_acc = accuracy_score(adv_truth, adv_preds)
        dev_task_acc_arr.append(adv_acc)
        dev_loss_arr.append(t_loss[0])
        logger.debug('dev-task epoch: {0}, acc: {1}, loss: {2}'.format(epoch, adv_acc, t_loss[0] / len(dev)))
        log_value('dev-task-acc', adv_acc, epoch)
        if adv_acc > best_score:
            best_score = adv_acc
            best_model_epoch = epoch
            model.save(models_dir + task + '/best_model')
        if epoch % 10 == 0:
            model.save(models_dir + task + '/epoch_' + str(epoch))
    logger.info('best_score:' + str(best_score))
    logger.info('best_epoch:' + str(best_model_epoch))
    logger.info('dev_task_acc:' + str(dev_task_acc_arr))
    logger.info('dev_loss:' + str(dev_loss_arr))


if __name__ == '__main__':
    arguments = docopt(__doc__)

    ro = arguments['--ro']
    num_adv = int(arguments['--num_adv'])
    lstm_size = int(arguments['--lstm_size'])
    task_str = arguments['--task']
    num_epoch = int(arguments['--epochs'])

    input_dir = data_dir

    if task_str not in task_dic:
        print 'task not supported in task_dic'
        exit(-1)
    input_dir += task_dic[task_str] + '/'
    input_vocab = input_dir + 'vocab'

    task = task_str
    task_type = int(arguments['--type'])

    if ro != str(-1):
        print 'using adversarial'
        task += '-n_adv:' + str(num_adv)
    else:
        if task_type == 1:
            task += '-type:1'
        else:
            task += '-type:2'

    pre_w2id = None

    if lstm_size > 1:
        task += '-lstm_size:{0}'.format(lstm_size)

    hid_size = int(arguments['--enc_size'])
    if hid_size != 300:
        task += '-hid:' + str(hid_size)

    rnn_type = arguments['--rnn_type']
    if rnn_type != 'lstm':
        task += '-' + arguments['--rnn_type']

    if ro != -1.0 and ro != 1.0:
        task += '-ro:' + str(ro)

    dropout = float(arguments['--dropout'])
    if dropout != 0.2:
        task += '-dropout:' + str(dropout)
        print 'dropout: {0}'.format(dropout)

    rnn_dropout = float(arguments['--rnn_dropout'])
    if rnn_dropout != 0.0:
        task += 'rnn_dropout:' + str(rnn_dropout)
        print 'using rnn dropout: {0}'.format(rnn_dropout)

    adv_hid_size = int(arguments['--adv_size'])
    if adv_hid_size != 300:
        task += '-adv_hid_size:' + str(adv_hid_size)
        print 'adversary hidden size: {0}'.format(adv_hid_size)

    adv_depth = int(arguments['--adv_depth'])
    if adv_depth != 1:
        task += '-depth:' + str(str(adv_depth))
        print 'adversary depth: {0}'.format(adv_depth)

    vec_dropout = float(arguments['--vec_drop'])
    if vec_dropout != 0.0:
        task += '-rep_dropout:' + str(vec_dropout)
        print 'dropout on the sentence representation'

    init = arguments['--init']
    if int(init) != 0:
        task += '-' + init
    model_dir = models_dir + task
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        print 'already exist. exiting...'
        exit(-1)

    logger = get_logger(task, model_dir)

    logger.info(arguments)
    logger.info(task)
    home = expanduser("~")
    configure(tensorboard_dir + task)

    x_train, x_test = get_data(task_str, input_dir)

    np.random.shuffle(x_train)

    out_size = 2

    with open(input_vocab, 'r') as f:
        vocab = f.readlines()
        vocab = map(lambda s: s.strip(), vocab)
    vocab_size = len(vocab)
    adv_net = AdvNN(hid_size, hid_size, out_size, hid_size, adv_hid_size, out_size, num_adv, vocab_size,
                    dropout, lstm_size, adv_depth, rnn_dropout=rnn_dropout, rnn_type=rnn_type)

    trainer = dy.MomentumSGDTrainer(adv_net._model)
    batch = 32
    if ro == str(-1):
        logger.debug('1 task')
        train_task(adv_net, x_train, x_test, trainer, num_epoch, batch, task_type, logger)
    else:
        logger.debug('2 tasks')
        train(adv_net, x_train, x_test, trainer, num_epoch, batch, vec_dropout, logger)

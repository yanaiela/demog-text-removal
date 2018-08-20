import dynet as dy
import numpy as np


class AdvNN(object):
    def __init__(self, task_in_size, task_hid_size, task_out_size, adv_in_size, adv_hid_size, adv_out_size, adv_count,
                 vocab_size, dropout, lstm_size, adv_depth=1, rnn_dropout=0.0, rnn_type='lstm'):
        model = dy.Model()

        if rnn_type == 'lstm':
            self._rnn = dy.LSTMBuilder(lstm_size, 300, task_in_size, model)
        elif rnn_type == 'gru':
            self._rnn = dy.GRUBuilder(lstm_size, 300, task_in_size, model)
        else:
            self._rnn = dy.SimpleRNNBuilder(lstm_size, 300, task_in_size, model)

        params = {}

        params['w_lookup'] = model.add_lookup_parameters((vocab_size, 300))

        in_task = task_in_size
        params["task_w1"] = model.add_parameters((task_hid_size, in_task))
        params["task_b1"] = model.add_parameters((task_hid_size))
        params["task_w2"] = model.add_parameters((task_out_size, task_hid_size))
        params["task_b2"] = model.add_parameters((task_out_size))

        for i in range(adv_count):
            for j in range(adv_depth):
                params["adv_" + str(i) + "_w" + str(j + 1)] = model.add_parameters((adv_hid_size, adv_in_size))
                params["adv_" + str(i) + "_b" + str(j + 1)] = model.add_parameters((adv_hid_size))
            params["adv_" + str(i) + "_w" + str(adv_depth + 1)] = model.add_parameters((adv_out_size, adv_hid_size))
            params["adv_" + str(i) + "_b" + str(adv_depth + 1)] = model.add_parameters((adv_out_size))

        params["contra_adv_w1"] = model.add_parameters((adv_hid_size, adv_in_size))
        params["contra_adv_b1"] = model.add_parameters((adv_hid_size))
        params["contra_adv_w2"] = model.add_parameters((adv_out_size, adv_hid_size))
        params["contra_adv_b2"] = model.add_parameters((adv_out_size))

        self._model = model
        self._hid_dim = task_hid_size
        self._in_dim = task_in_size
        self._adv_count = adv_count
        self._adv_depth = adv_depth
        self._params = params
        self._dropout = dropout
        self._rnn_dropout = rnn_dropout

    def encode_sentence(self, sentence, update_w=True, train=False):
        """
        simple rnn encoder.
        each token gets embedded, and calculating the rnn over all of them.
        returning the final hidden state
        """
        w_lookup = self._params['w_lookup']
        words = [dy.lookup(w_lookup, w, update=update_w) for w in sentence]

        rnn = self._rnn

        if train:
            rnn.set_dropout(self._rnn_dropout)
        else:
            rnn.disable_dropout()
        rnn_init = rnn.initial_state()
        states = rnn_init.transduce(words)
        return states[-1]

    def task_mlp(self, vec_sen, train, y_s=None):
        """
        calculating the mlp function over the sentence representation vector
        """
        w1 = dy.parameter(self._params["task_w1"])
        b1 = dy.parameter(self._params["task_b1"])
        w2 = dy.parameter(self._params["task_w2"])
        b2 = dy.parameter(self._params["task_b2"])

        if train:
            drop = self._dropout
        else:
            drop = 0

        if y_s is not None:
            v = dy.vecInput(1)
            v.set([y_s])
            in_vec = dy.concatenate([vec_sen, v])
        else:
            in_vec = vec_sen

        out = dy.tanh(dy.dropout(dy.affine_transform([b1, w1, in_vec]), drop))
        out = dy.affine_transform([b2, w2, out])

        return out

    def adv_mlp(self, vec_sen, adv_ind, train, vec_drop):
        """
        calculating the adversarial mlp over the sentence representation vector.
        more than a single adversarial mlp is supported
        """

        if train:
            drop = self._dropout
            out = dy.dropout(vec_sen, vec_drop)
        else:
            drop = 0
            out = vec_sen

        for i in range(self._adv_depth):
            w = dy.parameter(self._params["adv_" + str(adv_ind) + "_w" + str(i + 1)])
            b = dy.parameter(self._params["adv_" + str(adv_ind) + "_b" + str(i + 1)])
            out = dy.tanh(dy.dropout(dy.affine_transform([b, w, out]), drop))

        w = dy.parameter(self._params["adv_" + str(adv_ind) + "_w" + str(self._adv_depth + 1)])
        b = dy.parameter(self._params["adv_" + str(adv_ind) + "_b" + str(self._adv_depth + 1)])
        out = dy.affine_transform([b, w, out])

        return out

    # based on: Unsupervised Domain Adaptation by Backpropagation, Yaroslav Ganin & Victor Lempitsky
    def calc_loss(self, sentence, y_task, y_adv, train, ro, vec_drop=0):
        """
        calculating the loss over a single example.
        accumulating the main task and adversarial task loss together.
        """
        sen = self.encode_sentence(sentence, update_w=True, train=train)

        task_res = self.task_mlp(sen, train)
        task_probs = dy.softmax(task_res)
        task_loss = dy.pickneglogsoftmax(task_res, y_task)

        adversarial_res = []
        if ro > 0:
            adversarial_losses = []

            for i in range(self._adv_count):
                adv_res = self.adv_mlp(dy.flip_gradient(sen, ro), i, train, vec_drop)
                probs = dy.softmax(adv_res)
                adversarial_res.append(np.argmax(probs.npvalue()))
                adversarial_losses.append(dy.pickneglogsoftmax(adv_res, y_adv))

            total_loss = task_loss + dy.esum(adversarial_losses)
        else:
            total_loss = task_loss

        return total_loss, np.argmax(task_probs.npvalue()), adversarial_res

    def adv_loss(self, sentence, y_adv, train):
        """
        calculating the loss for just a single class. mainly for the baseline models.
        :param sentence:
        :param y_adv:
        :param train:
        :return:
        """
        sen = self.encode_sentence(sentence, update_w=True, train=train)
        adv_res = self.adv_mlp(sen, 0, train, 0)
        adv_probs = dy.softmax(adv_res)
        adv_loss = dy.pickneglogsoftmax(adv_res, y_adv)
        return adv_loss, np.argmax((adv_probs.npvalue()))

    def save(self, f_name):
        self._model.save(f_name)

    def load(self, f_name):
        self._model.populate(f_name)

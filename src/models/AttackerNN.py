import dynet as dy
import numpy as np


class AttackerNN(object):
    def __init__(self, mlp_layers, dropout):
        model = dy.Model()

        params = {}
        for i in range(len(mlp_layers)):
            params["adv_w" + str(i)] = model.add_parameters((mlp_layers[i][1], mlp_layers[i][0]))
            params["adv_b" + str(i)] = model.add_parameters((mlp_layers[i][1]))

        self._mlp_layers = len(mlp_layers)
        self._model = model
        self._params = params
        self._dropout = dropout

    def calc_loss(self, enc_sen, y_adv, vec_drop, train):
        """
        the attacker core functionality.
        mlp function, with (possibely) multi layers, and at least one.
        :param enc_sen:
        :param y_adv:
        :param vec_drop:
        :param train:
        :return:
        """

        w = dy.parameter(self._params["adv_w0"])
        b = dy.parameter(self._params["adv_b0"])

        if train:
            drop = self._dropout
            out = dy.dropout(enc_sen, vec_drop)
        else:
            drop = 0
            out = enc_sen

        out = dy.tanh(dy.dropout(dy.affine_transform([b, w, out]), drop))
        if self._mlp_layers > 2:
            for i in range(self._mlp_layers - 2):
                w = dy.parameter(self._params["adv_w" + str(i + 1)])
                b = dy.parameter(self._params["adv_b" + str(i + 1)])
                out = dy.tanh(dy.dropout(dy.affine_transform([b, w, out]), drop))
        w = dy.parameter(self._params["adv_w" + str(self._mlp_layers - 1)])
        b = dy.parameter(self._params["adv_b" + str(self._mlp_layers - 1)])
        out = dy.affine_transform([b, w, out])

        task_probs = dy.softmax(out)
        adv_loss = dy.pickneglogsoftmax(out, y_adv)
        return adv_loss, np.argmax(task_probs.npvalue())

    def save(self, f_name):
        self._model.save(f_name)

    def load(self, f_name):
        self._model.populate(f_name)

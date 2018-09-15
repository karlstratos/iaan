# Author: Karl Stratos (me@karlstratos.com)
import dynet as dy
import numpy as np
import os
import pickle
import random
import sys
import time
from common_architecture import CommonArchitecture
from information_theory import InformationTheory
from model import Model


class WBagPredictor(Model):

    def __init__(self, cmd=""):
        self.common = CommonArchitecture(self)
        self.info = InformationTheory()
        super(WBagPredictor, self).__init__(cmd)

    def config(self, arch, wdim, hdim, verbose=False):
        self.arch = arch
        self.wdim = wdim
        self.hdim = hdim
        self._verbose = verbose

    def init_parameters(self, wembs={}):
        self.common.init_common_parameters(wembs)  # Only word embs

        if self.arch == "wbag":
            self.U = self.m.add_parameters((len(self._w2i), self.wdim))

        elif self.arch == "hlstm":
            self.flstm1 = dy.LSTMBuilder(1, self.wdim, self.hdim/2, self.m)
            self.blstm1 = dy.LSTMBuilder(1, self.wdim, self.hdim/2, self.m)
            self.flstm2 = dy.LSTMBuilder(1, self.hdim, self.hdim/2, self.m)
            self.blstm2 = dy.LSTMBuilder(1, self.hdim, self.hdim/2, self.m)
            self.U = self.m.add_parameters((len(self._w2i), self.hdim))

        else:
            raise ValueError("unknown architecture: {0}".format(self.arch))

    def get_article_rep(self, input_seqs):
        if self.arch == "wbag":
            vec = dy.average([wemb for input_seq in input_seqs
                              for wemb in input_seq])

        elif self.arch == "hlstm":
            vec = self.common.get_hier_bilstm_avg(input_seqs,
                                                  self.flstm1, self.blstm1,
                                                  self.flstm2, self.blstm2)

        else:
            raise ValueError("unknown architecture: {0}".format(self.arch))
        return self.U * vec

    def get_loss(self, article_X, article_Y):
        dy.renew_cg()
        input_seqs_X = [[self.common.get_wemb(w) for w in sent]
                        for sent in article_X]
        rep = self.get_article_rep(input_seqs_X)
        on_probs = dy.logistic(rep)

        answers_numpy = np.zeros(len(self._w2i))
        present = {self._w2i[w]: True for sent in article_Y for w in sent}
        for i in present: answers_numpy[i] = 1.0
        answers = dy.inputTensor(answers_numpy)
        loss = dy.binary_log_loss(on_probs, answers)
        return loss

    def test(self, articles_X, articles_Y):
        avg_xent = 0.0

        for (article_X, article_Y) in zip(articles_X, articles_Y):
            dy.renew_cg()
            input_seqs_X = [[self.common.get_wemb(w if w in self._w2i else
                                                  self._UNK)
                             for w in sent] for sent in article_X]
            rep = self.get_article_rep(input_seqs_X)
            on_lprobs = self.info.log2(dy.logistic(rep)).value()
            present = {self._w2i[w if w in self._w2i else self._UNK]: True
                       for sent in article_Y for w in sent}
            xent = sum([- on_lprobs[i] for i in present])
            avg_xent += xent / len(articles_X)

        return avg_xent

    def train(self, model_path, articles_X, articles_Y, dev_articles_X,
              dev_articles_Y, lrate, epochs):
        self.m = dy.ParameterCollection()
        self._prepare_model_directory(model_path)
        wseqs = [wseq for wseq_list in articles_X + articles_Y
                 for wseq in wseq_list]
        self.build_dicts(wseqs, [[]])
        self.init_parameters()
        self._train_report(articles_X, articles_Y, lrate, epochs, 1)
        self.common.turn_on_training(0)  # No dropout

        best_hY_X = float("inf")
        trainer = dy.AdamTrainer(self.m, lrate)
        for epoch in xrange(epochs):
            self._log("Epoch {0:2d}  ".format(epoch + 1), False)
            epoch_start_time = time.time()
            inds = [i for i in xrange(len(articles_Y))]
            random.shuffle(inds)
            avg_loss = 0.0
            for data_num, i in enumerate(inds):
                if (data_num + 1) % 100 == 0:
                    print data_num + 1,
                    sys.stdout.flush()
                loss = self.get_loss(articles_X[i], articles_Y[i])
                avg_loss += loss.value() / len(inds)
                loss.backward()
                trainer.update()

            self._log("updates: {0}  ".format(len(inds)), False)
            self._log("avg_loss: {0:.2f}  ".format(avg_loss), False)
            self._log("({0:.1f}s)  ".format(time.time() -
                                            epoch_start_time), False)

            self.common.turn_off_training()
            hY_X = self.test(dev_articles_X, dev_articles_Y)
            self.common.turn_on_training(0)  # No dropout
            self._log("dev {0:.2f}  ".format(hY_X), False)
            if hY_X < best_hY_X:
                best_hY_X = hY_X
                self._log("new best - saving  ", False)
                self.save(model_path)
            self._log("")

    def save(self, model_path):
        self.m.save(os.path.join(model_path, "model"))
        with open(os.path.join(model_path, "info.pickle"), 'w') as outf:
            pickle.dump(
                (self._w2i,
                 self.arch,
                 self.wdim,
                 self.hdim),
                outf)

    def load_and_populate(self, model_path):
        self.m = dy.ParameterCollection()
        with open(os.path.join(model_path, "info.pickle")) as inf:
            (self._w2i,
             self.arch,
             self.wdim,
             self.hdim) = pickle.load(inf)
        self.init_parameters(wembs={})
        self.m.populate(os.path.join(model_path, "model"))

    def _train_report(self, articles_X, articles_Y, lrate, epochs, batch_size):
        num_words_X = sum([sum([len(sent) for sent in article])
                           for article in articles_X])
        num_words_Y = sum([sum([len(sent) for sent in article])
                           for article in articles_Y])

        self._log("___Data___")
        self._log("  # article pairs:    {0}".format(len(articles_X)))
        self._log("  # distinct words:   {0}".format(self.wsize()))
        self._log("  # words in X:       {0}".format(num_words_X))
        self._log("  # words in Y:       {0}".format(num_words_Y))
        self._log("  # pretrained wembs: {0}".format(self.num_pretrained_wembs))
        self._log("")
        self._log("___Model___")
        self._log("  arch:               {0}".format(self.arch))
        self._log("  wdim:               {0}".format(self.wdim))
        self._log("  hdim:               {0}".format(self.hdim))
        self._log("")
        self._log("___Training___")
        self._log("  lrate:              {0}".format(lrate))
        self._log("  epochs:             {0}".format(epochs))
        self._log("  batch size:         {0}".format(batch_size))
        self._log("")

    def _enable_lstm_dropout(self, drate):
        if self.arch == "hlstm":
            self.flstm1.set_dropout(drate)
            self.blstm1.set_dropout(drate)
            self.flstm2.set_dropout(drate)
            self.blstm2.set_dropout(drate)

    def _disable_lstm_dropout(self):
        if self.arch == "hlstm":
            self.flstm1.disable_dropout()
            self.blstm1.disable_dropout()
            self.flstm2.disable_dropout()
            self.blstm2.disable_dropout()

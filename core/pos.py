# Author: Karl Stratos (me@karlstratos.com)
import dynet as dy
import numpy as np
import os
import pickle
import random
import time
from common_architecture import CommonArchitecture
from model import Model
from window import Window


class PartOfSpeechInducer(Model):

    def __init__(self, cmd=""):
        self.common = CommonArchitecture(self)
        super(PartOfSpeechInducer, self).__init__(cmd)

    def config(self, arch, loss_type, zsize, wdim, cdim, jdim, width, swap,
               pseudocount, metric, verbose=False):
        self.arch = arch
        self.loss_type = loss_type
        self.zsize = zsize
        self.wdim = wdim
        self.cdim = cdim
        self.jdim = jdim
        self.width = width
        self.swap = swap
        self.pseudocount = pseudocount
        self.metric = metric
        self._verbose = verbose

    def init_parameters(self, wembs):
        crep_dim = self.common.init_common_parameters(wembs)

        if self.arch == "default":
            assert self.wdim > 0
            self.W_X = self.m.add_parameters((self.zsize,
                                              2 * self.wdim * self.width))
            self.W_Y = self.m.add_parameters((self.zsize, self.wdim + crep_dim))

        elif self.arch == "default-char":
            assert crep_dim > 0
            self.W_X = self.m.add_parameters((self.zsize,
                                              2 * crep_dim * self.width))
            self.W_Y = self.m.add_parameters((self.zsize, crep_dim))

        elif self.arch == "brown":
            assert self.wdim > 0
            self.W = self.m.add_parameters((self.zsize, self.wdim))

        elif self.arch == "brown-sep":
            assert self.wdim > 0
            self.W_X = self.m.add_parameters((self.zsize, self.wdim))
            self.W_Y = self.m.add_parameters((self.zsize, self.wdim))

        elif self.arch == "widen1":
            assert self.wdim > 0
            self.W_X = self.m.add_parameters((self.zsize,
                                              2 * self.wdim * self.width))
            self.W_Y = self.m.add_parameters((self.zsize,
                                              2 * self.wdim + crep_dim))

        elif self.arch == "widen2":
            assert self.wdim > 0
            self.W_X = self.m.add_parameters((self.zsize,
                                              2 * self.wdim * self.width))
            self.W_Y = self.m.add_parameters((self.zsize,
                                              3 * self.wdim + crep_dim))

        elif self.arch == "lstm":
            assert self.wdim > 0
            self.context_lstm1 = dy.LSTMBuilder(1, self.wdim, self.wdim, self.m)
            self.context_lstm2 = dy.LSTMBuilder(1, self.wdim, self.wdim, self.m)
            self.W_X = self.m.add_parameters((self.zsize, 2 * self.wdim))
            self.W_Y = self.m.add_parameters((self.zsize, self.wdim + crep_dim))

        else:
            raise ValueError("unknown architecture: {0}".format(self.arch))

    def compute_qp_pairs(self, wseqs, batch):
        dy.renew_cg()

        def assert_sentence_batch(wseqs, batch):
            (i, j) = batch[0]
            assert j == 0
            assert len(batch) == len(wseqs[i])
            for (i_next, j_next) in batch[1:]:
                assert i_next == i
                assert j_next == j + 1
                j = j_next
            return i

        if self.arch == "lstm":
            i = assert_sentence_batch(wseqs, batch)
            return self.compute_qp_pairs_lstm(wseqs[i])

        if self.arch == "default":
            compute_qp_pair = self.compute_qp_pair_default
        elif self.arch == "default-char":
            compute_qp_pair = self.compute_qp_pair_default_char
        elif self.arch == "brown":
            compute_qp_pair = self.compute_qp_pair_brown
        elif self.arch == "brown-sep":
            compute_qp_pair = self.compute_qp_pair_brown_sep
        elif self.arch == "widen1":
            compute_qp_pair = self.compute_qp_pair_widen1
        elif self.arch == "widen2":
            compute_qp_pair = self.compute_qp_pair_widen2
        else:
            raise ValueError("unknown self.architecture: {0}".format(self.arch))

        qp_pairs = []
        for (i, j) in batch:
            (qZ_X, pZ_Y) = compute_qp_pair(wseqs[i], j)
            qp_pairs.append((pZ_Y, qZ_X) if self.swap else (qZ_X, pZ_Y))
        return qp_pairs

    def compute_qp_pair_default(self, wseq, j):
        window = Window(wseq, self._BUF)
        X = window.left(j, self.width) + window.right(j, self.width)
        v_X = [self.common.get_wemb(w) for w in X]
        qZ_X = dy.softmax(self.W_X.expr() * dy.concatenate(v_X))

        v_Y = [v for v in [self.common.get_wemb(wseq[j]),
                           self.common.get_crep(wseq[j])] if v]
        pZ_Y = dy.softmax(self.W_Y.expr() * dy.concatenate(v_Y))

        return qZ_X, pZ_Y

    def compute_qp_pair_default_char(self, wseq, j):
        window = Window(wseq, self._BUF)
        X = window.left(j, self.width) + window.right(j, self.width)
        v_X = [self.common.get_crep(w) for w in X]
        qZ_X = dy.softmax(self.W_X.expr() * dy.concatenate(v_X))

        v_Y = [self.common.get_crep(wseq[j])]
        pZ_Y = dy.softmax(self.W_Y.expr() * dy.concatenate(v_Y))

        return qZ_X, pZ_Y

    def compute_qp_pair_brown(self, wseq, j):
        window = Window(wseq, self._BUF)
        X = window.left(j, 1)
        v_X = [self.common.get_wemb(w) for w in X]
        qZ_X = dy.softmax(self.W.expr() * dy.concatenate(v_X))

        Y = [wseq[j]]
        v_Y = [self.common.get_wemb(w) for w in Y]
        pZ_Y = dy.softmax(self.W.expr() * dy.concatenate(v_Y))

        return qZ_X, pZ_Y

    def compute_qp_pair_brown_sep(self, wseq, j):
        window = Window(wseq, self._BUF)
        X = window.left(j, 1)
        v_X = [self.common.get_wemb(w) for w in X]
        qZ_X = dy.softmax(self.W_X.expr() * dy.concatenate(v_X))

        Y = [wseq[j]]
        v_Y = [self.common.get_wemb(w) for w in Y]
        pZ_Y = dy.softmax(self.W_Y.expr() * dy.concatenate(v_Y))

        return qZ_X, pZ_Y

    def compute_qp_pair_widen1(self, wseq, j):
        window = Window(wseq, self._BUF)
        X = window.left(max(0, j - 1), self.width)
        X += window.right(j, self.width)
        v_X = [self.common.get_wemb(w) for w in X]
        qZ_X = dy.softmax(self.W_X.expr() * dy.concatenate(v_X))

        Y = window.left(j, 1)
        Y += [wseq[j]]
        v_Y = [v for v in [self.common.get_wemb(w) for w in Y] +
               [self.common.get_crep(wseq[j])] if v]
        pZ_Y = dy.softmax(self.W_Y.expr() * dy.concatenate(v_Y))

        return qZ_X, pZ_Y

    def compute_qp_pair_widen2(self, wseq, j):
        window = Window(wseq, self._BUF)
        X = window.left(max(0, j - 1), self.width)
        X += window.right(min(len(wseq) - 1, j + 1), self.width)
        v_X = [self.common.get_wemb(w) for w in X]
        qZ_X = dy.softmax(self.W_X.expr() * dy.concatenate(v_X))

        Y = window.left(j, 1)
        Y += [wseq[j]]
        Y += window.right(j, 1)
        v_Y = [v for v in [self.common.get_wemb(w) for w in Y] +
               [self.common.get_crep(wseq[j])] if v]
        pZ_Y = dy.softmax(self.W_Y.expr() * dy.concatenate(v_Y))

        return qZ_X, pZ_Y

    def compute_qp_pairs_lstm(self, wseq):
        #     <*> <-> the <-> dog <-> saw <-> the <-> cat <-> <*>
        inputs = [self.common.get_wemb(w)
                  for w in [self._BUF] + wseq + [self._BUF]]
        outputs1, outputs2 = self.common.run_bilstm(inputs,
                                                    self.context_lstm1,
                                                    self.context_lstm2)

        qp_pairs = []
        for j in xrange(len(wseq)):
            v_X = self.common.get_bilstm_left_right_from_outputs(outputs1,
                                                                 outputs2,
                                                                 j + 1, j + 1)
            qZ_X = dy.softmax(self.W_X.expr() * v_X)
            v_Y = [v for v in [self.common.get_wemb(wseq[j]),
                               self.common.get_crep(wseq[j])] if v]
            pZ_Y = dy.softmax(self.W_Y.expr() * dy.concatenate(v_Y))
            qp_pairs.append((qZ_X, pZ_Y))

        return qp_pairs

    def prepare_batches(self, wseqs, batch_size):
        if batch_size == 0:  # sentence-level batching
            inds = [i for i in xrange(len(wseqs))]
            random.shuffle(inds)
            batches = [[(i, j) for j in xrange(len(wseqs[i]))]
                       for i in inds]
        else:
            pairs = []
            for i in xrange(len(wseqs)):
                for j in xrange(len(wseqs[i])):
                    pairs.append((i, j))
            random.shuffle(pairs)
            batches = []
            for i in xrange(0, len(pairs), batch_size):
                batches.append(pairs[i:min(i + batch_size, len(pairs))])

        return batches

    def _enable_lstm_dropout(self, drate):
        if self.arch == "lstm":
            self.context_lstm1.set_dropout(drate)
            self.context_lstm2.set_dropout(drate)

    def _disable_lstm_dropout(self):
        if self.arch == "lstm":
            self.context_lstm1.disable_dropout()
            self.context_lstm2.disable_dropout()

    def measure_mi(self, wseqs):
        assert not self._is_training

        num_samples = 0
        joint = np.zeros((self.zsize, self.zsize))
        batches = self.prepare_batches(wseqs, 0)
        for batch in batches:
            qp_pairs = self.compute_qp_pairs(wseqs, batch)
            for (qZ_X, pZ_Y) in qp_pairs:
                num_samples += 1
                outer = qZ_X * dy.transpose(pZ_Y)
                joint += outer.value()
        joint /= num_samples
        mi = self.evaluator.mi_zero(joint)
        return mi

    def train(self, model_path, wseqs, lrate, drate, epochs, batch_size,
              wemb_path, tseqs=[]):
        wembs = self.read_wembs(wemb_path, self.wdim)
        self.m = dy.ParameterCollection()
        self._prepare_model_directory(model_path)
        self.build_dicts(wseqs, tseqs, wembs)
        self.init_parameters(wembs)
        trainer = dy.AdamTrainer(self.m, lrate)
        self._train_report(lrate, drate, epochs, batch_size)
        self.common.turn_on_training(drate)

        perf_best = 0.0
        for epoch in xrange(epochs):
            self._log("Epoch {0:2d}  ".format(epoch + 1), False)
            epoch_start_time = time.time()

            batches = self.prepare_batches(wseqs, batch_size)
            total_batch_loss = 0.0
            for batch in batches:
                qp_pairs = self.compute_qp_pairs(wseqs, batch)
                batch_loss = self.common.get_loss(qp_pairs)
                total_batch_loss +=  batch_loss.value()
                batch_loss.backward()
                trainer.update()

            avg_loss = total_batch_loss / len(batches)
            self._log("updates: {0}  ".format(len(batches)), False)
            self._log("loss: {0:.2f}  ".format(avg_loss), False)
            self._log("({0:.1f}s)  ".format(time.time() -
                                            epoch_start_time), False)
            if tseqs:
                self.common.turn_off_training()
                zseqs_X, zseqs_Y, zseqs_XY, infer_time = self.tag_all(wseqs)
                perf_max = self.common.evaluator_report(wseqs, tseqs, zseqs_X,
                                                        zseqs_Y, zseqs_XY,
                                                        infer_time,
                                                        self.measure_mi(wseqs),
                                                        self.metric)
                self.common.turn_on_training(drate)
                if perf_max > perf_best:
                    perf_best = perf_max
                    self._log("new best {0:.2f} - saving  ".format(perf_best),
                              False)
                    self.save(model_path)
            else:
                self.save(model_path)
            self._log("")

    def save(self, model_path):
        self.m.save(os.path.join(model_path, "model"))
        with open(os.path.join(model_path, "info.pickle"), 'w') as outf:
            pickle.dump(
                (self._w2i,
                 self._c2i,
                 self._jamo2i,
                 self.arch,
                 self.loss_type,
                 self.zsize,
                 self.wdim,
                 self.cdim,
                 self.jdim,
                 self.width,
                 self.swap,
                 self.pseudocount),
                outf)

    def load_and_populate(self, model_path):
        self.m = dy.ParameterCollection()
        with open(os.path.join(model_path, "info.pickle")) as inf:
            (self._w2i,
             self._c2i,
             self._jamo2i,
             self.arch,
             self.loss_type,
             self.zsize,
             self.wdim,
             self.cdim,
             self.jdim,
             self.width,
             self.swap,
             self.pseudocount) = pickle.load(inf)
        self.init_parameters(wembs={})
        self.m.populate(os.path.join(model_path, "model"))

    def _train_report(self, lrate, drate, epochs, batch_size):
        self._log("___Data___")
        self._log("  # sents:            {0}".format(self.num_seqs))
        self._log("  # words:            {0}".format(self.num_words))
        self._log("  # pretrained wembs: {0}".format(self.num_pretrained_wembs))
        self._log("  # distinct words:   {0}".format(self.wsize()))
        self._log("  # distinct chars:   {0}".format(self.csize()))
        self._log("  # distinct jamos:   {0}".format(self.jamo_size()))
        self._log("")
        self._log("___Model___")
        self._log("  arch:               {0}".format(self.arch))
        self._log("  loss type:          {0}".format(self.loss_type))
        self._log("  zsize:              {0}".format(self.zsize))
        self._log("  wdim:               {0}".format(self.wdim))
        self._log("  cdim:               {0}".format(self.cdim))
        self._log("  jdim:               {0}".format(self.jdim))
        self._log("  width:              {0}".format(self.width))
        self._log("  swap:               {0}".format(self.swap))
        self._log("  pseudocount:        {0}".format(self.pseudocount))
        self._log("")
        self._log("___Training___")
        self._log("  lrate:              {0}".format(lrate))
        self._log("  drate:              {0}".format(drate))
        self._log("  epochs:             {0}".format(epochs))
        self._log("  batch size:         {0}".format(batch_size))
        self._log("  metric:             {0}".format(self.metric))
        self._log("")

    def tag(self, wseq):
        """
             LB: I(Z; X)              MI: I(Z_Y; Z_X)
                              Z                         Z_X   Z_Y
                            q/|p                        q|     |p
                            / |                          |     |
                           X  Y                          X     Y
        """
        assert not self._is_training
        qp_pairs = self.compute_qp_pairs([wseq],
                                         [(0, j) for j in xrange(len(wseq))])
        zseq_X = []
        zseq_Y = []
        zseq_XY = []

        for (qZ_X, pZ_Y) in qp_pairs:
            zseq_X.append(np.argmax(qZ_X.npvalue()))
            zseq_Y.append(np.argmax(pZ_Y.npvalue()))
            if self.loss_type == "lb":
                zseq_XY.append(np.argmax(dy.cmult(qZ_X, pZ_Y).npvalue()))

        return zseq_X, zseq_Y, zseq_XY

    def tag_all(self, wseqs):
        infer_start_time = time.time()
        zseqs_X = []
        zseqs_Y = []
        zseqs_XY = []
        for wseq in wseqs:
            zseq_X, zseq_Y, zseq_XY = self.tag(wseq)
            zseqs_X.append(zseq_X)
            zseqs_Y.append(zseq_Y)
            if zseq_XY: zseqs_XY.append(zseq_XY)
        return zseqs_X, zseqs_Y, zseqs_XY, time.time() - infer_start_time

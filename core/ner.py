# Author: Karl Stratos (me@karlstratos.com)
import codecs
import dynet as dy
import numpy as np
import os
import pickle
import random
import time
from common_architecture import CommonArchitecture
from model import Model
from window import Window


class NamedEntityInducer(Model):

    def __init__(self, cmd=""):
        self.common = CommonArchitecture(self)
        self._dev = None
        super(NamedEntityInducer, self).__init__(cmd)

    def config(self, arch, loss_type, zsize, wdim, cdim, width, swap,
               pseudocount, verbose=False):
        self.arch = arch
        self.loss_type = loss_type
        self.zsize = zsize
        self.wdim = wdim
        self.cdim = cdim
        self.width = width
        self.swap = swap
        self.pseudocount = pseudocount
        self._verbose = verbose

    def init_parameters(self, wembs):
        crep_dim = self.common.init_common_parameters(wembs)

        assert self.wdim > 0  # All architectures use word embeddings.

        # BiLSTM for computing span representations
        spanrep_dim = 2 * self.wdim
        self.wlstm1 = dy.LSTMBuilder(1, self.wdim + crep_dim, self.wdim, self.m)
        self.wlstm2 = dy.LSTMBuilder(1, self.wdim + crep_dim, self.wdim, self.m)

        if self.arch == "default":
            self.W_X = self.m.add_parameters((self.zsize,
                                              2 * self.wdim * self.width))
            self.W_Y = self.m.add_parameters((self.zsize, spanrep_dim))

        elif self.arch == "lstm":
            self.context_lstm1 = dy.LSTMBuilder(1, self.wdim, self.wdim, self.m)
            self.context_lstm2 = dy.LSTMBuilder(1, self.wdim, self.wdim, self.m)
            self.W_X = self.m.add_parameters((self.zsize, 2 * self.wdim))
            self.W_Y = self.m.add_parameters((self.zsize, spanrep_dim))

        else:
            raise ValueError("unknown architecture: {0}".format(self.arch))

    def get_spanrep(self, wseq, span):
        inputs = [dy.concatenate([self.common.get_wemb(w),
                                  self.common.get_crep(w)])
                  for w in wseq[span[0]:span[1] + 1]]
        return self.common.get_bilstm_single(inputs, self.wlstm1, self.wlstm2)

    def compute_qp_pairs(self, wseqs, spanseqs, batch):
        dy.renew_cg()

        if self.arch == "default":
            compute_qp_pair = self.compute_qp_pair_default
        elif self.arch == "lstm":
            compute_qp_pair = self.compute_qp_pair_lstm
        else:
            raise ValueError("unknown architecture: {0}".format(self.arch))

        qp_pairs = []
        for (i, j) in batch:
            (qZ_X, pZ_Y) = compute_qp_pair(wseqs[i], spanseqs[i], j)
            qp_pairs.append((pZ_Y, qZ_X) if self.swap else (qZ_X, pZ_Y))
        return qp_pairs

    def compute_qp_pair_default(self, wseq, spanseq, j):
        (s, t) = spanseq[j]
        window = Window(wseq, self._BUF)
        X = window.left(s, self.width) + window.right(t, self.width)
        v_X = [self.common.get_wemb(w) for w in X]
        qZ_X = dy.softmax(self.W_X.expr() * dy.concatenate(v_X))

        v_Y = self.get_spanrep(wseq, (s, t))
        pZ_Y = dy.softmax(self.W_Y.expr() * v_Y)

        return qZ_X, pZ_Y

    def compute_qp_pair_lstm(self, wseq, spanseq, j):
        #     <*> <-> the <-> dog <-> saw <-> the <-> cat <-> <*>
        inputs = [self.common.get_wemb(w)
                  for w in [self._BUF] + wseq + [self._BUF]]
        outputs1, outputs2 = self.common.run_bilstm(inputs,
                                                    self.context_lstm1,
                                                    self.context_lstm2)
        (s, t) = spanseq[j]
        v_X = self.common.get_bilstm_left_right_from_outputs(outputs1,
                                                             outputs2,
                                                             s + 1, t + 1)
        qZ_X = dy.softmax(self.W_X.expr() * v_X)
        v_Y = self.get_spanrep(wseq, (s, t))
        pZ_Y = dy.softmax(self.W_Y.expr() * v_Y)

        return qZ_X, pZ_Y

    def prepare_batches(self, wseqs, spanseqs, batch_size):
        if batch_size == 0:  # sentence-level batching
            inds = [i for i in xrange(len(wseqs))]
            random.shuffle(inds)
            batches = [[(i, j) for j in xrange(len(spanseqs[i]))]
                       for i in inds]
        else:
            pairs = []
            for i in xrange(len(wseqs)):
                for j in xrange(len(spanseqs[i])):
                    pairs.append((i, j))
            random.shuffle(pairs)
            batches = []
            for i in xrange(0, len(pairs), batch_size):
                batches.append(pairs[i:min(i + batch_size, len(pairs))])

        return batches

    def _enable_lstm_dropout(self, drate):
        self.wlstm1.set_dropout(drate)
        self.wlstm2.set_dropout(drate)
        if self.arch == "lstm":
            self.context_lstm1.set_dropout(drate)
            self.context_lstm2.set_dropout(drate)

    def _disable_lstm_dropout(self):
        self.wlstm1.disable_dropout()
        self.wlstm2.disable_dropout()
        if self.arch == "lstm":
            self.context_lstm1.disable_dropout()
            self.context_lstm2.disable_dropout()

    def measure_mi(self, wseqs, spanseqs):
        assert not self._is_training
        num_samples = 0
        joint = np.zeros((self.zsize, self.zsize))
        batches = self.prepare_batches(wseqs, spanseqs, 0)
        for batch in batches:
            qp_pairs = self.compute_qp_pairs(wseqs, spanseqs, batch)
            for (qZ_X, pZ_Y) in qp_pairs:
                num_samples += 1
                outer = qZ_X * dy.transpose(pZ_Y)
                joint += outer.value()
        joint /= num_samples
        mi = self.evaluator.mi_zero(joint)
        return mi

    def train(self, model_path, wseqs, spanseqs, lrate, drate, epochs,
              batch_size, wemb_path):
        wembs = self.read_wembs(wemb_path, self.wdim)
        self.m = dy.ParameterCollection()
        self._prepare_model_directory(model_path)
        self.build_dicts(wseqs, [], wembs)  # No tseqs
        self.init_parameters(wembs)
        trainer = dy.AdamTrainer(self.m, lrate)
        num_spans = sum([len(spanseq) for spanseq in spanseqs])
        self._train_report(num_spans, lrate, drate, epochs, batch_size)
        self.common.turn_on_training(drate)

        (wseqs_dev, spanseqs_dev,
         entityseqs_dev) = self.read_ner_data(self._dev, read_entities=True)

        acc_best = 0.0
        for epoch in xrange(epochs):
            self._log("Epoch {0:2d}  ".format(epoch + 1), False)
            epoch_start_time = time.time()

            batches = self.prepare_batches(wseqs, spanseqs, batch_size)
            total_batch_loss = 0.0
            for batch in batches:
                qp_pairs = self.compute_qp_pairs(wseqs, spanseqs, batch)
                batch_loss = self.common.get_loss(qp_pairs)
                total_batch_loss +=  batch_loss.value()
                batch_loss.backward()
                trainer.update()

            avg_loss = total_batch_loss / len(batches)
            self._log("updates: {0}  ".format(len(batches)), False)
            self._log("loss: {0:.2f}  ".format(avg_loss), False)
            self._log("({0:.1f}s)  ".format(time.time() -
                                            epoch_start_time), False)
            if entityseqs_dev:
                self.common.turn_off_training()
                zseqs_X, zseqs_Y, zseqs_XY, infer_time = self.tag_all(
                    wseqs_dev, spanseqs_dev)
                acc_max = self.common.evaluator_report(
                    wseqs_dev, entityseqs_dev, zseqs_X, zseqs_Y, zseqs_XY,
                    infer_time, self.measure_mi(wseqs, spanseqs))
                self.common.turn_on_training(drate)
                if acc_max > acc_best:
                    acc_best = acc_max
                    self._log("new best {0:.2f} - saving  ".format(acc_best),
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
                 self.arch,
                 self.loss_type,
                 self.zsize,
                 self.wdim,
                 self.cdim,
                 self.width,
                 self.swap,
                 self.pseudocount),
                outf)

    def load_and_populate(self, model_path):
        self.m = dy.ParameterCollection()
        with open(os.path.join(model_path, "info.pickle")) as inf:
            (self._w2i,
             self._c2i,
             self.arch,
             self.loss_type,
             self.zsize,
             self.wdim,
             self.cdim,
             self.width,
             self.swap,
             self.pseudocount) = pickle.load(inf)
        self.init_parameters(wembs={})
        self.m.populate(os.path.join(model_path, "model"))

    def _train_report(self, num_spans, lrate, drate, epochs, batch_size):
        self._log("___Data___")
        self._log("  # sents:            {0}".format(self.num_seqs))
        self._log("  # words:            {0}".format(self.num_words))
        self._log("  # spans:            {0}".format(num_spans))
        self._log("  # pretrained wembs: {0}".format(self.num_pretrained_wembs))
        self._log("  # distinct words:   {0}".format(self.wsize()))
        self._log("  # distinct chars:   {0}".format(self.csize()))
        if self._dev:
            self._log("  using as dev data:  {0}".format(self._dev))
        self._log("")
        self._log("___Model___")
        self._log("  arch:               {0}".format(self.arch))
        self._log("  loss type:          {0}".format(self.loss_type))
        self._log("  zsize:              {0}".format(self.zsize))
        self._log("  wdim:               {0}".format(self.wdim))
        self._log("  cdim:               {0}".format(self.cdim))
        self._log("  width:              {0}".format(self.width))
        self._log("  swap:               {0}".format(self.swap))
        self._log("  pseudocount:        {0}".format(self.pseudocount))
        self._log("")
        self._log("___Training___")
        self._log("  lrate:              {0}".format(lrate))
        self._log("  drate:              {0}".format(drate))
        self._log("  epochs:             {0}".format(epochs))
        self._log("  batch size:         {0}".format(batch_size))
        self._log("")

    def tag(self, wseq, spanseq):
        """
             LB: I(Z; X)              MI: I(Z_Y; Z_X)
                              Z                         Z_X   Z_Y
                            q/|p                        q|     |p
                            / |                          |     |
                           X  Y                          X     Y
        """
        assert not self._is_training
        qp_pairs = self.compute_qp_pairs([wseq], [spanseq],
                                         [(0, j) for j in xrange(len(spanseq))])
        zseq_X = []
        zseq_Y = []
        zseq_XY = []

        for (qZ_X, pZ_Y) in qp_pairs:
            zseq_X.append(np.argmax(qZ_X.npvalue()))
            zseq_Y.append(np.argmax(pZ_Y.npvalue()))
            if self.loss_type == "lb":
                zseq_XY.append(np.argmax(dy.cmult(qZ_X, pZ_Y).npvalue()))

        return zseq_X, zseq_Y, zseq_XY

    def tag_all(self, wseqs, spanseqs):
        infer_start_time = time.time()
        zseqs_X = []
        zseqs_Y = []
        zseqs_XY = []
        for i in xrange(len(wseqs)):
            zseq_X, zseq_Y, zseq_XY = self.tag(wseqs[i], spanseqs[i])
            zseqs_X.append(zseq_X)
            zseqs_Y.append(zseq_Y)
            if zseq_XY: zseqs_XY.append(zseq_XY)
        return zseqs_X, zseqs_Y, zseqs_XY, time.time() - infer_start_time

# Author: Karl Stratos (me@karlstratos.com)
import dynet as dy
from information_theory import *
from jamo import h2j


class CommonArchitecture(object):
    """
    Initializes/manipulates common components across Model architectures.
    Assumes that all invoked attributes are already populated.
    """

    def __init__(self, model):
        self.info = InformationTheory()
        self.model = model

    def init_common_parameters(self, wembs={}):
        model = self.model
        zsize = model.zsize
        wdim = model.wdim
        cdim = model.cdim
        jdim = model.jdim

        # Add word lookup parameters.
        if wdim > 0:
            model.wlook = model.m.add_lookup_parameters((len(model._w2i), wdim))
            for w in wembs: model.wlook.init_row(model._w2i[w], wembs[w])

        # Add character-/jamo-level parameters.
        crep_dim = 2 * max(cdim, jdim)
        if crep_dim > 0:
            if cdim > 0:
                model.clook = model.m.add_lookup_parameters((len(model._c2i),
                                                             cdim))
            if jdim > 0:
                model.jlook = model.m.add_lookup_parameters((len(model._jamo2i),
                                                             jdim))
                model.jamo_W = model.m.add_parameters((jdim, 3 * jdim))
                model.jamo_b = model.m.add_parameters((jdim))

            model.clstm1 = dy.LSTMBuilder(1, cdim + jdim, crep_dim / 2, model.m)
            model.clstm2 = dy.LSTMBuilder(1, cdim + jdim, crep_dim / 2, model.m)

        return crep_dim

    def get_wemb(self, w, update_flag=True):
        model = self.model
        if model.wdim == 0: return None
        return dy.lookup(model.wlook, model.get_w_ind(w), update=update_flag)

    def get_cemb(self, c, update_flag=True):
        model = self.model
        if model.cdim == 0: return None
        return dy.lookup(model.clook, model.get_c_ind(c), update=update_flag)

    def get_jamo_rep(self, char, update_flag=True):
        model = self.model
        if model.jdim == 0: return None

        jamos = h2j(char)
        if len(jamos) == 1:  # Non-Hangul (ex: @, Q)
            return dy.lookup(model.jlook, model.get_jamo_ind(jamos[0]),
                             update=update_flag)

        jamos_concat = dy.concatenate([
            dy.lookup(model.jlook, model.get_jamo_ind(jamos[0])),
            dy.lookup(model.jlook, model.get_jamo_ind(jamos[1])),
            dy.lookup(model.jlook, model.get_jamo_ind(jamos[2])) if \
            len(jamos) > 2 else dy.lookup(model.jlook,
                                          model._jamo2i[model._EMP],
                                          update=update_flag)])
        return self.activate(model.jamo_W * jamos_concat +
                             model.jamo_b)

    def get_crep(self, chars, update_flag=True):
        model = self.model
        if model.cdim == 0 and model.jdim == 0: return None
        inputs = [dy.concatenate([v for v in [self.get_cemb(c, update_flag),
                                              self.get_jamo_rep(c, update_flag)]
                                  if v])
                  for c in chars]
        return self.get_bilstm_single(inputs, model.clstm1, model.clstm2,
                                      update_flag)

    def estimate_joint_priors(self, qp_pairs, smooth=False):
        model = self.model
        zsize = model.zsize
        pseudocount = model.pseudocount

        joint_samples = []
        qZ_samples = []
        pZ_samples = []
        for (qZ_X, pZ_Y) in qp_pairs:
            joint_samples.append(qZ_X * dy.transpose(pZ_Y))
            qZ_samples.append(qZ_X)
            pZ_samples.append(pZ_Y)

        if smooth:
            v_unif = dy.inputTensor(np.ones((zsize)) / zsize)
            A_unif = dy.inputTensor(np.ones((zsize, zsize)) / (zsize ** 2))
            for _ in xrange(pseudocount):
                joint_samples.append(A_unif)
                qZ_samples.append(v_unif)
                pZ_samples.append(v_unif)

        joint = dy.average(joint_samples)
        qZ = dy.average(qZ_samples)
        pZ = dy.average(pZ_samples)

        return joint, qZ, pZ

    def get_loss(self, qp_pairs):
        if self.model.loss_type == "lb":
            pZ = dy.average([pZ_Y for (_, pZ_Y) in qp_pairs])
            hZ = self.info.entropy(pZ)
            hZ_X = dy.average([self.info.cross_entropy(pZ_Y, qZ_X)
                               for (qZ_X, pZ_Y) in qp_pairs])
            loss = hZ_X - hZ

        elif self.model.loss_type == "mi":
            joint, qZ, pZ = self.estimate_joint_priors(qp_pairs, smooth=True)
            loss = - self.info.mi_with_priors(joint, qZ, pZ)

        elif self.model.loss_type == "mi-zero":
            joint, qZ, pZ = self.estimate_joint_priors(qp_pairs, smooth=False)
            loss = - self.info.mi_zero_with_priors(joint, qZ, pZ)

        else:
            raise ValueError("unknown loss type: {0}".format(loss_type))

        return loss

    def evaluator_report(self, wseqs, tseqs, zseqs_X, zseqs_Y, zseqs_XY,
                         infer_time, mi_value, metric, newline=False):
        model = self.model
        if metric == "m2o":
            perf_X = model.evaluator.compute_many2one_acc(tseqs, zseqs_X)
            perf_Y = model.evaluator.compute_many2one_acc(tseqs, zseqs_Y)
            perf_XY = model.evaluator.compute_many2one_acc(tseqs, zseqs_XY) \
                     if zseqs_XY else float("-inf")
        elif metric == "vm":
            perf_X = model.evaluator.compute_v_measure(tseqs, zseqs_X)
            perf_Y = model.evaluator.compute_v_measure(tseqs, zseqs_Y)
            perf_XY = model.evaluator.compute_v_measure(tseqs, zseqs_XY) \
                     if zseqs_XY else float("-inf")
        else:
            raise ValueError("unknown metric: {0}".format(metric))

        model._log("MI: {0:.2f}  ".format(mi_value), False)
        model._log("metric: {0}  ".format(metric), False)
        model._log("X perf: {0:.2f}  ".format(perf_X), False)
        model._log("Y perf: {0:.2f}  ".format(perf_Y), False)
        if perf_XY > -1: model._log("XY perf: {0:.2f}  ".format(perf_XY), False)
        model._log("({0:.1f}s)  ".format(infer_time), False)

        if newline: model._log("")
        return max([perf_X, perf_Y, perf_XY])

    def run_lstm(self, inputs, lstm, update_flag=True):
        s = lstm.initial_state(update=update_flag)
        return s.transduce(inputs)

    def run_bilstm(self, inputs, lstm1, lstm2, update_flag=True):
        outputs1 = self.run_lstm(inputs, lstm1, update_flag)  # f1 -> f2 -> f3
        outputs2 = self.run_lstm(inputs[::-1], lstm2,         # b3 -> b2 -> b1
                                 update_flag)
        return outputs1, outputs2[::-1]                # [f1 f2 f3], [b1 b2 b3]

    def get_bilstm_all(self, inputs, lstm1, lstm2, update_flag=True):
        outputs1, outputs2 = self.run_bilstm(inputs, lstm1, lstm2, update_flag)
        return [dy.concatenate([outputs1[i], outputs2[i]]) for
                i in xrange(len(outputs1))]

    def get_hier_bilstm_avg(self, input_seqs, flstm1, blstm1, flstm2, blstm2,
                            update_flag=True):
        seqreps = []
        for input_seq in input_seqs:
            seqreps.append(dy.average(self.get_bilstm_all(input_seq, flstm1,
                                                          blstm1, update_flag)))
        return dy.average(self.get_bilstm_all(seqreps, flstm2, blstm2,
                                              update_flag))

    def get_bilstm_single(self, inputs, lstm1, lstm2, update_flag=True):
        outputs1, outputs2 = self.run_bilstm(inputs, lstm1, lstm2, update_flag)
        return dy.concatenate([outputs1[-1], outputs2[0]])

    def get_bilstm_left_right(self, inputs, lstm1, lstm2, start, end,
                              update_flag=True):
        outputs1, outputs2 = self.run_bilstm(inputs, lstm1, lstm2, update_flag)
        return self.get_bilstm_left_right_from_outputs(outputs1, outputs2,
                                                       start, end)

    def get_bilstm_left_right_from_outputs(self, outputs1, outputs2, start,
                                           end):
        assert len(outputs1) == len(outputs2)
        assert start > 0 and end < len(outputs1) - 1 and start <= end
        return dy.concatenate([outputs1[start - 1], outputs2[end + 1]])

    def activate(self, h):
        return {"tanh": dy.tanh,
                "sigmoid": dy.logistic,
                "relu": dy.rectify}[self.model._activation](h)

    def turn_on_training(self, drate):
        self.model._is_training = True
        if drate > 0:
            self.model._input_drop = True
            if self.model.cdim > 0:
                self.model.clstm1.set_dropout(drate)
                self.model.clstm2.set_dropout(drate)
            self.model._enable_lstm_dropout(drate)  # Must be defined in model

    def turn_off_training(self):
        self.model._is_training = False
        self.model._input_drop = False
        if self.model.cdim > 0:
            self.model.clstm1.disable_dropout()
            self.model.clstm2.disable_dropout()
        self.model._disable_lstm_dropout()  # Must be defined in model

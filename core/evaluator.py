# Author: Karl Stratos (me@karlstratos.com)
import numpy as np
from collections import Counter


class Evaluator(object):

    def count_cooccurence(self, tseqs, zseqs):
        cooccur = {}
        assert len(tseqs) == len(zseqs)
        for i in xrange(len(tseqs)):
            assert len(tseqs[i]) == len(zseqs[i])
            for (t, z) in zip(tseqs[i], zseqs[i]):
                if not z in cooccur: cooccur[z] = Counter()
                cooccur[z][t] += 1
        return cooccur

    def get_majority_mapping(self, tseqs, zseqs):
        cooccur = self.count_cooccurence(tseqs, zseqs)
        mapping = {}
        for z in cooccur:
            mapping[z] = max(cooccur[z].items(), key=lambda x: x[1])[0]
        return mapping

    def compute_many2one_acc(self, tseqs, zseqs):
        mapping = self.get_majority_mapping(tseqs, zseqs)

        num_instances = 0
        num_correct = 0
        for i in xrange(len(tseqs)):
            for (t, z) in zip(tseqs[i], zseqs[i]):
                num_instances += 1
                if mapping[z] == t:
                    num_correct += 1
        acc = float(num_correct) / num_instances * 100

        return acc

    def mi_zero(self, joint):
        assert abs(1.0 - sum(sum(joint))) < 1e-6
        prior1 = np.sum(joint, axis=1)
        prior2 = np.sum(joint, axis=0)
        mi = 0.0
        for z1 in xrange(joint.shape[0]):
            for z2 in xrange(joint.shape[1]):
                if joint[z1, z2] > 0.0:
                    mi += joint[z1, z2] * np.log2(joint[z1, z2] /
                                                  prior1[z1] /
                                                  prior2[z2])
        return mi

    def get_index_mapping(self, tseqs):
        index_mapping = {}
        for tseq in tseqs:
            for t in tseq:
                if not t in index_mapping:
                    i = len(index_mapping)
                    index_mapping[t] = i
        return index_mapping

    def compute_mi_bigram(self, tseqs):
        t2i = self.get_index_mapping(tseqs)
        i_buffer = len(t2i)  # Special start/end buffer index

        joint = np.zeros((len(t2i) + 1, len(t2i) + 1))
        for tseq in tseqs:
            joint[i_buffer, t2i[tseq[0]]] += 1
            for i in xrange(1, len(tseq)):
                joint[t2i[tseq[i - 1]], t2i[tseq[i]]] += 1
            joint[t2i[tseq[-1]], i_buffer] += 1
        joint /= sum(sum(joint))

        mi = self.mi_zero(joint)
        return mi, joint

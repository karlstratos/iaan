# Author: Karl Stratos (me@karlstratos.com)
import dynet as dy
import numpy as np
import random


class InformationTheory(object):
    """
    Methods that handle zero probability call value(), so they can't be batched.
    """
    def cross_entropy_structbag(self, P, Q):
        """
        P (K x m) represents a distribution over STRUCTURED labels where each
        label is a BAG of K INDEPENDENT symbols taking values in {1 ... m}.
        That is, z = (z1 ... zK) is assigned probability P1(z1) * ... * PK(zK).
        (Similarly for Q.) By the independence, H(P, Q) = sum_k H(Pk, Qk).
        """
        return - dy.sum_dim(dy.cmult(P, self.log2(Q)), [0, 1])

    def cross_entropy(self, p, q):
        return - dy.sum_elems(dy.cmult(p, self.log2(q)))

    def cross_entropy_zero(self, p, q):  # Doesn't handle p(z) > 0, q(z) = 0.
        p_np = p.value()
        return - dy.esum([p[z] * self.log2(q[z]) for z in xrange(p.dim()[0][0])
                          if p_np[z] > 0.0])

    def entropy(self, p): return self.cross_entropy(p, p)
    def entropy_zero(self, p): return self.cross_entropy_zero(p, p)

    def conditional_entropy(self, conditional, prior):
        return dy.esum([prior[z] * self.entropy(conditional[z])
                        for z in xrange(prior.dim()[0][0])])

    def conditional_entropy_zero(self, conditional, prior):
        return dy.esum([prior[z] * self.entropy_zero(conditional[z])
                        for z in xrange(prior.dim()[0][0])])

    def conditional1(self, joint, prior2):
        return self.conditional2(dy.transpose(joint), prior2)

    def conditional2(self, joint, prior1):
        size1, size2 = joint.dim()[0]
        conditional = []
        for z in xrange(size1):
            z_copied = dy.concatenate([prior1[z] for _ in xrange(size2)])
            conditional.append(dy.cdiv(dy.pick(joint, z), z_copied))
        return conditional

    def kl(self, p, q):  # kl doesn't handle zero probabilities.
        return self.cross_entropy(p, q) - self.entropy(p)

    def mi(self, joint):
        prior1 = dy.sum_dim(joint, [1])
        prior2 = dy.sum_dim(joint, [0])
        return self.mi_with_priors(joint, prior1, prior2)

    def mi_with_priors(self, joint, prior1, prior2):
        conditional2 = self.conditional2(joint, prior1)
        return self.entropy(prior2) - self.conditional_entropy(conditional2,
                                                               prior1)

    def mi_zero(self, joint):
        prior1 = dy.sum_dim(joint, [1])
        prior2 = dy.sum_dim(joint, [0])
        return self.mi_zero_with_priors(joint, prior1, prior2)

    def mi_zero_with_priors(self, joint, prior1, prior2):
        size1, size2 = joint.dim()[0]
        joint_np = joint.value()
        weighted_dependences = []
        for z1 in xrange(size1):
            for z2 in xrange(size2):
                if joint_np[z1, z2] > 0.0:
                    weighted_dependences.append(
                        joint[z1][z2] * (self.log2(joint[z1][z2]) -
                                         self.log2(prior1[z1]) -
                                         self.log2(prior2[z2])))
        return dy.esum(weighted_dependences)

    def log2(self, value):
        return dy.cdiv(dy.log(value), dy.log(dy.scalarInput(2.0)))

    def unif(self, size):
        return dy.inputTensor([1.0 / size for _ in xrange(size)])

    def unif_joint(self, size1, size2):
        matrix = np.ones((size1, size2))
        return dy.inputTensor(matrix / sum(sum(matrix)))

    def rand(self, size):
        vector = np.random.rand(size)
        return dy.inputTensor(vector / sum(vector))

    def rand_joint(self, size1, size2):
        matrix = np.random.rand(size1, size2)
        return dy.inputTensor(matrix / sum(sum(matrix)))

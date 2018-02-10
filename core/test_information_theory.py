import math
import unittest
from information_theory import *


class TestInformationTheory(unittest.TestCase):

    def setUp(self):
        self.info = InformationTheory()

    def test_entropy(self):
        entropy_hot = self.info.entropy(dy.inputTensor([0.9999999,
                                                        0.0000001 / 3.0,
                                                        0.0000001 / 3.0,
                                                        0.0000001 / 3.0]))
        self.assertAlmostEqual(entropy_hot.value(), 0.0, places=5)

        entropy_unif = self.info.entropy(dy.inputTensor([0.25, 0.25, 0.25,
                                                         0.25]))
        self.assertAlmostEqual(entropy_unif.value(), 2.0, places=5)

        entropy_random = self.info.entropy(self.info.rand(16)).value()
        self.assertGreaterEqual(entropy_random, 0.0)
        self.assertLessEqual(entropy_random, 4.0)

    def test_entropy_zero(self):
        entropy_hot = self.info.entropy_zero(dy.inputTensor([0.5, 0.0, 0.5,
                                                             0.0]))
        self.assertAlmostEqual(entropy_hot.value(), 1.0, places=5)

        rand = self.info.rand(16)
        entropy_random = self.info.entropy_zero(rand).value()
        self.assertAlmostEqual(entropy_random, self.info.entropy(rand).value(),
                               places=5)

    def test_conditional_entropy(self):
        self.assertAlmostEqual(
            self.info.conditional_entropy([self.info.unif(16)
                                           for y in xrange(4)],
                                          self.info.rand(4)).value(),
            4.0, places=5)

    def test_conditional_entropy_zero(self):
        conditional = [dy.inputTensor([0.5, 0.0, 0.5, 0.0]),
                       dy.inputTensor([0.25, 0.25, 0.25, 0.25]),
                       dy.inputTensor([0.0, 0.0, 0.0, 1.0]),
                       dy.inputTensor([0.25, 0.25, 0.25, 0.25])]
        prior = dy.inputTensor([0.5, 0.0, 0.4, 0.1])

        # (0.5 * 1) + (0.0 * 2) + (0.4 * 0) + (0.1 * 2) = 0.7
        self.assertAlmostEqual(
            self.info.conditional_entropy_zero(conditional, prior).value(),
            0.7, places=5)

    def test_kl(self):
        p = self.info.rand(16)
        q = self.info.rand(16)
        self.assertAlmostEqual(self.info.kl(p, p).value(), 0.0, places=5)
        self.assertGreaterEqual(self.info.kl(p, q).value(), 0.0)

    def test_mi(self):
        joint = dy.inputTensor([[0.4999999, 0.0000001],
                                [0.0000001, 0.4999999]])
        self.assertAlmostEqual(self.info.mi(joint).value(), 1.0, places=5)

        joint = dy.inputTensor([[0.25, 0.25],
                                [0.25, 0.25]])
        self.assertAlmostEqual(self.info.mi(joint).value(), 0.0, places=5)

        # Symmetry
        self.assertAlmostEqual(self.info.mi(joint).value(),
                               self.info.mi(dy.transpose(joint)).value())

    def test_mi_zero(self):
        joint = dy.inputTensor([[0.5, 0.0],
                                [0.0, 0.5]])
        self.assertAlmostEqual(self.info.mi_zero(joint).value(), 1.0, places=5)

        joint = dy.inputTensor([[0.0, 0.0],
                                [1.0, 0.0]])
        self.assertAlmostEqual(self.info.mi_zero(joint).value(), 0.0, places=5)

        joint = self.info.rand_joint(10, 20)
        self.assertAlmostEqual(self.info.mi_zero(joint).value(),
                               self.info.mi(joint).value(),
                               places=5)

    def test_conditional(self):
        joint = dy.inputTensor([[0.2, 0.1],
                                [0.1, 0.2],
                                [0.3, 0.1]])

        prior2 = dy.inputTensor([0.6, 0.4])
        conditional1 = self.info.conditional1(joint, prior2)
        self.assertAlmostEqual(conditional1[0][0].value(), 0.2 / 0.6, places=5)
        self.assertAlmostEqual(conditional1[0][1].value(), 0.1 / 0.6, places=5)
        self.assertAlmostEqual(conditional1[0][2].value(), 0.3 / 0.6, places=5)
        self.assertAlmostEqual(conditional1[1][0].value(), 0.1 / 0.4, places=5)
        self.assertAlmostEqual(conditional1[1][1].value(), 0.2 / 0.4, places=5)
        self.assertAlmostEqual(conditional1[1][2].value(), 0.1 / 0.4, places=5)

        prior1 = dy.inputTensor([0.3, 0.3, 0.4])
        conditional2 = self.info.conditional2(joint, prior1)
        self.assertAlmostEqual(conditional2[0][0].value(), 0.2 / 0.3, places=5)
        self.assertAlmostEqual(conditional2[0][1].value(), 0.1 / 0.3, places=5)
        self.assertAlmostEqual(conditional2[1][0].value(), 0.1 / 0.3, places=5)
        self.assertAlmostEqual(conditional2[1][1].value(), 0.2 / 0.3, places=5)
        self.assertAlmostEqual(conditional2[2][0].value(), 0.3 / 0.4, places=5)
        self.assertAlmostEqual(conditional2[2][1].value(), 0.1 / 0.4, places=5)


if __name__ == '__main__':
    unittest.main()

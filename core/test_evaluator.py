import unittest
from evaluator import *
from information_theory import *


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = Evaluator()
        self.tseqs = [["D", "N", "V", "D", "N"], ["N", "V", "N", "V", "V"]]
        self.zseqs = [[0, 1, 1, 0, 1], [0, 1, 1, 1, 1]]

    def test_count_cooccurence(self):
        cooccur = self.evaluator.count_cooccurence(self.tseqs, self.zseqs)
        self.assertEqual(cooccur[0]["D"], 2)
        self.assertEqual(cooccur[0]["N"], 1)
        self.assertEqual(cooccur[1]["N"], 3)
        self.assertEqual(cooccur[1]["V"], 4)

    def test_get_majority_mapping(self):
        mapping = self.evaluator.get_majority_mapping(self.tseqs, self.zseqs)
        self.assertEqual(mapping[0], "D")
        self.assertEqual(mapping[1], "V")

    def test_compute_many2one_acc(self):
        self.assertEqual(self.evaluator.compute_many2one_acc(self.tseqs,
                                                             self.zseqs), 60.0)

    def test_compute_v_measure(self):
        print self.evaluator.compute_v_measure(self.tseqs, self.zseqs)
        self.assertAlmostEqual(self.evaluator.compute_v_measure(
            self.tseqs, self.zseqs), 0.46336, places=5)

    def test_mi_zero(self):
        info = InformationTheory()  # Assumes the correctness of info.mi(joint).
        joint = info.rand_joint(10, 20)
        self.assertAlmostEqual(self.evaluator.mi_zero(joint.value()),
                               info.mi(joint).value(), places=5)

    def test_compute_mi_bigram(self):
        mi, joint = self.evaluator.compute_mi_bigram(self.zseqs)
        self.assertTrue(np.array_equal(joint,
                                       [[0.0, 3.0 / 12.0, 0.0],
                                        [1.0 / 12.0, 4.0 / 12.0, 2.0 / 12.0],
                                        [2.0 / 12.0, 0.0, 0.0]]))

if __name__ == '__main__':
    unittest.main()

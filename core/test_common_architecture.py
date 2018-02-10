import dynet as dy
import unittest
from model import Model
from common_architecture import CommonArchitecture


class TestCommonArchitecture(unittest.TestCase):

    def setUp(self):
        self.m = dy.ParameterCollection()
        self.lstm1 = dy.LSTMBuilder(1, 1, 1, self.m)
        self.lstm2 = dy.LSTMBuilder(1, 1, 1, self.m)
        self.model = Model()
        self.common = CommonArchitecture(self.model)
        self.inputs = [dy.inputTensor([1.0]),
                       dy.inputTensor([2.0]),
                       dy.inputTensor([3.0]),
                       dy.inputTensor([4.0]),
                       dy.inputTensor([5.0])]
        self.inputs_reverse = [dy.inputTensor([5.0]),
                               dy.inputTensor([4.0]),
                               dy.inputTensor([3.0]),
                               dy.inputTensor([2.0]),
                               dy.inputTensor([1.0])]

    def test_get_bilstm_left_right(self):
        s1 = self.lstm1.initial_state()
        s2 = self.lstm2.initial_state()
        # 1 [2 3] 4 5

        # *f1* -> [f2 -> f3] -> f4 -> f5
        outputs1 = s1.transduce(self.inputs)

        # b1 <- [b2 <- b3] <- *b2* <- b1
        outputs2 = s2.transduce(self.inputs_reverse)[::-1]

        # [f1 b2]
        lr = dy.concatenate([outputs1[0], outputs2[3]])

        subject = self.common.get_bilstm_left_right(self.inputs, self.lstm1,
                                                    self.lstm2, 1, 2)
        for i in xrange(len(lr.value())):
            self.assertAlmostEqual(lr[i].value(), subject[i].value(),
                                   places=5)

    def test_get_bilstm_all(self):
        s1 = self.lstm1.initial_state()
        s2 = self.lstm2.initial_state()

        # f1 -> f2 -> f3 -> f4 -> f5
        outputs1 = s1.transduce(self.inputs)

        # b5 -> b4 -> b3 -> b2 -> b1
        outputs2 = s2.transduce(self.inputs_reverse)

        # b1 -> b2 -> b3 -> b4 -> b5
        outputs2 = outputs2[::-1]

        subject = self.common.get_bilstm_all(self.inputs, self.lstm1,
                                             self.lstm2)
        for i in xrange(len(subject)):
            self.assertAlmostEqual(subject[i].value(),
                                   dy.concatenate([outputs1[i],
                                                   outputs2[i]]).value(),
                                   places=5)

    def test_get_bilstm_single(self):
        s1 = self.lstm1.initial_state()
        s2 = self.lstm2.initial_state()

        # f1 -> f2 -> f3 -> f4 -> f5
        outputs1 = s1.transduce(self.inputs)

        # b5 -> b4 -> b3 -> b2 -> b1
        outputs2 = s2.transduce(self.inputs_reverse)

        # [f5 b1]
        single = dy.concatenate([outputs1[-1], outputs2[-1]])

        subject = self.common.get_bilstm_single(self.inputs, self.lstm1,
                                                self.lstm2)
        for i in xrange(len(single.value())):
            self.assertAlmostEqual(single[i].value(), subject[i].value(),
                                   places=5)

    def test_run_bilstm(self):
        s1 = self.lstm1.initial_state()
        s2 = self.lstm2.initial_state()

        # f1 -> f2 -> f3 -> f4 -> f5
        outputs1 = s1.transduce(self.inputs)

        # b1 <- b2 <- b3 <- b2 <- b1
        outputs2 = s2.transduce(self.inputs_reverse)[::-1]

        subject1, subject2 = self.common.run_bilstm(self.inputs, self.lstm1,
                                                    self.lstm2)
        for i in xrange(len(outputs1)):
            self.assertAlmostEqual(subject1[i].value(), outputs1[i].value(),
                                   places=5)
            self.assertAlmostEqual(subject2[i].value(), outputs2[i].value(),
                                   places=5)

    def test_run_lstm(self):
        s1 = self.lstm1.initial_state()
        outputs1 = s1.transduce(self.inputs)
        subject1 = self.common.run_lstm(self.inputs, self.lstm1)
        for i in xrange(len(outputs1)):
            self.assertAlmostEqual(subject1[i].value(), outputs1[i].value(),
                                   places=5)

        s1_reverse = self.lstm1.initial_state()
        outputs1_reverse = s1.transduce(self.inputs_reverse)
        subject1_reverse = self.common.run_lstm(self.inputs, self.lstm1,
                                                reverse=True)
        for i in xrange(len(outputs1_reverse)):
            self.assertAlmostEqual(subject1_reverse[i].value(),
                                   outputs1_reverse[i].value(),
                                   places=5)


if __name__ == '__main__':
    unittest.main()

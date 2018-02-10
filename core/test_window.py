import unittest
from window import *


class TestWindow(unittest.TestCase):

    def setUp(self):
        self.wseq = ["the", "dog", "saw", "the", "cat"]
        self._BUF = "<*>"
        self.window = Window(self.wseq, self._BUF)

    def test_left(self):
        self.assertEqual(self.window.left(0, 0), [])
        self.assertEqual(self.window.left(0, 1), [self._BUF])
        self.assertEqual(self.window.left(1, 0), [])
        self.assertEqual(self.window.left(1, 1), ["the"])
        self.assertEqual(self.window.left(1, 2), [self._BUF, "the"])

    def test_right(self):
        self.assertEqual(self.window.right(4, 0), [])
        self.assertEqual(self.window.right(4, 1), [self._BUF])
        self.assertEqual(self.window.right(3, 0), [])
        self.assertEqual(self.window.right(3, 1), ["cat"])
        self.assertEqual(self.window.right(3, 2), ["cat", self._BUF])

    def test_left_all(self):
        self.assertEqual(self.window.left_all(0), [])
        self.assertEqual(self.window.left_all(0, buffer_size=2),
                         [self._BUF, self._BUF])
        self.assertEqual(self.window.left_all(3), ["the", "dog", "saw"])
        self.assertEqual(self.window.left_all(3, 1),
                         [self._BUF, "the", "dog", "saw"])

    def test_right_all(self):
        self.assertEqual(self.window.right_all(4), [])
        self.assertEqual(self.window.right_all(4, buffer_size=2),
                         [self._BUF, self._BUF])
        self.assertEqual(self.window.right_all(1), ["saw", "the", "cat"])
        self.assertEqual(self.window.right_all(1, 1),
                         ["saw", "the", "cat", self._BUF])


if __name__ == '__main__':
    unittest.main()

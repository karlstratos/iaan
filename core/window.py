# Author: Karl Stratos (me@karlstratos.com)


class Window(object):

    def __init__(self, wseq, buffer_symbol):
        self.wseq = wseq
        self.buffer_symbol = buffer_symbol

    def left(self, i, width):
        buffering = [self.buffer_symbol for _ in xrange(width - i)]
        left_words = self.wseq[max(0, i - width):i]
        return buffering + left_words

    def right(self, i, width):
        buffering = [self.buffer_symbol for _ in xrange((i + width) -
                                                        (len(self.wseq) - 1))]
        right_words = self.wseq[i + 1: min(len(self.wseq), i + width) + 1]
        return right_words + buffering

    def left_all(self, i, buffer_size=0):
        return self.left(i, i + buffer_size)

    def right_all(self, i, buffer_size=0):
        return self.right(i, len(self.wseq) - i - 1 + buffer_size)

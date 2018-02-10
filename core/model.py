# Author: Karl Stratos (me@karlstratos.com)
import os
import random
import sys
from collections import Counter
from evaluator import *
from jamo import h2j


class Model(object):
    """
    Convention
               w: word string     (ALWAYS encoded to Unicode using UTF-8)
               c: char string
               t: tag string
    """

    def __init__(self, cmd=""):
        self._UNK = unicode("<?>", "utf-8")    # Unknown symbol
        self._BUF = unicode("<*>", "utf-8")    # Buffer symbol
        self._EMP = unicode("<E>", "utf-8")    # Empty symbol (for jamo tail)
        self.evaluator = Evaluator()
        self._is_training = False
        self._input_drop = False
        self._verbose = False
        self._cmd = cmd
        self._activation = "tanh"
        self._log_path = ""
        self.zsize = 0
        self.wdim = 0
        self.cdim = 0
        self.jdim = 0
        self.pseudocount = 0

    def compute_drop_flag(self, item, count):
        """
           1-frequent item gets dropped with probability 0.25/1.25 = 0.2.
        1000-frequent item gets dropped with probability 0.25/1000.25 = 0.00025.
        """
        drop_probability = 0.25 / (count[item] + 0.25)
        uniform_sample = random.random()  # [0, 1)
        return uniform_sample < drop_probability

    def drop(self, item, count):  # item is never UNK.
        """
        Convention: This is invoked only if self._input_drop is set True.
        """
        # Do not drop if
        if ((not self._is_training) or  # 1. the model is training, or
            item == self._BUF or        # 2. the item is BUF, or
            item == self._EMP):         # 3. the item is EMP
            return item

        # Drop probability inversely proportional to item frequency.
        drop_flag = self.compute_drop_flag(item, count)
        return self._UNK if drop_flag else item

    def get_w_ind(self, w):
        w = self.drop(w, self.w_count) if self._input_drop else w
        return self._w2i[w] if w in self._w2i else self._w2i[self._UNK]

    def get_c_ind(self, c):
        c = self.drop(c, self.c_count) if self._input_drop else c
        return self._c2i[c] if c in self._c2i else self._c2i[self._UNK]

    def get_jamo_ind(self, jamo):
        jamo = self.drop(jamo, self.jamo_count) if self._input_drop else jamo
        return self._jamo2i[jamo] if jamo in self._jamo2i \
            else self._jamo2i[self._UNK]

    def wsize(self): return len(self._w2i)
    def csize(self): return len(self._c2i)
    def tsize(self): return len(self._t2i)
    def jamo_size(self): return len(self._jamo2i)

    def _prepare_model_directory(self, model_path):
        if os.path.isfile(model_path): os.remove(model_path)
        if not os.path.exists(model_path): os.makedirs(model_path)
        self._log_path = os.path.join(model_path, "log")
        if os.path.isfile(self._log_path): os.remove(self._log_path)
        with open(self._log_path, 'w') as logf:
            logf.write("[LOG]\n\n")
            if self._cmd: logf.write("  {0}\n\n".format(self._cmd))

    def _log(self, string, newline=True):
        if self._verbose:
            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()
        if self._log_path:
            with open(self._log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

    def read_wseqs(self, wseq_path, tag_path=""):
        with open(wseq_path, 'r') as wseq_file:
            wseqs = [[unicode(w, "utf-8") for w in line.split()]
                     for line in wseq_file if line.split()]

        tseqs = []
        if tag_path:
            with open(tag_path, 'r') as tag_file:
                tseqs = [line.split() for line in tag_file if line.split()]
            assert len(wseqs) == len(tseqs)
            for wseq, tseq in zip(wseqs, tseqs): assert len(wseq) == len(tseq)

        return wseqs, tseqs

    def read_spanseqs(self, span_path, entity_path=""):
        def get_spanseq(line):
            toks = line.split()
            assert len(toks) % 2 == 0
            spanseq = [(int(toks[i]), int(toks[i + 1])) for
                       i in xrange(0, len(toks), 2)]
            return spanseq
        with open(span_path, 'r') as span_file:
            spanseqs = [get_spanseq(line) for line in span_file if line.split()]

        entityseqs = []
        if entity_path:
            with open(entity_path, 'r') as entity_file:
                entityseqs = [line.split() for line in entity_file if
                              line.split()]
            assert len(spanseqs) == len(entityseqs)
            for spanseq, entityseq in zip(spanseqs, entityseqs):
                assert len(spanseq) == len(entityseq)

        return spanseqs, entityseqs

    def read_ner_data(self, data_path, read_entities=True):
        if not data_path: return [], [], []
        wseqs, _ = self.read_wseqs(data_path)

        span_path = data_path.replace(".words", ".spans")
        if not os.path.isfile(span_path):
            raise IOError("can't find span file: {0}".format(span_path))

        if read_entities:
            entity_path = data_path.replace(".words", ".entities")
            if not os.path.isfile(entity_path):
                raise IOError("can't find entity file: {0}".format(entity_path))
        else:
            entity_path = ""

        spanseqs, entityseqs = self.read_spanseqs(span_path, entity_path)
        return wseqs, spanseqs, entityseqs

    def read_wembs(self, wemb_path, wdim=None):
        if not wemb_path: return {}
        wemb = {}
        with open(wemb_path, 'r') as wemb_file:
            for line in wemb_file:
                toks = line.split()
                w, emb = unicode(toks[0], "utf-8"), [float(f) for f in toks[1:]]
                if wdim: assert len(emb) == wdim
                wemb[w] = emb
        return wemb

    def build_dicts(self, wseqs, tseqs, pretrained_wembs={}):
        self._w2i = {}        #   dog   -> 35887
        self._i2w = {}
        self._c2i = {}        #     d   ->    12
        self._i2c = {}
        self._t2i = {}        #  NOUN   ->     3
        self._i2t = {}
        self._jamo2i = {}     #   [jamo]   ->  1
        self._i2jamo = {}
        self.w_count = Counter()
        self.c_count = Counter()
        self.t_count = Counter()
        self.jamo_count = Counter()
        self.num_seqs = len(wseqs)
        self.num_words = 0

        for wseq in wseqs:
            self.num_words += len(wseq)
            for w in wseq:
                self.w_count[w] += 1
                if not w in self._w2i:
                    i = len(self._w2i)
                    self._w2i[w] = i
                    self._i2w[i] = w

                for c in w:
                    self.c_count[c] += 1
                    if not c in self._c2i:
                        i = len(self._c2i)
                        self._c2i[c] = i
                        self._i2c[i] = c

                    if self.jdim > 0:
                        for jamo in h2j(c):
                            self.jamo_count[jamo] += 1
                            if not jamo in self._jamo2i:
                                i = len(self._jamo2i)
                                self._jamo2i[jamo] = i
                                self._i2jamo[i] = jamo

        for tseq in tseqs:
            for t in tseq:
                self.t_count[t] += 1
                if not t in self._t2i:
                    i = len(self._t2i)
                    self._t2i[t] = i
                    self._i2t[i] = t

        # Expand word/char dicts based on pretrained embeddings (if any).
        self.num_pretrained_wembs = len(pretrained_wembs)
        for w in pretrained_wembs:
            if isinstance(w, str):  # Possibly given ascii string wembs
                w = unicode(w, "utf-8")
            if not w in self._w2i:
                i = len(self._w2i)
                self._w2i[w] = i
                self._i2w[i] = w
                self.w_count[w] = 1

                for c in w:
                    if not c in self._c2i:
                        i = len(self._c2i)
                        self._c2i[c] = i
                        self._i2c[i] = c
                        self.c_count[c] = 1

                    if self.jdim > 0:
                        for jamo in h2j(c):
                            if not jamo in self._jamo2i:
                                i = len(self._jamo2i)
                                self._jamo2i[jamo] = i
                                self._i2jamo[i] = jamo
                                self.jamo_count[jamo] = 1

        # Add the BUF word symbol.
        assert not self._BUF in self._w2i
        w_BUF_i = len(self._w2i)
        self._w2i[self._BUF] = w_BUF_i
        self._i2w[w_BUF_i] = self._BUF

        # Add the UNK word/char symbol.
        assert not (self._UNK in self._w2i or self._UNK in self._c2i)
        w_UNK_i = len(self._w2i)
        c_UNK_i = len(self._c2i)
        self._w2i[self._UNK] = w_UNK_i
        self._c2i[self._UNK] = c_UNK_i
        self._i2w[w_UNK_i] = self._UNK
        self._i2c[c_UNK_i] = self._UNK

        if self.jdim > 0:
            assert not (self._UNK in self._jamo2i or self._EMP in self._jamo2i)
            jamo_UNK_i = len(self._jamo2i)
            self._jamo2i[self._UNK] = jamo_UNK_i
            self._i2jamo[jamo_UNK_i] = self._UNK
            jamo_EMP_i = len(self._jamo2i)
            self._jamo2i[self._EMP] = jamo_EMP_i
            self._i2jamo[jamo_EMP_i] = self._EMP

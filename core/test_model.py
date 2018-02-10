# -*- coding: utf-8 -*-
#
#   ===> unicode("some string", "utf-8") = u"some string"
import unittest
from model import *


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data_en_path = "data/pos/sample_en.words"
        self.gold_en_path = "data/pos/sample_en.tags"
        self.wemb_en_path = "data/pos/sample_en.emb"

        self.data_ko_path = "data/pos/sample_ko.words"
        self.gold_ko_path = "data/pos/sample_ko.tags"

        self.data_ja_path = "data/pos/sample_ja.words"

    def assert_counts_en(self, model):
        self.assertEqual(model.w_count[u"the"], 6)
        self.assertEqual(model.w_count[u"chased"], 3)
        self.assertEqual(model.w_count[u"cat"], 2)
        self.assertEqual(model.w_count[u"mouse"], 2)
        self.assertEqual(model.w_count[u"dog"], 2)

        self.assertEqual(model.c_count[u'e'], 11)
        self.assertEqual(model.c_count[u'h'], 9)
        self.assertEqual(model.c_count[u't'], 8)
        self.assertEqual(model.c_count[u'c'], 5)
        self.assertEqual(model.c_count[u'a'], 5)
        self.assertEqual(model.c_count[u's'], 5)
        self.assertEqual(model.c_count[u'd'], 5)
        self.assertEqual(model.c_count[u'o'], 4)
        self.assertEqual(model.c_count[u'm'], 2)
        self.assertEqual(model.c_count[u'u'], 2)
        self.assertEqual(model.c_count[u'g'], 2)

        self.assertEqual(model.t_count["D"], 6)
        self.assertEqual(model.t_count["N"], 6)
        self.assertEqual(model.t_count["V"], 3)

    def test_build_dicts_en(self):
        model = Model()
        wseqs, tseqs = model.read_wseqs(self.data_en_path, self.gold_en_path)
        model.build_dicts(wseqs, tseqs)

        self.assertEqual(model.wsize(), 7)  # UNK, BUF
        self.assertEqual(model.csize(), 12)  # UNK
        self.assertEqual(model.tsize(), 3)
        self.assertEqual(model.num_seqs, 3)
        self.assertEqual(model.num_words, 15)
        self.assert_counts_en(model)

    def test_pretrained_wembs_en(self):
        model = Model()
        wseqs, tseqs = model.read_wseqs(self.data_en_path, self.gold_en_path)
        wembs = model.read_wembs(self.wemb_en_path)
        model.build_dicts(wseqs, tseqs, wembs)

        self.assertEqual(model.wsize(), 8)  # "cow" new
        self.assertEqual(model.csize(), 13)  # "w" new
        self.assert_counts_en(model)
        self.assertEqual(model.w_count["cow"], 1)  # new word
        self.assertEqual(model.c_count['w'], 1)  # new char

    def assert_counts_ko(self, model, use_jamo=False):
        self.assertEqual(model.w_count[u"고양이"], 2)
        self.assertEqual(model.w_count[u"생쥐"], 2)
        self.assertEqual(model.w_count[u"개"], 2)
        self.assertEqual(model.w_count[u"가"], 3)
        self.assertEqual(model.w_count[u"를"], 3)
        self.assertEqual(model.w_count[u"쫓았다"], 3)

        self.assertEqual(model.c_count[u'고'], 2)
        self.assertEqual(model.c_count[u'양'], 2)
        self.assertEqual(model.c_count[u'이'], 2)
        self.assertEqual(model.c_count[u'가'], 3)
        self.assertEqual(model.c_count[u'생'], 2)
        self.assertEqual(model.c_count[u'쥐'], 2)
        self.assertEqual(model.c_count[u'를'], 3)
        self.assertEqual(model.c_count[u'쫓'], 3)
        self.assertEqual(model.c_count[u'았'], 3)
        self.assertEqual(model.c_count[u'다'], 3)
        self.assertEqual(model.c_count[u'개'], 2)

        if use_jamo:
            # There are *different* versions of Unicode for Korean jamo letters:
            # "Hangul Jamo" U+[1100-11FF] and "Hangul Compatibility Jamo"
            # U+[3131-318E].
            #
            #    - When you type it on Mac, you get Hangul Compatibility Jamo.
            #    - But the Python jamo processor gives you Hangul Jamo.
            #
            # So the following jamo letters are copied and pasted (!) from my
            # Python output. They will *not* work if you type them yourself.
            self.assertEqual(model.jamo_count[u'ᄀ'], 7)
            self.assertEqual(model.jamo_count[u'ᄉ'], 2)
            self.assertEqual(model.jamo_count[u'ᆻ'], 3)
            self.assertEqual(model.jamo_count[u'ᄌ'], 2)
            self.assertEqual(model.jamo_count[u'ᄃ'], 3)
            self.assertEqual(model.jamo_count[u'ᄍ'], 3)
            self.assertEqual(model.jamo_count[u'ᆾ'], 3)
            self.assertEqual(model.jamo_count[u'ᄅ'], 3)  # head consonant
            self.assertEqual(model.jamo_count[u'ᆯ'], 3)  # tail consonant
            self.assertEqual(model.jamo_count[u'ᅩ'], 5)
            self.assertEqual(model.jamo_count[u'ᅳ'], 3)
            self.assertEqual(model.jamo_count[u'ᄋ'], 7)  # head consonant
            self.assertEqual(model.jamo_count[u'ᆼ'], 4)  # tail consonant
            self.assertEqual(model.jamo_count[u'ᅣ'], 2)
            self.assertEqual(model.jamo_count[u'ᅵ'], 2)
            self.assertEqual(model.jamo_count[u'ᅡ'], 9)
            self.assertEqual(model.jamo_count[u'ᅢ'], 4)
            self.assertEqual(model.jamo_count[u'ᅱ'], 2)
        else:
            self.assertEqual(model.jamo_size(), 0)

        self.assertEqual(model.t_count["N"], 6)
        self.assertEqual(model.t_count["O"], 6)
        self.assertEqual(model.t_count["V"], 3)

    def test_build_dicts_ko(self):
        model = Model()
        wseqs, tseqs = model.read_wseqs(self.data_ko_path, self.gold_ko_path)
        model.jdim = 1  # Build a jamo dictionary.
        model.build_dicts(wseqs, tseqs)

        self.assertEqual(model.wsize(), 8)  # UNK, BUF
        self.assertEqual(model.csize(), 12)  # UNK
        self.assertEqual(model.tsize(), 3)
        self.assertEqual(model.jamo_size(), 20)  # UNK, EMP
        self.assertEqual(model.num_seqs, 3)
        self.assertEqual(model.num_words, 15)
        self.assert_counts_ko(model, model.jdim > 0)

        model.jdim = 0  # Do not build a jamo dictionary.
        model.build_dicts(wseqs, tseqs)
        self.assert_counts_ko(model, model.jdim > 0)

    def test_pretrained_wembs_ko(self):
        model = Model()
        wseqs, tseqs = model.read_wseqs(self.data_ko_path, self.gold_ko_path)
        wembs = {"#트윗": [0.1, 0.5]}
        model.jdim = 1  # Build a jamo dictionary.
        model.build_dicts(wseqs, tseqs, wembs)

        self.assertEqual(model.wsize(), 9)  # +1
        self.assertEqual(model.csize(), 15)  # +3
        self.assertEqual(model.jamo_size(), 23)  # +3 (including '#')
        self.assert_counts_ko(model, model.jdim > 0)
        self.assertEqual(model.w_count[u"#트윗"], 1)  # new word
        self.assertEqual(model.c_count[u'#'], 1)  # new char
        self.assertEqual(model.c_count[u'트'], 1)  # new char
        self.assertEqual(model.c_count[u'윗'], 1)  # new char
        self.assertEqual(model.jamo_count[u'#'], 1)  # new jamo
        self.assertEqual(model.jamo_count[u'ᄐ'], 1)  # new jamo
        self.assertEqual(model.jamo_count[u'ᆺ'], 1)  # new jamo

    def assert_counts_ja(self, model):
        self.assertEqual(model.w_count[u"猫"], 2)
        self.assertEqual(model.w_count[u"マウス"], 2)
        self.assertEqual(model.w_count[u"犬"], 2)
        self.assertEqual(model.w_count[u"は"], 3)
        self.assertEqual(model.w_count[u"を"], 3)
        self.assertEqual(model.w_count[u"追いかけた"], 3)

        self.assertEqual(model.c_count[u'猫'], 2)
        self.assertEqual(model.c_count[u'マ'], 2)
        self.assertEqual(model.c_count[u'ウ'], 2)
        self.assertEqual(model.c_count[u'ス'], 2)
        self.assertEqual(model.c_count[u'犬'], 2)
        self.assertEqual(model.c_count[u'は'], 3)
        self.assertEqual(model.c_count[u'を'], 3)
        self.assertEqual(model.c_count[u'追'], 3)
        self.assertEqual(model.c_count[u'い'], 3)
        self.assertEqual(model.c_count[u'か'], 3)
        self.assertEqual(model.c_count[u'け'], 3)
        self.assertEqual(model.c_count[u'た'], 3)

    def test_build_dicts_ja(self):
        model = Model()
        wseqs, tseqs = model.read_wseqs(self.data_ja_path, "")
        model.build_dicts(wseqs, [])

        self.assertEqual(model.wsize(), 8)  # UNK, BUF
        self.assertEqual(model.csize(), 13)  # UNK
        self.assertEqual(model.tsize(), 0)
        self.assertEqual(model.num_seqs, 3)
        self.assertEqual(model.num_words, 15)
        self.assert_counts_ja(model)

    def test_get_ind_special_symbols(self):
        model = Model()
        wseqs, tseqs = model.read_wseqs(self.data_ko_path, self.gold_ko_path)
        model.jdim = 1  # Build a jamo dictionary.
        model.build_dicts(wseqs, tseqs)

        BUF_i = model._w2i[model._BUF]
        EMP_i = model._jamo2i[model._EMP]
        w_i = model._w2i[u"고양이"]
        c_i = model._c2i[u'고']
        jamo_i = model._jamo2i[u'ᄃ']

        model._is_training = False
        model._input_drop = False

        # Should never be dropped.
        for _ in xrange(10000):
            self.assertEqual(model.get_w_ind(model._BUF), BUF_i)
            self.assertEqual(model.get_jamo_ind(model._EMP), EMP_i)
            self.assertEqual(model.get_w_ind(u"고양이"), w_i)
            self.assertEqual(model.get_c_ind(u'고'), c_i)
            self.assertEqual(model.get_jamo_ind(u'ᄃ'), jamo_i)

        model._is_training = True
        model._input_drop = True
        w_dropped_at_least_once = False
        c_dropped_at_least_once = False
        jamo_dropped_at_least_once = False
        for _ in xrange(10000):
            # BUF and EMP should still never be dropped.
            self.assertEqual(model.get_w_ind(model._BUF), BUF_i)
            self.assertEqual(model.get_jamo_ind(model._EMP), EMP_i)

            # Non-special word/jamo/char should be dropped sometimes.
            if model.get_w_ind(u"고양이") == model._w2i[model._UNK]:
                w_dropped_at_least_once = True
            if model.get_c_ind(u'고') == model._c2i[model._UNK]:
                c_dropped_at_least_once = True
            if model.get_jamo_ind(u'ᄃ') == model._jamo2i[model._UNK]:
                jamo_dropped_at_least_once = True

        self.assertTrue(w_dropped_at_least_once)
        self.assertTrue(c_dropped_at_least_once)
        self.assertTrue(jamo_dropped_at_least_once)


if __name__ == '__main__':
    unittest.main()

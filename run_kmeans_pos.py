# Author: Karl Stratos (me@karlstratos.com)
#
# python run_kmeans_pos.py  data/pos/conll2003/conll2003.train.words  data/pos/conll2003/conll2003.dev.words  --zsize 4 --emb data/pos/conll2003/rcv1.emb --pred /tmp/pred
# python run_kmeans_pos.py  data/pos/conll2003/conll2003.train.words  data/pos/conll2003/conll2003.dev.words  --zsize 4 --rand --pred /tmp/pred
import argparse
import codecs
import numpy as np
import pickle
from core.model import *
from sklearn.cluster import KMeans


class PartOfSpeechKMeans(Model):

    def __init__(self, cmd=""):
        self._test = ""
        super(PartOfSpeechKMeans, self).__init__(cmd)

    def config(self, zsize, verbose=False):
        self.zsize = zsize
        self._verbose = verbose

    def train(self, wseqs, wemb_path, tseqs):
        wdict = {}
        for wseq in wseqs:
            for w in wseq: wdict[w] = True

        wembs = self.read_wembs(wemb_path)
        dim = len(wembs.items()[0][1])
        X = []
        i = 0
        w2i = {}
        i2w = {}
        for w in wdict:
            X.append(wembs[w] if w in wembs else np.zeros((dim)))
            w2i[w] = i
            i2w[i] = w
            i += 1
        print "-----------------"
        print "Clustering {0} words into {1} groups".format(len(X), self.zsize)
        X = np.array(X)
        kmeans = KMeans(n_clusters=self.zsize, random_state=0).fit(X)
        print "-----------------"
        predmap = kmeans.predict(X)
        pseqs = []
        for wseq in wseqs:
            pseq = []
            for w in wseq:
                pseq.append(predmap[w2i[w]])
            pseqs.append(pseq)
        acc = self.evaluator.compute_many2one_acc(tseqs, pseqs)
        print "Accuracy: {0:.2f}".format(acc)

        c2w = {}
        for i, c in enumerate(predmap):
            if not c in c2w: c2w[c] = []
            c2w[c].append(i2w[i])
        for c in c2w:
            print c
            print c2w[c]
            print


def main(args):
    random.seed(42)
    model = PartOfSpeechKMeans(' '.join(sys.argv))
    wseqs, tseqs = model.read_wseqs(args.data, args.gold)

    model.config(args.zsize, args.verbose)
    model.train(wseqs, args.emb, tseqs)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data", type=str,
                           help="text data (one sentence per line)")
    argparser.add_argument("gold", type=str,
                           help="gold labels (one tag sequence per line)")
    argparser.add_argument("--zsize", type=int, default=45,
                           help="number of tags: %(default)d")
    argparser.add_argument("--emb", type=str,
                           help="path to pretrained word embeddings")
    argparser.add_argument("--pred", type=str,
                           help="prediction output file")
    argparser.add_argument("--verbose", action="store_true",
                           help="print?")

    parsed_args = argparser.parse_args()
    main(parsed_args)

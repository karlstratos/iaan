# Author: Karl Stratos (me@karlstratos.com)
#
# python run_kmeans_ner.py  data/ner/conll2003/conll2003.train.words  data/ner/conll2003/conll2003.dev.words  --zsize 4 --emb data/ner/conll2003/rcv1.emb --pred /tmp/pred
# python run_kmeans_ner.py  data/ner/conll2003/conll2003.train.words  data/ner/conll2003/conll2003.dev.words  --zsize 4 --rand --pred /tmp/pred
import argparse
import codecs
import numpy as np
import pickle
from core.model import *
from sklearn.cluster import KMeans


class NamedEntityKMeans(Model):

    def __init__(self, cmd=""):
        self._test = ""
        super(NamedEntityKMeans, self).__init__(cmd)

    def config(self, zsize, verbose=False):
        self.zsize = zsize
        self._verbose = verbose

    def train(self, wseqs, spanseqs, wemb_path, rand, pred):
        (wseqs_test, spanseqs_test,
         entityseqs_test) = self.read_ner_data(self._test, read_entities=True)

        if rand:
            pseqs_test = []
            for i in xrange(len(wseqs_test)):
                pseqs_test.append(np.random.choice(self.zsize,
                                                   len(spanseqs_test[i])))
            acc = self.evaluator.compute_many2one_acc(entityseqs_test,
                                                      pseqs_test)
            print "Random accuracy: {0:.2f}".format(acc)
            return

        wembs = self.read_wembs(wemb_path)
        dim = len(wembs.items()[0][1])
        X = []
        for i in xrange(len(wseqs)):
            for j in xrange(len(spanseqs[i])):
                s, t = spanseqs[i][j]
                words = wseqs[i][s:t + 1]
                v = np.zeros((dim))
                num = 0
                for w in words:
                    if w in wembs:
                        num += 1
                        v += wembs[w]
                X.append(v / num if num > 0 else v)
        print "-----------------"
        print "Clustering {0} spans into {1} groups".format(len(X), self.zsize)
        X = np.array(X)
        kmeans = KMeans(n_clusters=self.zsize, random_state=0).fit(X)
        Y = []
        for i in xrange(len(wseqs_test)):
            for j in xrange(len(spanseqs_test[i])):
                s, t = spanseqs_test[i][j]
                words = wseqs_test[i][s:t + 1]
                v = np.zeros((dim))
                num = 0
                for w in words:
                    if w in wembs:
                        num += 1
                        v += wembs[w]
                Y.append(v / num if num > 0 else v)
        print "-----------------"
        print "Predicting groups for {0} test spans".format(len(Y))
        Y = np.array(Y)
        predseq = kmeans.predict(Y)
        goldseq = []
        for entityseq_test in entityseqs_test:
            goldseq.extend(entityseq_test)
        acc = self.evaluator.compute_many2one_acc([goldseq], [predseq])
        print "Accuracy: {0:.2f}".format(acc)

        if pred:
            k = 0
            with codecs.open(pred, 'w', "utf-8") as outf:
                for i in xrange(len(wseqs_test)):
                    for j in xrange(len(spanseqs_test[i])):

                        start, end = spanseqs_test[i][j]
                        mention = '_'.join(wseqs_test[i][start:end + 1])
                        outf.write(mention + '\t' +
                                   entityseqs_test[i][j] + '\t' +
                                   str(predseq[k]))
                        k += 1
                        outf.write('\n')
                    outf.write('\n')


def main(args):
    random.seed(42)
    model = NamedEntityKMeans(' '.join(sys.argv))
    wseqs, spanseqs, _ = model.read_ner_data(args.data)
    model._test = args.test

    model.config(args.zsize, args.verbose)
    model.train(wseqs, spanseqs, args.emb, args.rand, args.pred)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data", type=str,
                           help="text data (one sentence per line)")
    argparser.add_argument("test", type=str,
                           help="test data")
    argparser.add_argument("--zsize", type=int, default=4,
                           help="number of tags: %(default)d")
    argparser.add_argument("--emb", type=str,
                           help="path to pretrained word embeddings")
    argparser.add_argument("--rand", action="store_true",
                           help="random prediction?")
    argparser.add_argument("--pred", type=str,
                           help="prediction output file")
    argparser.add_argument("--verbose", action="store_true",
                           help="print?")

    parsed_args = argparser.parse_args()
    main(parsed_args)

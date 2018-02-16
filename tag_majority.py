# Author: Karl Stratos (me@karlstratos.com)
#
# python tag_majority.py data/pos/orig/en.words data/pos/orig/en.tags --pred tmp/en45_majority.pred
import argparse
import pickle
import time
from core.evaluator import *
from core.model import *


class Majority(Model):
    """Supervised majority tag mapper."""

    def train(self, wseqs, tseqs):
        evaluator = Evaluator()
        self.mapping = evaluator.get_majority_mapping(tseqs, wseqs)
        zseqs = self.tag_all(wseqs)
        acc = evaluator.compute_many2one_acc(tseqs, zseqs)
        return zseqs, acc

    def tag(self, wseq): return [self.mapping[w] for w in wseq]
    def tag_all(self, wseqs): return [self.tag(wseq) for wseq in wseqs]

def main(args):
    model = Majority(' '.join(sys.argv))
    wseqs, tseqs = model.read_wseqs(args.data, args.gold)

    zseqs, acc = model.train(wseqs, tseqs)
    print "acc: {0:.2f}".format(acc)
    if args.pred:
        with open(args.pred, 'w') as outf:
            for i in xrange(len(wseqs)):
                for (w, t, z) in zip(wseqs[i], tseqs[i], zseqs[i]):
                    outf.write(w + '\t' + t + '\t' + z + '\n')
                outf.write('\n')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data", type=str,
                           help="text data (one sentence per line)")
    argparser.add_argument("gold", type=str,
                           help="gold labels (one tag sequence per line)")
    argparser.add_argument("--pred", type=str,
                           help="prediction output file")

    parsed_args = argparser.parse_args()
    main(parsed_args)

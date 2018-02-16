# Author: Karl Stratos (me@karlstratos.com)
#
# python predict_brown.py tmp/en45.brown data/pos/orig/en.words data/pos/orig/en.tags --pred tmp/pred.en45_brown
import argparse
import sys
from collections import Counter
from core.evaluator import *


def read_brown(brown_path):
    C = {}
    z2i = {}
    with open(brown_path) as inf:
        for line in inf:
            if line:
                z, w, _ = line.split()
                if not z in z2i:
                    i = len(z2i)
                    z2i[z] = i
                C[w] = z2i[z]
    return C, z2i


def read_seqs(data, gold, C):
    wseqs = []
    tseqs = []
    zseqs = []

    with open(data) as inf:
        for line in inf:
            if line:
                wseq = line.split()
                zseq = [C[w] for w in wseq]
                wseqs.append(wseq)
                zseqs.append(zseq)

    with open(gold) as inf:
        for line in inf:
            if line:
                tseqs.append(line.split())

    return wseqs, tseqs, zseqs


def evaluate(tseqs, zseqs):
    evaluator = Evaluator()
    acc = evaluator.compute_many2one_acc(tseqs, zseqs)
    return acc


def main(args):
    C, z2i = read_brown(args.brown)
    wseqs, tseqs, zseqs = read_seqs(args.data, args.gold, C)
    acc = evaluate(tseqs, zseqs)
    print 'acc: {0:.3}'.format(acc)
    if args.pred:
        with open(args.pred, 'w') as outf:
            for i in xrange(len(wseqs)):
                for (w, t, z) in zip(wseqs[i], tseqs[i], zseqs[i]):
                    outf.write(w + '\t' + t + '\t' + str(z) + '\n')
                outf.write('\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("brown", type=str,
                           help="brown clustering on data (full coverage)")
    argparser.add_argument("data", type=str,
                           help="text data (one sentence per line)")
    argparser.add_argument("gold", type=str,
                           help="gold labels (one tag sequence per line)")
    argparser.add_argument("--pred", type=str,
                           help="prediction output file")
    parsed_args = argparser.parse_args()
    main(parsed_args)

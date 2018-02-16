# Author: Karl Stratos (me@karlstratos.com)
#
# python analyze_pred.py tmp/pred.en45_brown --choice 1 --threshold 10
# python analyze_pred.py tmp/pred.en45_mutualizer --choice 2 --threshold 10
import argparse
import numpy as np
import sys
from collections import Counter
from core.evaluator import *


def read(pred_path, choice):
    with open(pred_path) as pred_file:
        lines = pred_file.readlines()
    wseqs = []
    tseqs = []
    zseqs = []

    wseq = []
    tseq = []
    zseq = []

    tcount = Counter()
    zcount = Counter()

    for line in lines:
        toks = line.split()
        if toks:
            w = toks[0]
            t = toks[1]
            z = toks[1 + choice]
            wseq.append(w)
            tseq.append(t)
            zseq.append(z)
            tcount[t] += 1
            zcount[z] += 1
        else:
            if wseq:
                wseqs.append(wseq)
                tseqs.append(tseq)
                zseqs.append(zseq)
                wseq = []
                tseq = []
                zseq = []
    if wseq:
        wseqs.append(wseq)
        tseqs.append(tseq)
        zseqs.append(zseq)

    return wseqs, tseqs, zseqs, tcount, zcount


def count(tseqs, mseqs):  # mseqs[i][j] = mapping(zseqs[i][j])
    num_instances = sum([len(tseq) for tseq in tseqs])
    t2m_count = {}
    acc = 0.0
    num_errors = {}
    for i in xrange(len(mseqs)):
        for j in xrange(len(mseqs[i])):
            m = mseqs[i][j]
            t = tseqs[i][j]
            num_errors[t] = 0
            if m != t: num_errors[t] += 1
            if m == t: acc += 100.0 / num_instances
            if not t in t2m_count: t2m_count[t] = Counter()
            t2m_count[t][m] += 1
    return t2m_count, acc, num_errors


def get_max_tag_errors(t2z_count):
    items = []
    for t in t2z_count:
        for z in t2z_count[t]:
            if t != z:
                items.append(((t, z), t2z_count[t][z]))
    slist = sorted(items, key=lambda x: x[1], reverse=True)
    return slist


def main(args):
    evaluator = Evaluator()
    wseqs, tseqs, zseqs, tcount, zcount = read(args.pred_path, args.choice)
    mapping = evaluator.get_majority_mapping(tseqs, zseqs)
    mseqs = [[mapping[z] for z in zseq] for zseq in zseqs]
    mcount = Counter()
    for z in zcount: mcount[mapping[z]] += zcount[z]
    t2m_count, acc, num_errors = count(tseqs, mseqs)

    mi_t, _ = evaluator.compute_mi_bigram(tseqs)
    mi_z, _ = evaluator.compute_mi_bigram(zseqs)
    mi_m, _ = evaluator.compute_mi_bigram(mseqs)

    print "--------------"
    print "  acc:     {0:.4}".format(acc)
    print "tsize:     {0}".format(len(tcount))
    print "zsize:     {0}".format(len(zcount))
    print "msize:     {0}".format(len(mcount))
    print "t bigram MI: {0:.3}".format(mi_t)
    print "z bigram MI: {0:.3}".format(mi_z)
    print "m bigram MI: {0:.3}".format(mi_m)
    print "--------------"

    print
    print "----------------"
    print "Majority mapping"
    print "----------------"
    mapped_z = {}
    for z in zcount:
        m = mapping[z]
        if not m in mapped_z: mapped_z[m] = []
        mapped_z[m].append(z)
    sorted_list = sorted(mapped_z.items(), key=lambda x: len(x[1]),
                         reverse=True)
    for m, zs in sorted_list:
        print '{0:6s}\t({1})\t'.format(m, len(zs)),
        for z in zs: print z,
        print
    print


    print
    print "Most frequent t vs m"
    print
    for (t, ct_t), (m, ct_m) in zip(sorted(tcount.items(), key=lambda x: x[1],
                                           reverse=True)[:args.threshold],
                                    sorted(mcount.items(), key=lambda x: x[1],
                                           reverse=True)[:args.threshold]):
        print "{0:>15}{1:>15}  {2:>15}{3:>15}".format(t, ct_t, m, ct_m)

    print
    print "Max tag errors"
    print
    print "{0:>15}{1:>15}{2:>15}".format("Tag", "Predicted", "This Many")
    print "           --------------------------------------------------"
    slist = get_max_tag_errors(t2m_count)
    for (t, m), ct in slist[:min(len(slist), args.threshold)]:
        print "{0:>15}{1:>15}{2:>15}".format(t, m, ct)
    print

    if args.out:
        with open(args.out, 'w') as outf:
            for i in xrange(len(wseqs)):
                for j in xrange(len(wseqs[i])):
                    w = wseqs[i][j]
                    t = tseqs[i][j]
                    m = mseqs[i][j]
                    outf.write(w + '\t' + t + '\t' + m + '\n')
                outf.write('\n')

    if args.conf:
        print
        print "Confusion matrix"
        print
        serr = sorted(num_errors.items(), key=lambda x: x[1], reverse=True)
        sorted_ts = zip(*serr)[0]
        sorted_ms = [t for t in sorted_ts if t in mcount]
        print ("{:>6}" * (len(sorted_ms) + 1)).format("", *sorted_ms)
        for t in sorted_ts:
            sys.stdout.write("{0:>6}".format(t))
            for m in sorted_ms:
                ct = t2m_count[t][m] if m in t2m_count[t] else 0
                if ct and t != m:
                    sys.stdout.write("{0:>6}".format(ct))
                else:
                    sys.stdout.write("{0:>6}".format(""))
            sys.stdout.write("\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("pred_path", type=str,
                           help="prediction file")
    argparser.add_argument("--choice", type=int, default=2,
                           help="choice of zseqs to analyze: %(default)d")
    argparser.add_argument("--threshold", type=int, default=10,
                           help="threshold: %(default)d")
    argparser.add_argument("--out", type=str,
                           help="output path to majority-mapped predictions")
    argparser.add_argument("--conf", action="store_true",
                           help="show confusion table?")

    parsed_args = argparser.parse_args()
    main(parsed_args)

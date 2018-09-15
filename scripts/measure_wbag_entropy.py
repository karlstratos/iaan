# Author: Karl Stratos (me@karlstratos.com)

import argparse
import math
import sys
from collections import Counter

def main(args):
    vocab_article_count = Counter()
    num_articles = 0

    print "data: {0}".format(args.data)
    with open(args.data) as f:
        num_articles = 0
        for line in f:
            num_articles += 1
            if num_articles % 10000 == 0:
                print num_articles,
                sys.stdout.flush()
            toks = line.split()
            present = {tok:True for tok in toks}
            for tok in present:
                vocab_article_count[tok] += 1
        print

    print
    print "{0} articles".format(num_articles)
    print "{0} distinct words".format(len(vocab_article_count))

    onprobs = {word: float(vocab_article_count[word]) / num_articles
             for word in vocab_article_count}
    pairs = sorted(onprobs.items(), key=lambda x: x[1], reverse=True)

    print
    print "{0} largest occurrence probabilities".format(args.N)
    for i in xrange(args.N):
        print "{0:20s} {1:10f}".format(pairs[i][0], pairs[i][1])

    print
    print "{0} smallest occurrence probabilities".format(args.N)
    for i in xrange(1, args.N + 1):
        print "{0:20s} {1:10f}".format(pairs[-i][0], pairs[-i][1])

    Hps = {}
    for word in onprobs:
        p = onprobs[word]
        Hp = 0.0
        if p > 0.0: Hp -= p * math.log(p, 2.0)
        if p < 1.0: Hp -= (1 - p) * math.log(1 - p, 2.0)
        Hps[word] = Hp
    pairs = sorted(Hps.items(), key=lambda x: x[1], reverse=True)

    print
    print "{0} largest occurrence entropies".format(args.N)
    for i in xrange(args.N):
        print "{0:20s} {1:10f}".format(pairs[i][0], pairs[i][1])


    entropy = sum([Hps[word] for word in Hps])
    print
    print "Entropy (in binary bits): {0:5f}".format(entropy)
    print "Upper bound: {0:0f}".format(len(Hps))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data", type=str,
                           help="path to data")
    argparser.add_argument("--N", type=int, default=10,
                           help="number of examples to show")
    parsed_args = argparser.parse_args()
    main(parsed_args)

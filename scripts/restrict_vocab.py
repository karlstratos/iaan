# Author: Karl Stratos (me@karlstratos.com)
import argparse
import sys
from collections import Counter

def main(args):
    if args.data2: assert args.output2
    vocab = Counter()
    num_words = 0

    print "data: {0}".format(args.data)
    with open(args.data) as f:
        line_num = 0
        for line in f:
            line_num += 1
            if line_num % 10000 == 0:
                print line_num,
                sys.stdout.flush()
            toks = line.split()
            num_words += len(toks)
            for tok in toks:
                vocab[tok] += 1
        print

    if args.data2:
        print "second data: {0}".format(args.data2)
        with open(args.data2) as f:
            line_num = 0
            for line in f:
                line_num += 1
                if line_num % 10000 == 0:
                    print line_num,
                    sys.stdout.flush()
                toks = line.split()
                num_words += len(toks)
                for tok in toks:
                    vocab[tok] += 1
            print

    print
    print "{0} distinct words in {1} total words".format(len(vocab), num_words)

    pairs = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    cumulative_count_10000 = 0
    cumulative_count_50000 = 0
    cumulative_count_100000 = 0
    cumulative_count_200000 = 0
    cumulative_count_300000 = 0
    for i, (_, count) in enumerate(pairs):
        if i < 10000: cumulative_count_10000 += count
        if i < 50000: cumulative_count_50000 += count
        if i < 100000: cumulative_count_100000 += count
        if i < 200000: cumulative_count_200000 += count
        if i < 300000: cumulative_count_300000 += count

    print
    print "{0:.1f}% count coverage with 10k vocab".format(
        float(cumulative_count_10000) / num_words * 100)
    print "{0:.1f}% count coverage with 50k vocab".format(
        float(cumulative_count_50000) / num_words * 100)
    print "{0:.1f}% count coverage with 100k vocab".format(
        float(cumulative_count_100000) / num_words * 100)
    print "{0:.1f}% count coverage with 200k vocab".format(
        float(cumulative_count_200000) / num_words * 100)
    print "{0:.1f}% count coverage with 300k vocab".format(
        float(cumulative_count_300000) / num_words * 100)

    vocab_size = int(raw_input("Enter vocab size: "))
    allowed_vocab = {}
    for i in xrange(vocab_size):
        allowed_vocab[pairs[i][0]] = True

    print "writing data to: {0}".format(args.output)
    with open(args.output, 'w') as outf:
        with open(args.data) as f:
            line_num = 0
            for line in f:
                line_num += 1
                if line_num % 10000 == 0:
                    print line_num,
                    sys.stdout.flush()
                toks = line.split()
                for i in xrange(len(toks)):
                    if not toks[i] in allowed_vocab: toks[i] = args.unk
                outf.write(' '.join(toks) + '\n')
            print

    if args.data2:
        print "writing second data to: {0}".format(args.output2)
        with open(args.output2, 'w') as outf:
            with open(args.data2) as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    if line_num % 10000 == 0:
                        print line_num,
                        sys.stdout.flush()
                    toks = line.split()
                    for i in xrange(len(toks)):
                        if not toks[i] in allowed_vocab: toks[i] = args.unk
                    outf.write(' '.join(toks) + '\n')
                print


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data", type=str,
                           help="path to data")
    argparser.add_argument("output", type=str,
                           help="path to output")
    argparser.add_argument("--data2", type=str,
                           help="path to second data")
    argparser.add_argument("--output2", type=str,
                           help="path to second output")
    argparser.add_argument("--unk", type=str, default="<?>",
                           help="symbol for rare words: %(default)s")
    parsed_args = argparser.parse_args()
    main(parsed_args)

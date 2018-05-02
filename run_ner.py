# Author: Karl Stratos (me@karlstratos.com)
import argparse
import codecs
import random
import sys
from core.ner import NamedEntityInducer


def main(args):
    random.seed(42)
    model = NamedEntityInducer(' '.join(sys.argv))

    if args.train:
        wseqs, spanseqs, _ = model.read_ner_data(args.data, read_entities=False)
        model._dev = args.dev

        model.config(args.arch, args.loss, args.zsize, args.wdim,
                     args.cdim, args.width, args.swap, args.smooth,
                     args.verbose)

        model.train(args.model, wseqs, spanseqs, args.lrate, args.drate,
                    args.epochs, args.batch, args.emb)

    else:
        wseqs, spanseqs, entityseqs = model.read_ner_data(args.data,
                                                          read_entities=True)

        model.load_and_populate(args.model)
        zseqs_X, zseqs_Y, zseqs_XY, infer_time = model.tag_all(wseqs, spanseqs)
        model._verbose = True
        model.common.evaluator_report(wseqs, entityseqs, zseqs_X, zseqs_Y,
                                      zseqs_XY, infer_time,
                                      model.measure_mi(wseqs, spanseqs), "m2o",
                                      newline=True)
        if args.pred:
            with codecs.open(args.pred, 'w', "utf-8") as outf:
                for i in xrange(len(wseqs)):
                    for j in xrange(len(spanseqs[i])):
                        start, end = spanseqs[i][j]
                        mention = '_'.join(wseqs[i][start:end + 1])
                        outf.write(mention + '\t' +
                                   entityseqs[i][j] + '\t' +
                                   str(zseqs_X[i][j]) + '\t' +
                                   str(zseqs_Y[i][j]))
                        if zseqs_XY: outf.write('\t' + str(zseqs_XY[i][j]))
                        outf.write('\n')
                    outf.write('\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model", type=str,
                           help="path to model directory")
    argparser.add_argument("data", type=str,
                           help="text data (one sentence per line), "
                           "must end in .words and be accompanied by "
                           ".spans/.entities files")
    argparser.add_argument("--dev", type=str,
                           help="dev data (one sentence per line), "
                           "must end in .words and be accompanied by "
                           ".span/.entities files")
    argparser.add_argument("--train", action="store_true",
                           help="train a new model?")
    argparser.add_argument("--arch", type=str, default="default",
                           help="architecture: %(default)s")
    argparser.add_argument("--loss", type=str, default="lb",
                           help="loss: %(default)s")
    argparser.add_argument("--zsize", type=int, default=12,
                           help="number of tags: %(default)d")
    argparser.add_argument("--wdim", type=int, default=100,
                           help="word embedding dimension: %(default)d")
    argparser.add_argument("--cdim", type=int, default=50,
                           help="chararcter embedding dimension: %(default)d")
    argparser.add_argument("--width", type=int, default=2,
                           help="width for context words: %(default)d")
    argparser.add_argument("--swap", action="store_true",
                           help="swap X and Y architectures?")
    argparser.add_argument("--smooth", type=int, default=1,
                           help="smoothing param for raw MI: %(default)d")
    argparser.add_argument("--emb", type=str,
                           help="path to pretrained word embeddings")
    argparser.add_argument("--lrate", type=float, default=0.001,
                           help="learning rate: %(default)f")
    argparser.add_argument("--drate", type=float, default=0.0,
                           help="dropout rate: %(default)f")
    argparser.add_argument("--epochs", type=int, default=30,
                           help="number of training epochs: %(default)d")
    argparser.add_argument("--batch", type=int, default=0,
                           help="batch size (0 if sentence): %(default)d")
    argparser.add_argument("--pred", type=str,
                           help="prediction output file")
    argparser.add_argument("--verbose", action="store_true",
                           help="print?")
    argparser.add_argument("--dynet-mem", type=int, default=512)      # cmd
    argparser.add_argument("--dynet-seed", type=int, default=42)      # cmd
    argparser.add_argument("--dynet-autobatch", type=int, default=1)  # cmd

    parsed_args = argparser.parse_args()
    main(parsed_args)

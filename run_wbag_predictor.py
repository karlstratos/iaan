# Author: Karl Stratos (me@karlstratos.com)
#
# python run_wbag_predictor.py --dynet-mem 512 --dynet-seed 42 --dynet-autobatch 1 --verbose /tmp/model --X data/semantics/val_context_articles.txt.10 --Y data/semantics/val_question_articles.txt.10  --devX data/semantics/val_context_articles.txt.10 --devY data/semantics/val_question_articles.txt.10 --train --epochs 100 --lrate 0.01
#
# python run_wbag_predictor.py --dynet-mem 512 --dynet-seed 42 --dynet-autobatch 1 --verbose /tmp/model --testX data/semantics/val_context_articles.txt.10 --testY data/semantics/val_question_articles.txt.10
import argparse
import codecs
import random
import sys
from core.wbag_predictor import WBagPredictor

def main(args):
    random.seed(42)
    model = WBagPredictor(' '.join(sys.argv))

    if args.train:
        articles_X = model.read_articles(args.X)
        articles_Y = model.read_articles(args.Y)
        dev_articles_X = model.read_articles(args.devX)
        dev_articles_Y = model.read_articles(args.devY)
        model.config(args.arch, args.wdim, args.hdim, args.verbose)
        model.train(args.model, articles_X, articles_Y, dev_articles_X,
                    dev_articles_Y, args.lrate, args.epochs)
    else:
        test_articles_X = model.read_articles(args.testX)
        test_articles_Y = model.read_articles(args.testY)
        model.load_and_populate(args.model)
        hY_X = model.test(test_articles_X, test_articles_Y)
        print hY_X


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model", type=str,
                           help="path to model directory")
    argparser.add_argument("--X", type=str)
    argparser.add_argument("--Y", type=str)
    argparser.add_argument("--devX", type=str,
                           default="data/semantics/val_context_articles.txt")
    argparser.add_argument("--devY", type=str,
                           default="data/semantics/val_question_articles.txt")
    argparser.add_argument("--testX", type=str,
                           default="data/semantics/test_context_articles.txt")
    argparser.add_argument("--testY", type=str,
                           default="data/semantics/test_question_articles.txt")
    argparser.add_argument("--train", action="store_true",
                           help="train a new model?")
    argparser.add_argument("--arch", type=str, default="wbag",
                           help="architecture: %(default)s")
    argparser.add_argument("--wdim", type=int, default=100,
                           help="word embedding dimension: %(default)d")
    argparser.add_argument("--hdim", type=int, default=50,
                           help="LSTM dimension: %(default)d")
    argparser.add_argument("--lrate", type=float, default=0.01,
                           help="learning rate: %(default)f")
    argparser.add_argument("--epochs", type=int, default=30,
                           help="number of training epochs: %(default)d")
    argparser.add_argument("--verbose", action="store_true",
                           help="print?")
    argparser.add_argument("--dynet-gpu", action="store_true")      # cmd
    argparser.add_argument("--dynet-mem", type=int, default=512)      # cmd
    argparser.add_argument("--dynet-seed", type=int, default=42)      # cmd
    argparser.add_argument("--dynet-autobatch", type=int, default=1)  # cmd

    parsed_args = argparser.parse_args()
    main(parsed_args)

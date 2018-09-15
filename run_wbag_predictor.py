# Author: Karl Stratos (me@karlstratos.com)
import argparse
import codecs
import random
import sys
from core.wbag_predictor import WBagPredictor

train_X = "data/semantics/train_context_articles_vocab100k.txt.1000"
train_Y = "data/semantics/train_question_articles_vocab100k.txt.1000"
dev_X = "data/semantics/val_context_articles.txt.10"
dev_Y = "data/semantics/val_question_articles.txt.10"
test_X = "data/semantics/test_context_articles.txt.10"
test_Y = "data/semantics/test_question_articles.txt.10"

def main(args):
    random.seed(42)
    model = WBagPredictor(' '.join(sys.argv))

    if args.train:
        articles_X = model.read_articles(train_X)
        articles_Y = model.read_articles(train_Y)
        dev_articles_X = model.read_articles(dev_X)
        dev_articles_Y = model.read_articles(dev_Y)
        model.config(args.arch, args.wdim, args.hdim, args.verbose)
        model.train(args.model, articles_X, articles_Y, dev_articles_X,
                    dev_articles_Y, args.lrate, args.epochs)
    else:
        test_articles_X = model.read_articles(test_X)
        test_articles_Y = model.read_articles(test_Y)
        model.load_and_populate(args.model)
        hY_X = model.test(test_articles_X, test_articles_Y)
        print hY_X


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model", type=str,
                           help="path to model directory")
    argparser.add_argument("--train", action="store_true",
                           help="train a new model?")
    argparser.add_argument("--arch", type=str, default="wbag",
                           help="architecture: %(default)s")
    argparser.add_argument("--wdim", type=int, default=100,
                           help="word embedding dimension: %(default)d")
    argparser.add_argument("--hdim", type=int, default=50,
                           help="LSTM dimension: %(default)d")
    argparser.add_argument("--lrate", type=float, default=0.001,
                           help="learning rate: %(default)f")
    argparser.add_argument("--epochs", type=int, default=30,
                           help="number of training epochs: %(default)d")
    argparser.add_argument("--verbose", action="store_true",
                           help="print?")
    argparser.add_argument("--dynet-mem", type=int, default=512)      # cmd
    argparser.add_argument("--dynet-seed", type=int, default=42)      # cmd
    argparser.add_argument("--dynet-autobatch", type=int, default=1)  # cmd

    parsed_args = argparser.parse_args()
    main(parsed_args)

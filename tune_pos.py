# Author: Karl Stratos (me@karlstratos.com)
#
# python tune_pos.py /scratch/scratch/pos_tuning_en45_wdim200_cdim100_width2_epoch30_batch0_default_lb data/pos/orig/en.words data/pos/orig/en.tags --zsize 45 --wdim 200 --cdim 100 --width 2 --epochs 30 --batch 0 --arch default --loss lb
import argparse
import os
import re
import subprocess
import sys
from multiprocessing import Pool

LRATES = [0.005, 0.001, 0.0005, 0.0001]  # TODO: randomized search?
DRATES = [0, 0.1, 0.2]
NUM_WORKERS = min(16, len(LRATES) * len(DRATES))

def get_model_path(path, lrate, drate):
    model_path = "{0}_lrate{1}_drate{2}".format(path, lrate, drate)
    return model_path

def f((mem, seed, autobatch, model_path, data, gold, arch, zsize, wdim, cdim,
       jdim, lrate, drate, epochs, batch, metric, width, swap, emb, loss_type)):
    cmd = "python run_pos.py "
    cmd += "--dynet-mem {0} ".format(mem)
    cmd += "--dynet-seed {0} ".format(seed)
    cmd += "--dynet-autobatch {0} ".format(autobatch)
    cmd += "{0} ".format(model_path)
    cmd += "{0} ".format(data)
    cmd += "--gold {0} ".format(gold)
    cmd += "--train "
    cmd += "--arch {0} ".format(arch)
    cmd += "--zsize {0} ".format(zsize)
    cmd += "--wdim {0} ".format(wdim)
    cmd += "--cdim {0} ".format(cdim)
    cmd += "--jdim {0} ".format(jdim)
    cmd += "--lrate {0} ".format(lrate)
    cmd += "--drate {0} ".format(drate)
    cmd += "--epochs {0} ".format(epochs)
    cmd += "--batch {0} ".format(batch)
    cmd += "--metric {0} ".format(metric)
    cmd += "--width {0} ".format(width)
    if swap: cmd += "--swap "
    if emb: cmd += "--emb {0} ".format(emb)
    cmd += "--loss {0} ".format(loss_type)
    subprocess.call(cmd, shell=True)

def get_best_result(model_path):
    with open(os.path.join(model_path, "log")) as logfile:
        lines = logfile.readlines()
    epoch = 1
    acc = 0.0
    negloss = 0.0
    for line in lines:
        epoch_match_list = re.findall("Epoch\s+(\d+)", line)
        negloss_match_list = re.findall("loss:\s+-(\d+\.\d+)", line)
        new_best_match_list = re.findall("new best\s+(\d+\.\d+)", line)
        if negloss_match_list:
            assert len(negloss_match_list) == 1
            this_negloss = float(negloss_match_list[0])
            if this_negloss > negloss: negloss = this_negloss
        if new_best_match_list:
            assert len(epoch_match_list) == 1
            assert len(new_best_match_list) == 1
            epoch = int(epoch_match_list[0])
            acc = float(new_best_match_list[0])

    return epoch, acc, negloss

def main(args):
    p = Pool(NUM_WORKERS)
    configs = []
    for lrate in LRATES:
        for drate in DRATES:
            model_path = get_model_path(args.path, lrate, drate)
            configs.append((args.dynet_mem, args.dynet_seed,
                            args.dynet_autobatch, model_path, args.data,
                            args.gold, args.arch, args.zsize,
                            args.wdim, args.cdim, args.jdim, lrate, drate,
                            args.epochs, args.batch, args.metric, args.width,
                            args.swap, args.emb, args.loss))
    p.map(f, configs)

    results = {}
    for lrate in LRATES:
        for drate in DRATES:
            model_path = get_model_path(args.path, lrate, drate)
            results[(lrate, drate)] = get_best_result(model_path)

    # Print tables.
    print
    print "acc"
    sys.stdout.write("drate\\lrate")
    for lrate in LRATES: sys.stdout.write(" {0}".format(lrate))
    sys.stdout.write("\n")
    for drate in DRATES:
        sys.stdout.write("{0:.1f}".format(drate))
        for lrate in LRATES:
            sys.stdout.write(" {0:.2f}".format(results[(lrate, drate)][1]))
        print

    print
    print "epoch"
    sys.stdout.write("drate\\lrate")
    for lrate in LRATES: sys.stdout.write(" {0}".format(lrate))
    sys.stdout.write("\n")
    for drate in DRATES:
        sys.stdout.write("{0:.1f}".format(drate))
        for lrate in LRATES:
            sys.stdout.write(" {0}".format(results[(lrate, drate)][0]))
        print

    print
    print "-loss"
    sys.stdout.write("drate\\lrate")
    for lrate in LRATES: sys.stdout.write(" {0}".format(lrate))
    sys.stdout.write("\n")
    for drate in DRATES:
        sys.stdout.write("{0:.1f}".format(drate))
        for lrate in LRATES:
            sys.stdout.write(" {0}".format(results[(lrate, drate)][2]))
        print



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str,
                           help="prefix for model paths (e.g., /tmp/model)")
    argparser.add_argument("data", type=str,
                           help="text data (one sentence per line)")
    argparser.add_argument("gold", type=str,
                           help="gold labels (one tag sequence per line)")
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
    argparser.add_argument("--jdim", type=int, default=0,
                           help="jamo embedding dimension: %(default)d")
    argparser.add_argument("--width", type=int, default=2,
                           help="width for context words: %(default)d")
    argparser.add_argument("--swap", action="store_true",
                           help="swap X and Y architectures?")
    argparser.add_argument("--emb", type=str,
                           help="path to pretrained word embeddings")
    argparser.add_argument("--epochs", type=int, default=30,
                           help="number of training epochs: %(default)d")
    argparser.add_argument("--batch", type=int, default=0,
                           help="batch size (0 if sentence): %(default)d")
    argparser.add_argument("--metric", type=str, default="m2o",
                           help="evaluation metric")
    argparser.add_argument("--verbose", action="store_true",
                           help="print?")
    argparser.add_argument("--dynet-mem", type=int, default=512)
    argparser.add_argument("--dynet-seed", type=int, default=42)
    argparser.add_argument("--dynet-autobatch", type=int, default=1)

    parsed_args = argparser.parse_args()
    main(parsed_args)

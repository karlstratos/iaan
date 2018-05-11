# Author: Karl Stratos (me@karlstratos.com)
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

def main(args):
    epochs = []
    neglosses = []
    MIs = []
    with open(args.log, 'r') as logfile:
        for line in logfile:
            epoch_match_list = re.findall("Epoch\s+(\d+)", line)
            negloss_match_list = re.findall("loss:\s+-(\d+\.\d+)", line)
            MI_match_list = re.findall("MI:\s+(\d+\.\d+)", line)
            if negloss_match_list:
                assert len(negloss_match_list) == 1
                epoch = int(epoch_match_list[0])
                negloss = float(negloss_match_list[0])
                MI = float(MI_match_list[0])
                epochs.append(epoch)
                neglosses.append(negloss)
                MIs.append(MI)

    print epochs
    print neglosses
    print MIs
    plt.xlabel('Epochs')
    plt.ylabel('Mutual Information (Bits)')
    plt.plot(epochs, neglosses, label="per-batch objective")
    plt.plot(epochs, MIs, label="mutual information")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("log", type=str,
                           help="log file")
    parsed_args = argparser.parse_args()
    main(parsed_args)

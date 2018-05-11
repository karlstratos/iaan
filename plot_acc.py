# Author: Karl Stratos (me@karlstratos.com)
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

def main(args):
    epochs = []
    Xs = []
    Ys = []
    XYs = []
    with open(args.log, 'r') as logfile:
        for line in logfile:
            epoch_match_list = re.findall("Epoch\s+(\d+)", line)
            X_match_list = re.findall("X acc:\s+(\d+\.\d+)", line)
            Y_match_list = re.findall("Y acc:\s+(\d+\.\d+)", line)
            XY_match_list = re.findall("XY acc:\s+(\d+\.\d+)", line)
            if epoch_match_list:
                epoch = int(epoch_match_list[0])
                X = float(X_match_list[0])
                Y = float(Y_match_list[0])
                XY = float(XY_match_list[0])
                epochs.append(epoch)
                Xs.append(X)
                Ys.append(Y)
                XYs.append(XY)

    print epochs
    print Xs
    print Ys
    print XYs
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, XYs, label="XY")
    plt.plot(epochs, Ys, label="Y")
    plt.plot(epochs, Xs, label="X")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("log", type=str,
                           help="log file")
    parsed_args = argparser.parse_args()
    main(parsed_args)

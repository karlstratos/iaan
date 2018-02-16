# Author: Karl Stratos (me@karlstratos.com)
import argparse
import dynet as dy
import matplotlib.pyplot as plt
import numpy as np
import random
from core.information_theory import InformationTheory

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    info = InformationTheory()
    num_points_except_end = args.num_points - 1
    stepsize = 1.0 / num_points_except_end
    found = False
    epsilon = 1e-6

    while not found:
        Omega1 = info.rand_joint(args.zsize, args.zsize)
        Omega2 = info.rand_joint(args.zsize, args.zsize)
        #Omega1 = dy.inputTensor([[1.0, 0.0],
        #                         [0.0, 0.1]]) # NOT doubly stochastic!
        #Omega2 = dy.inputTensor([[0.0, 0.1],
        #                         [0.1, 0.0]])
        Omega1 = dy.inputTensor([[0.4940,  0.3006],
                                 [0.1383, 0.0671]])
        Omega2 = dy.inputTensor([[0.1513, 0.2415],
                                 [0.2545, 0.3527]])


        print
        print "Going from: "
        print Omega1.value()
        print "to"
        print Omega2.value()
        print

        alpha = 0
        point_indices = []
        mi_values = []
        increasing = False
        decreasing = False
        num_turns = 0

        for point_index in xrange(args.num_points):
            Omega = (1.0 - alpha) * Omega1  + alpha * Omega2
            mi_value = info.mi_zero(Omega).value()
            point_indices.append(point_index + 1)
            mi_values.append(mi_value)
            alpha += stepsize

            if point_index == 1:
                print "point {0}, MI: {1} -> {2}".format(point_index + 1,
                                                         mi_value_before,
                                                         mi_value),
                if mi_value > mi_value_before:
                    increasing = True
                    decreasing = False
                if mi_value < mi_value_before:
                    increasing = False
                    decreasing = True

                if increasing:
                    print "increasing"
                if decreasing:
                    print "decreasing"

            elif point_index > 1:
                if increasing:
                    print "point {0} increasing, now MI: {1} -> {2}".format(
                        point_index + 1, mi_value_before, mi_value),

                    if mi_value < mi_value_before - epsilon:
                        increasing = False
                        decreasing = True
                        print "inc->dec",
                        num_turns += 1
                        print "TURNED {0} times".format(num_turns),
                        if num_turns == args.turn and not found:
                            print " ------ FOUND",
                            found = True
                    print

                if decreasing:
                    print "point {0} decreasing, now MI: {1} -> {2}".format(
                        point_index + 1, mi_value_before, mi_value),

                    if mi_value > mi_value_before +  epsilon:
                        increasing = True
                        decreasing = False
                        print "dec->inc",
                        num_turns += 1
                        print "TURNED {0} times".format(num_turns),
                        if num_turns == args.turn and not found:
                            print " ------ FOUND",
                            found = True
                    print

            mi_value_before = mi_value
        #break

    assert len(point_indices) == args.num_points
    assert len(mi_values) == args.num_points
    plt.plot(point_indices, mi_values)
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--zsize", type=int, default=2,
                           help="number of variables: %(default)d")
    argparser.add_argument("--num-points", type=int, default=100,
                           help="number of interpolated points: %(default)d")
    argparser.add_argument("--turn", type=int, default=3,
                           help="number of turns: %(default)d")
    argparser.add_argument("--seed", type=int, default=1024,
                           help="random seed: %(default)d")

    parsed_args = argparser.parse_args()
    main(parsed_args)

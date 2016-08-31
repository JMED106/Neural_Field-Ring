import getopt
import sys

import numpy as np

from tools import DictToObj

__author__ = 'Jose M. Esnaola Acebes'

""" Load this file to easily handle data saved in main. It loads the numpy object saved
    in ./results and converts it to a Dict. The latter is converted to an object to
    be able to use the dot notation, instead of the brackets typical of dictionaries.
    To use it in python: run load_data.py -f <name_of_file>
"""

pi = np.pi


def __init__(argv):
    try:
        opts, args = getopt.getopt(argv, "hf:", ["file="])
    except getopt.GetoptError:
        print 'load_data.py [-f <file>]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'load_data.py [-f <file>]'
            sys.exit()
        elif opt in ("-f", "--file"):
            fin = arg
            return fin
        else:
            print 'load_data.py [-f <file>]'
            sys.exit()


fin = __init__(sys.argv[1:])
d = np.load(fin)
data = DictToObj(d.item())
phi = np.linspace(-pi, pi, data.parameters.l)
phip = np.linspace(-pi, pi, data.parameters.l + 1)

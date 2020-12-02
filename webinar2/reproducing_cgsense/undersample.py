#!/usr/bin/env python3

import numpy as np
import cfl
import argparse

parser = argparse.ArgumentParser(description='Undersample cfl')
parser.add_argument('dim', metavar='D', type=int, action='store',
                    help='Dimension')
parser.add_argument('acc', metavar='acc', type=int, 
                    help='Acceleration factor')
parser.add_argument('input', metavar='IN', type=str, 
                    help='Input cfl')
parser.add_argument('output', metavar='OUT', type=str, 
                    help='Output cfl')

args = parser.parse_args()

in_ = cfl.readcfl(args.input)

tmp = np.moveaxis(in_, args.dim, 0)

out_ = np.moveaxis(tmp[::args.acc,...], 0, args.dim)

cfl.writecfl(args.output, out_)


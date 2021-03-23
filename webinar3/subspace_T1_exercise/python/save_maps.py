#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 
# Copyright 2020-2021. Uecker Lab, University Medical Center Goettingen.
#
# Author: xiaoqing.wang@med.uni-goettingen.de

"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
import cfl
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class save_maps(object):
   
        
    def __init__(self, image, colorbartype, lowerbound, upperbound, outfile):
        
        colorbartype =  str(colorbartype)
               
        viridis = cm.get_cmap(colorbartype, 256)
        newcolors = viridis(np.linspace(0, 1, 256))
        dark = np.array([0.0, 0.0, 0.0, 1])
        newcolors[:1, :] = dark
        newcmp = ListedColormap(newcolors)
        
        
        plt.figure()
        plt.imshow(abs(image),interpolation='nearest',cmap=newcmp,vmin=lowerbound, vmax=upperbound)
#        plt.show() # do not use show here, so that it does not require an X server
        
        plt.imsave(outfile, image, format="png",cmap=newcmp,vmin=lowerbound, vmax=upperbound);

if __name__ == "__main__":
    
    #Error if wrong number of parameters
    if( len(sys.argv) != 6):
        print( "Function for saving quantitative MR maps with colormap" )
        print( "Usage: save_maps.py <input image> <colorbartype(e.g., viridis)> <lowerbound> <upperbound> <outfile>" )
        exit()
        
    image = np.abs(cfl.readcfl(sys.argv[1]).squeeze())
    save_maps(image, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

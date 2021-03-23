#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 
# Copyright 2020. Uecker Lab, University Medical Center Goettingen.
#
# Authors: Nick Scholand, 2019-2020
# nick.scholand@med.uni-goettingen.de
# Xiaoqing Wang. 2019-2020
# xiaoqing.wang@med.uni-goettingen.de
"""

import numpy as np
import time
from scipy.optimize import curve_fit
import math
from pylab import *

import sys
import os
sys.path.insert(0, os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
import cfl

def T2_func(x, a, b):
    return a * np.exp( -1 / b * x )

def T1s_func(x, a, b, c):
    return np.abs(a - (a + b) * np.exp( -x * c))


class mapping_piecewise(object):
    
    def getmap(self, data, TI):
        
        max_dim = np.shape( data )
        
        store = np.zeros( (max_dim[0], max_dim[1], 6) ) # 6 = 3 parameter maps with 3 error maps
        nf = max_dim[2]
        
        time = np.abs(TI)
                
        for x in range(0, max_dim[0] ):       
            for y in range(0, max_dim[1]):
                    
#                    print( "Pixel: ("+str(x)+", "+str(y)+")" )
                    
                    signal = data[x, y, 0:nf]
                    
                    #Do fitting
                    if (self.para == "T2"):
                        popt, pcov = curve_fit(T2_func, time, signal, p0=(1, 0.1))
                        
                    elif (self.para == "T1"):
                        try:
                            popt, pcov = curve_fit(T1s_func, time, signal, p0=(1.0, 1.0, 1.0))
                        except RuntimeError:
                            store[x, y, 0] = 0
                            store[x, y, 1] = 0
                            store[x, y, 2] = 0
                            store[x, y, 3] = 0
                            store[x, y, 4] = 0
                            store[x, y, 5] = 0
                            continue                           
                        
                    else:
                        print("No parameter is chosen! Please specify one!")
                        raise
                        
                    #calculate error
                    perr = np.sqrt(np.diag(pcov))
                                            
                    store[x, y, 0] = popt[0]
                    store[x, y, 1] = popt[1]
                    store[x, y, 2] = popt[2]
                    store[x, y, 3] = perr[0] 
                    store[x, y, 4] = perr[1]
                    store[x, y, 5] = perr[2]
        
        return store
    
   
        
    def __init__(self, infile, para, TI, outfile):  
        self.infile = sys.argv[1]
        self.para = sys.argv[2]
        self.TIfile = sys.argv[3]
        self.outfile = sys.argv[4]
        
        start = time.time()
       
        self.oridata = np.array( cfl.readcfl(self.infile).squeeze() ) #dim = [x, y, time, slice]
        self.TI = np.array( cfl.readcfl(self.TIfile).squeeze() ) 
        
        a = np.mean(self.oridata[:,:,-1])
        
        self.oridata = 1.0*self.oridata/a
        
        self.map = self.getmap( np.abs(self.oridata), self.TI )
     
        cfl.writecfl(self.outfile, self.map)
        
        end = time.time()
        print("Ellapsed time: " + str(end - start) + " s")


        
if __name__ == "__main__":
    #Error if wrong number of parameters
    if( len(sys.argv) != 5):
        print( "Function for creating T1 and T2 maps from cfl image." )
        print( "Usage: mapping_piecewise.py <infile> <maptype(T1 or T2)>  <inversion/echo time vector> <outfile>" )
        exit()
        
    mapping_piecewise( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] )
    

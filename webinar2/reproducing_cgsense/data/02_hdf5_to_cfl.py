#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import os
import os.path
import cfl


with os.scandir(os.path.curdir) as dirit:
    for f in dirit:
        prefix = 'rawdata_'
        if f.name.endswith(".h5"): 
            #remove rawdata_ prefix and .h5 extension
            if f.name.startswith(prefix):
                fstrip = f.name[len(prefix):-3]
            else:
                fstrip = f.name[:-3]
            h5_dataset = h5py.File(f, 'r')
            outdir = fstrip + "_cfl"
            os.makedirs(outdir, exist_ok=True)
            for key in list(h5_dataset.keys()):
                keydata = h5_dataset.get(key)[()]
                cfl.writecfl(os.path.join(outdir, key), keydata)

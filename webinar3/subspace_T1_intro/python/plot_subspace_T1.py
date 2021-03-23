
import os
import os.path
import sys
sys.path.insert(0, os.path.join(os.environ['TOOLBOX_PATH'], 'python'))

#import warnings
#warnings.simplefilter('error', RuntimeWarning)

from cfl import readcfl
from cfl import writecfl
import numpy as np

import scipy.misc
from scipy import ndimage
#import sys

import importlib

#import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import math
from random import randrange


def array_plot1(ax, arr, rows=4, cols=4):
    idx = 0
    xticks = 6e-3*np.arange(0, len(arr[:,idx]), 1)
    for idx in range(rows*cols):
        ax.plot(xticks, np.real(arr[:,idx]),linewidth=2, markersize=5,label="Comp. {}".format(idx+1))
        ax.legend()
        idx += 1 
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Signal')
    ax.set_title('Temporal Subspace Curves')

 
def array_plot2(ax, arr, rows=4, cols=4):
    idx = 0
    xticks = 6e-3*np.arange(0, len(arr[:,idx]), 1)
    for idx in range(rows*cols):
        ax.plot(xticks, np.real(arr[:,idx]),linewidth=2, label=str(idx))
        idx += 1 
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Signal')
    ax.set_title('Simulated Dictionary (subset)')

#---------------------------------------------------------------#
Signal = readcfl("./dicc2")

if False:
    random_index = list(np.random.choice(100000, 7, replace=False))
else:
    random_index = [18921, 47522, 45034, 14515, 90331, 7023, 74182]
Signal1 = Signal[:,random_index]


S = readcfl("./S")

U = readcfl("./U")


U_truc = U[:,0:5]



S1 = S/np.max(abs(S))

S2 = np.cumsum(S1,axis=0)

xticks = np.arange(1, 31, 1)
plt.rcParams.update({'font.size': 11})
fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (6.4*3, 4.8))

ax2.plot(xticks, np.abs(S2[0:30])/np.real(S2[-1]),'b^-',linewidth=2, markersize=6)
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Percentage [100%]')
ax2.set_title('Accumulated PCA Coefficients')

array_plot2(ax1, Signal1, rows=7, cols=1)
array_plot1(ax3, U_truc, rows=5, cols=1)

fig.savefig("Subspace_T1.png", dpi=300, bbox_inches='tight')#, pad_inches=0)

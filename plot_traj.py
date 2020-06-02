# -*- coding: utf-8 -*-
"""
  Author: 
  2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
"""

import sys

# you need to specify your own python directory under bart
sys.path.append('/home/ztan/bart_webinar/python')
import cfl

import matplotlib.pyplot as plt
import numpy as np

def plot_traj(traj_file, png_file):
    
    traj = cfl.readcfl(traj_file)
    print('traj dims: ', traj.shape)

    D = traj.ndim
    if D < 16:
        
        for d in range(D, 16, 1):
            traj = np.expand_dims(traj, axis=d)
    
    assert traj.ndim == 16

    N_spk = traj.shape[ 2]
    N_eco = traj.shape[ 5]
    N_frm = traj.shape[10] 

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    for nf in range(N_frm):

        ax = plt.subplot(1,N_frm, nf+1)

        for ne in range(N_eco):

            for ns in range(N_spk):

                kx = np.squeeze(traj[0,:,ns,0,0,ne,0,0,0,0,nf])
                ky = np.squeeze(traj[1,:,ns,0,0,ne,0,0,0,0,nf])
                plt.plot(kx.real, ky.real, '.g')
                plt.axis('off')
                ax.set_aspect('equal', 'box')
                plt.text(kx[-1].real, ky[-1].real, str(ns), fontsize=16)
    
    plt.savefig(png_file, bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == "__main__":

    plot_traj(sys.argv[1], sys.argv[2])

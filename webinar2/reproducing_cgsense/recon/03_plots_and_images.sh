#!/usr/bin/python3



import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import os
import cfl

os.makedirs("figures", exist_ok=True)

braindir="brain_radial_96proj_12ch_cfl"
heartdir="heart_radial_55proj_34ch_cfl"
UNDERS={}
UNDERS[braindir]=('96','48','32','24')
UNDERS[heartdir]=('55','33','22','11')

coil_ims = []
single_it_ims = []
cg_ims = []
for u in UNDERS[braindir]:
    nu = cfl.readcfl(braindir + "/" + u + "/nufft_" + u).squeeze()
    coil_ims.append(nu[::-1,::-1,...,0])
    cg_ims.append(cfl.readcfl(braindir + "/pics_u_" + u + "_InScale")[::-1,::-1,...])
    single_it_ims.append(cfl.readcfl(braindir + "/pics_u_" + u + "_it_1_InScale")[::-1,::-1,...])
    
# Figure 1: Comparison brain reconstruction
# 4 x 3 figure 
# Accelerations in rows
# first coil nufft, 1 step CG and 30 step CG in cols

figure = plt.figure(figsize=(5, 6))
figure.subplots_adjust(hspace=0.1, wspace=0.1)
gs = mp.gridspec.GridSpec(
            4, 3)
gs.tight_layout(figure)
figure.patch.set_facecolor('w')
ax = []
for grid in gs:
    ax.append(plt.subplot(grid))
    ax[-1].grid(False)
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])

for j in range(4):
    ax[3*j].imshow(np.abs(coil_ims[j]), cmap='gray')
    ax[3*j+1].imshow(np.abs(single_it_ims[j]), cmap='gray')
    ax[3*j+2].imshow(np.abs(cg_ims[j]), cmap='gray')
    ax[3*j].set_ylabel("Acc " + str(j+1), rotation=0, labelpad=20)
    ax[3*j+1].text(
      cg_ims[j].shape[0]-20, cg_ims[j].shape[1]-5, "1", color="w")
    ax[3*j+2].text(cg_ims[j].shape[-1]-50, cg_ims[j].shape[-1]-5,
                  str(30), color="w")
    if j == 0:
        ax[3*j].set_title("Single coil")
        ax[3*j+1].set_title("Initial")
        ax[3*j+2].set_title("Final")
plt.savefig("./figures/bart_Comparison_Reconstruction_Brain.png",dpi=600)

# Read logs

delta = np.empty((4,30))
Delta = np.empty((3,30))
for i,u in enumerate(UNDERS[braindir]):
    it_delta = 0
    it_Delta = 0
    with open(braindir + "/log_pics_u_" + u, "r") as logf:
        for line in logf:
            if line.startswith('#'):
                delta[i, it_delta] = float(line.rstrip('\n').split(' ')[1])**2
                it_delta += 1
            elif line.startswith('[Iter'):
                Delta[i-1, it_Delta] = float(line.rstrip('\n').split(' ')[-1])**2
                it_Delta += 1
        if it_delta != 30:
            print("Error!")
    delta[i,:] = delta[i,:]/delta[i,0]

print(delta)


figure = plt.figure(figsize=(6, 5))
figure.tight_layout()
plt.xlabel('Iterations')
plt.ylabel('Log$_{10}$ $\delta$')
plt.title("Brain Reconstruction $\delta$ criterion")
ax_delta = []
labels = ["Acc 1", "Acc 2", "Acc 3", "Acc 4"]
linestyle = ["-", ":", "-.", "--"]
print(delta.shape)
for j in range(delta.shape[0]):
    ax_delta.append(plt.plot(np.log10(delta[j,...]),
                          label=labels[j], linestyle=linestyle[j]))
plt.legend()
plt.savefig("./figures/bart_Conv_rate_small_delta.png",dpi=600)

# Figure 3: big delta

print(Delta)
figure = plt.figure(figsize=(6, 5))
figure.tight_layout()
plt.xlabel('Iterations')
plt.ylabel('Log$_{10}$ $\Delta$')
plt.title("Brain Reconstruction $\Delta$ criterion")
ax_Delta = []
labels = ["Acc 2", "Acc 3", "Acc 4"]
linestyle = [":", "-.", "--"]
for j in range(Delta.shape[0]):
    ax_Delta.append(plt.plot(np.log10(Delta[j,...]),
                          label=labels[j], linestyle=linestyle[j]))
plt.legend()
plt.savefig("./figures/bart_Conv_rate_big_delta.png",dpi=600)




# Heart ########################################################################

heart_ims = []
for u in UNDERS[heartdir]:
    heart_ims.append(cfl.readcfl(heartdir + "/pics_u_" + u + "_InScale")[::-1,::-1,...])

figure = plt.figure(figsize=(8, 4))
figure.subplots_adjust(hspace=0, wspace=0.05)
gs = mp.gridspec.GridSpec(
            1, 4)
gs.tight_layout(figure)
figure.patch.set_facecolor('w')
ax = []
for grid in gs:
    ax.append(plt.subplot(grid))
    ax[-1].grid(False)
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
labels = ["55", "33", "22", "11"]
for j in range(len(heart_ims)):
    ax[j].imshow(np.abs(heart_ims[j]), cmap='gray')
    ax[j].text(
      heart_ims[j].shape[0]-25, heart_ims[j].shape[0]-5, labels[j], color="w")
plt.savefig("./figures/bart_Heart.png", dpi=300)


# vim: set ts=4 sw=4 tw=0 et:

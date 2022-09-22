---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "YKqKMhoBO2eF"}

# BART Webinar: The `reconet` Command

This tutorial uses the [BART](http://mrirecon.github.io/bart/) command-line interface (CLI) and presents how to train and apply neural networks for image reconstruction using BART.

The `bart reconet` command implements physics based reconstruction networks solving the inverse SENSE problem.
The SENSE operator is generically implemented. For example, it supports multiple sets of coil sensitivity maps or non-Cartesian trajectories.
Proximal mappings and gradient steps can be used as data-consistency terms such that `reconet` can be used for example to train/infer the Variational Network<sup>1</sup> or MoDL<sup>2</sup>.

The basic structure of the ```bart reconet``` command reads:
```bash
$ bart reconet --network=varnet --train <kspace> <coils> <weights> <reference>
$ bart reconet --network=varnet --apply <kspace> <coils> <weights> <reconstruction>
```

In this notebook, we present a self-contained example, how to train the Variational Network an MoDL on the NYU machine learning dataset<sup>1</sup> available at [mridata.org](http://mridata.org/), i.e. we preprocess the k-space data and estimate coil sensitivity maps using ESPIRiT.

For more information on the implementation details, we refer to out preprint [Deep, Deep Learning with BART](https://arxiv.org/abs/2202.14005).


**Author**: [Moritz Blumenthal](mailto:moritz.blumenthal@med.uni-goettingen.de)

**Presenter**: [Moritz Blumenthal](mailto:moritz.blumenthal@med.uni-goettingen.de)

**Institution**: University Medical Center Göttingen


1: Hammernik, K. et al. (2018), [Learning a variational network for reconstruction of accelerated MRI data](https://doi.org/10.1002/mrm.26977). Magn. Reson. Med., 79: 3055-3071.

2: Aggarwal, H. K. et al.(2019), [MoDL: Model-Based Deep Learning Architecture for Inverse Problems](https://doi.org/10.1109/TMI.2018.2865356). IEEE Trans. Med. Imag., 38(2): 394-405

+++ {"id": "_qPjjJAVISzl"}

## 1 General Remarks and Early Setup

+++ {"id": "_QVfcK0IJgyn"}

This notebook is designed to run on a local system and on Google Colab. It uses the python kernel, however, almost all commands use the `%%bash` cell magic to be executed in a `bash` subshell. If live output of a bash command is desired, we use the exclamation mark `!` to execute single lines.
`bash` environment variables are set in python to be shared across all following cells.

+++ {"id": "5b53kn3GPl-5"}

### 1.1 Google Colab

+++ {"id": "pR0B-a6eNmIS"}

To run BART on Google Colab, this notebook automatically installs dependencies and sets up the GPUs if the environment variable `COLAB=1`is set. If you run this notebook on your local system, you might not want this setup. Please set `COLAB=0`in this case.

For a detailed explanation, see the [How to Run BART on Google Colaboratory](https://github.com/mrirecon/bart-workshop/tree/master/ismrm2021).

This tutorial needs a GPU instance:
- Go to Edit → Notebook Settings
- Choose GPU from Hardware Accelerator drop-down menu

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: -RbGvp5NOAmC
outputId: 65d375b2-a2a6-49ac-ff42-e11da43f922a
---
%env COLAB=1
```

+++ {"id": "DIjoHxmxPAZQ"}

Not all GPUs on Google Colab support CUDA 11, we downgrade CUDA if necessary:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: lZuy2uE_Oeoq
outputId: 6b0cbe92-230c-4857-c8bc-d1735052921e
---
%%bash

[ $COLAB -ne 1 ] && echo "Skipp cell (not on Colab)" && exit 0

# Use CUDA 10.1 when on Tesla K80

# Estimate GPU Type
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

echo "GPU Type:"
echo $GPU_NAME

if [ "Tesla K80" = "$GPU_NAME" ];
then
    echo "GPU type Tesla K80 does not support CUDA 11. Set CUDA to version 10.1."

    # Change default CUDA to version 10.1
    cd /usr/local
    rm cuda
    ln -s cuda-10.1 cuda
else
    echo "Current GPU supports default CUDA-11."
    echo "No further actions are necessary."
fi

echo "GPU Information:"
nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
nvcc --version
```

+++ {"id": "fg7mH1Yb-gc3"}

To have the same experience as if this notebook was checked cloned with git, we clone the BART-Webinars repository and change the current working directory to the `webinar6` directory:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 3JI7sVj8-kjR
outputId: 663ccb1f-73ed-49bb-a472-6135537eb5e7
---
%%bash

[ $COLAB -ne 1 ] && echo "Skipp cell (not on Colab)" && exit 0

[ -d bart-webinars ] && rm -r bart-webinars
git clone https://github.com/mrirecon/bart-webinars.git > /dev/null
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: tvA7f7w0_R5M
outputId: 8d83881c-890d-4fa4-bc46-9e145b4a93e9
---
import os
pwd=os.getcwd()
if os.environ["COLAB"] == "1":
    pwd=pwd+"/bart-webinars/webinar6/demo1_reconet/"

%cd $pwd
```

+++ {"id": "hFPpDTITOFyj"}

### 1.2 Demonstration Mode

+++ {"id": "b0liFDEBJzGD"}

Since training neural takes too long time for demonstration purposes, we skip training in the demonstration mode by setting the environment variable `DEMO=1`.
If you want to reproduce the training, set `DEMO=0`, however, be warned that the storage on Google Colab might not be sufficient to hold and preprocess all training data and that the Colab runtime might disconnect due to the long training time.

In this notebook five networks are trained, each takes about 1h on an NVIDIA A100 GPU but can take much more time on older GPUs. Choose wisely, if you want to retrain them.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: ZPVm19eEJwRv
outputId: 6e6b8d1f-c2e9-43b7-8b12-4f3cd2bfa17a
---
%env DEMO=1
```

+++ {"id": "khVBI7Qatops"}

Please note that the DEMO mode assumes that your current working directory is `webinar6` in the bart-webinars directory. You can change the current working directory persistently using `%cd bart-webinars/webinar6`. 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 8hHSXZf6BY0e
outputId: f256df03-5296-4994-d8c9-d5cc08984958
---
%cd ./
```

+++ {"id": "VWoQW2zAMPhM"}

# Setup BART

+++ {"id": "c9QAsFoSSgG6"}

### 2.1 Install libraries

We install dependencies for BART. Make sure that you have installed the requirements if you run locally:

```{code-cell} ipython3
:id: fQEVVn59O2eP

%%bash

[ $COLAB -ne 1 ] && echo "Skipp cell (not on Colab)" && exit 0

# Install BARTs dependencies
apt-get install -y make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev &> /dev/null

# Install additional dependencies for converting ISMRMRD files
apt-get install -y libismrmrd-dev libboost-all-dev libhdf5-serial-dev &> /dev/null

```

+++ {"id": "fFy511-hSgG8"}

### 2.2 Clone and Compile BART

We clone BART into the current working directory of this notebook and delete any previous installation in this directory.

```{code-cell} ipython3
:id: UG4RDIhQSgG8

%%bash

# Clone Bart
[ -d bart ] && rm -r bart
git clone https://github.com/mrirecon/bart/ bart &> /dev/null
```

```{code-cell} ipython3
:id: YiN-TWaf39OB

%%bash

cd bart

# Define compile options
COMPILE_SPECS=" PARALLEL=1
                CUDA=1
                NON_DETERMINISTIC=1
                ISMRMRD=1
                "

printf "%s\n" $COMPILE_SPECS > Makefiles/Makefile.local


if [ $COLAB -eq "1" ]
then
    # set path to cuda for Colab
    echo "CUDA_BASE=/usr/local/cuda" >> Makefiles/Makefile.local
    echo "CUDA_LIB=lib64" >> Makefiles/Makefile.local
fi


make &> /dev/null
```

+++ {"id": "iCwkW7baO2eQ"}

### 2.3 Add BART to PATH variable

We add the BART directory to the PATH variable and include the python wrapper for reading *.cfl files:

```{code-cell} ipython3
:id: 9WCi0TsVO2eQ

import os
import sys

os.environ['TOOLBOX_PATH']=os.getcwd()+"/bart/"
os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python/")
```

+++ {"id": "Z6UbLX2Ki7Ey"}

Check BART setup:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 3WwBQWEUi7Ey
outputId: a85c957f-1995-43a0-dd11-9b251023e41c
---
%%bash

echo "# The BART used in this notebook:"
which bart
echo "# BART version: "
bart version
```

+++ {"id": "X3IEQ_tLmZue"}

### 2.4 Interactive CFL-Viewer

For visualization of reconstructions etc., we define an interactive viewer for *.cfl files based on ipython widgets.

Usage: plot(["file1", "file2"])

```{code-cell} ipython3
:id: V4rpi89pmOsX

%matplotlib inline

def plot(files, title=None):
  import numpy as np
  from matplotlib import pyplot as plt
  import cfl
  import os

  from ipywidgets import interact, interactive, fixed, interact_manual
  import ipywidgets as widgets

  def update(Range=(0.,1.),Coil=0, Map=0, Slice=0, Batch=0):

    if(title==None):
      wtitle=[]
      for file in files:
        head, tail = os.path.split(file)
        wtitle.append(tail)
    else:
      wtitle=title
    
    data=cfl.readcfl(files[0])
    rat=data.shape[0]/data.shape[1]
    width=16
    
    ncols=len(files)
    nrows=1
    
    rat=rat
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=True, squeeze=False, figsize=(width, width*rat))

    for i in range(len(files)):
 
        data=cfl.readcfl(files[i])
        nshp = [1] * 16

        for k in range(len(data.shape)):
          nshp[k] = data.shape[k]

        idx=[0]*16
        idx[3]=min(Coil,nshp[3]-1)
        idx[4]=min(Map,nshp[4]-1)
        idx[13]=min(Slice,nshp[13]-1)
        idx[15]=min(Batch,nshp[15]-1)

        idx[0]=slice(None, None, -1)
        idx[1]=slice(None, None, None)

        dat=np.abs(data.reshape(nshp)[tuple(idx)])
        if 0 < np.max(dat):
          dat=dat/np.max(dat)

        axs.flatten()[i].imshow(dat,cmap="gray",vmin=Range[0], vmax=Range[1])
        axs.flatten()[i].set_title(wtitle[i])
    for ax in axs.flatten():
      ax.axis("off")
    
    plt.show()

  nshp = [1] * 16
  for file in files:
    data=cfl.readcfl(file)
    for i in range(len(data.shape)):
      nshp[i] = max(data.shape[i], nshp[i])
  
  interact(update,
           Range=widgets.FloatRangeSlider(value=[0.,1.], min=0, max=1.),
           Coil=widgets.IntSlider(min=0, max=nshp[3]-1, step=1, value=0),
           Map=widgets.IntSlider(min=0, max=nshp[4]-1, step=1, value=0),
           Slice=widgets.IntSlider(min=0, max=nshp[13]-1, step=1, value=0),
           Batch=widgets.IntSlider(min=0, max=nshp[15]-1, step=1, value=0))
```

+++ {"id": "F6xUmpowS-Os"}

# The Variational Network and MoDL with `BART reconet`

The `reconet` command implements physics based reconstruction networks solving the inverse SENSE problem.
We have implemented

> Variational Network<sup>1</sup>:
$$
x^{(i)} = x^{(i-1)}  - \lambda \nabla||Ax -b||^2 + Net(x^{(i-1)}, \Theta^{(i)} )
$$
> MoDL<sup>2</sup>:
$$
\begin{align}
z^{(i)} &= Net\left(x^{(i-1)}, \Theta \right)\\
x^{(i)} &= \mathrm{argmin}_x ||Ax -b||^2 + \lambda ||x - z^{(i)}||^2
\end{align}
$$

>Where
+ $A$ - MRI forward operator $\mathcal{PFC}$
    + $\mathcal{P}$ - Sampling pattern
    + $\mathcal{F}$ - Fourier transform
    + $\mathcal{C}$ - Coil sensitivity maps
+ $b$ - measured k-space data
+ $x^{(i)}$ - reconstruction after $i$ iterations
+ $x^{(0)}=A^Hb$ - initialization
+ $\Theta$ - Weights

>1: Hammernik, K. et al. (2018), [Learning a variational network for reconstruction of accelerated MRI data](https://doi.org/10.1002/mrm.26977). Magn. Reson. Med., 79: 3055-3071.

>2: Aggarwal, H. K. et al.(2019), [MoDL: Model-Based Deep Learning Architecture for Inverse Problems](https://doi.org/10.1109/TMI.2018.2865356). IEEE Trans. Med. Imag., 38(2): 394-405

To **train**, **evaluate** or **apply** unrolled networks, we provide the `bart reconet` command. Let us look at the help:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: xtmtrrdsO2ec
outputId: b19f86d3-7c36-42a3-8935-fae693aba346
---
%%bash

bart reconet -h
```

+++ {"id": "ic0r40D7Wja7"}

## 3 Obtain k-Space Data

In this tutorial, we work with the *coronal_pd* data from the NYU machine learning dataset.
The data is available in the ismrmrd format at `mridata.org`.
We tried to match the UUIDs from `mridata.org` with the patient numbers from the Variational Network publication, however, that did not work for all patients.
Here, we restrict to the patients which could be matched.

+++ {"id": "oOPsAtawT6AZ"}

### 3.1 Defining Training and Validation Datasets

+++ {"id": "TL_NgTorUBV2"}

In the following cell, we define environment variables describing which patient should be used for the training dataset and which patient for validation.
Moreover, we create one `W`orking directory and an `A`rchive directory for the downloaded datasets to be reused.

```{code-cell} ipython3
:id: AnEpAP0JUPl8

if "1" == os.environ["DEMO"]:

    # in DEMO mode, we use the same data for (short) training and validation
    os.environ['DAT']="pat_19"            # All data to be processed
    os.environ['TRN_DAT']="pat_19"        # Training dataset
    os.environ['VAL_DAT']="pat_19"        # Validation dataset
    os.environ['WDIR']=os.getcwd()+"/coronal_pd_demo/"
else:

    os.environ['TRN_DAT']="pat_2 pat_3 pat_4 pat_5 pat_6 pat_7 pat_8 pat_9 pat_10 pat_11 pat_12 pat_13 pat_14"
    os.environ['VAL_DAT']="pat_19"
    os.environ['DAT'] = os.environ['TRN_DAT'] + " " + os.environ['VAL_DAT']
    os.environ['WDIR']=os.getcwd()+"/coronal_pd/"

os.environ['ADIR']=os.getcwd()+"/archive/coronal_pd/"

!mkdir -p $ADIR
!mkdir -p $WDIR
```

+++ {"id": "r4vtcp23y_xN"}

### 3.2 Download Required Knee Data from mridata.org

+++ {"id": "O4glwu4gy_xO"}

In the following two cells, we download the datasets defined in the `DAT` variable if they are not already in the archive.

Afterwards, we use the `bart ismrmrd` tool to convert the ismrmrd format to the BART native *.cfl format. Be cautious, BART support for ismrmrd format is very limited and experimental!

Both cells perform the same operation, but they are split to provide live output in the first one.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: c1fXK_z0y_xO
outputId: e85497a2-fa88-4714-c8cc-6b66c2ab2e9d
---
if not(os.path.isfile(os.environ["ADIR"]+"pat_19.hdr")):
    !wget -O $ADIR/pat_19.h5 http://mridata.org/download/993716e9-b5f0-4c30-a059-c003182d0f9c    # download data from mridata.org
    !bart ismrmrd --interleaved-siemens -o0 $ADIR/pat_19.h5 $ADIR/pat_19                         # transform to cfl
```

```{code-cell} ipython3
:id: PGtGxksxy_xP

%%bash

UUIDs=(
1b996273-8d22-4f6e-a062-1e8cb4e9800b
3dafaac6-e40a-45c3-a285-c88944e72dab
662e8e76-9f0c-4f90-8fa0-85d1be9e68db
6b05c23b-a69d-44b9-a176-a0e3181fad57
70983198-5b4e-4081-91f4-0c461f7daebd
724570fe-a822-4faf-9835-8011e318f836
78d1ca02-a565-472d-bdc0-76450ade8cdf
7d1e0fb6-0a43-4548-980c-4f0cdb20c367
80c558dc-908f-4c64-84b0-e8e16da214c0
9270505a-8d77-4e43-ac43-0d9910b81510
993716e9-b5f0-4c30-a059-c003182d0f9c
9d3c581f-0b10-446f-88ef-1b3014e400d6
a0062756-d20b-4dc7-ba1a-76b5367a7c45
a0de6aae-7096-4cc7-bbb4-a4e4d159f680
b03f0bf5-200e-45b5-818e-e4a37142e2f5
c1fb122c-708c-4581-aab0-5b01f382946f
c734e0b0-a0dc-418a-ba68-44b158b00c16
cd255c11-ff09-4dd1-96e1-b7c5ed06f29c
dc05dd93-dd4c-478d-b6b5-79fb80095b73
e1be2ec9-1934-463a-bc92-cfcb5b00031a   
)

NAMEs=(
pat_5
pat_3
pat_18
pat_2
pat_17
pat_6
pat_11
pat_1
pat_14
pat_7
pat_19
pat_4
pat_12
pat_16
pat_13
pat_6
pat_8
pat_9
pat_9
pat_10
)

for d in $DAT
do
    if [[ ! -f $ADIR/${d}.hdr ]]
    then
        #we match the UUID to the patient number
        for i in ${!NAMEs[*]}
        do
            if [ ${NAMEs[i]} = ${d} ]
            then
                wget -O $ADIR/${d}.h5 -o /dev/null http://mridata.org/download/${UUIDs[i]}  # download data from mridata.org
                bart ismrmrd --interleaved-siemens -o0 $ADIR/${d}.h5 $ADIR/${d} >/dev/null  # transform to cfl
            fi
        done
    fi
done
```

+++ {"id": "2mpQBSYeV6-u"}

### 3.3 Extract Slices

+++ {"id": "_Joxvc-dy_xP"}

We extract 20 (or 5 in DEMO mode) slices of each k-space and copy the data to our working directory.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: icZnoNray_xP
outputId: 740d4cbf-c31c-441b-bbb5-c5b44ed939c1
---
%%bash

if [ $DEMO -eq 1 ]
then
    EXTRACT="20 25" #we extract 5 slices
else
    EXTRACT="10 30" #we extract 20 slices
fi

for d in $DAT
do
    bart extract 13 $EXTRACT $ADIR/${d} $WDIR/${d}_ksp_os
done

ls $WDIR
```

+++ {"id": "HBkGadOCjjjn"}

## 4 Estimating Coil Sensitivity Maps and Reference Reconstruction

We use the ESPIRiT method to estimate coil sensitivity maps from the k-space center. 
Further, we define the coil-combined reconstruction of the fully-sampled k-space data as reference



As reference reconstruction, we use the coil combined images.

We visualize the results

+++ {"id": "u73_zR_mXdFm"}

### 4.1 Estimating Coil Sensitivity Maps

ESPIRiT is implemented in BART by the `ecalib`command.
As the `ecalib`-command does not support a batch mode, we loop over the slices of each patient, extract them and run the `ecalib`-command on each of them.
Finally, the estimated coil sensitivity maps are stacked along the slice dimension.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 6eqIahBZX4dY
outputId: df3f7dc3-c3d2-433d-8322-bffff0d97345
---
%%bash

TDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`;
trap 'rm -rf "$TDIR"' EXIT

cd $TDIR

export OMP_NUM_THREADS=1

for pat in $DAT
do
  echo "estimate coils for ${pat}"
  pat=$WDIR/$pat #get absolute path
  
  COLS=""

  for i in $(seq 0 $(($(bart show -d 13 ${pat}_ksp_os)-1))) # loop over all slices of the patient
  do
    bart slice 13 $i ${pat}_ksp_os ksp_$i                   # extract the ith slice
    bart ecalib -a -m1 -r24 ksp_$i col_$i >/dev/null        # estimate coil sensitivity maps
                                                            # -> if "&" is appended, all slices are computes in parallel
    COLS+=" col_$i"                                         # append coil to list for joinint
  done

  # wait for ecalib if called in subprocess
  wait
  
  bart join 13 $COLS ${pat}_col_os                          # join all the slices of one patient
done
```

+++ {"id": "bxZVwgyMYKiV"}

### 4.2 Removing Frequency Oversampling

As we are only interested in the center 320 pixels along the frequency-encoding direction, we remove the frequency oversampling of the k-space data and coil sensitivity maps.

As a byproduct, we obtain the coil images (cim) of the reduced field of view.

```{code-cell} ipython3
:id: fDN7J6gfYr5L

%%bash

#Remove frequency oversampling

TDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`;
trap 'rm -rf "$TDIR"' EXIT
cd $TDIR

for pat in $DAT
do
  pat=$WDIR/$pat

  bart fft -i -u $(bart bitmask 0 1) ${pat}_ksp_os cim_os
  bart resize -c 0 320 cim_os ${pat}_cim
  bart fft -u  $(bart bitmask 0 1) ${pat}_cim ${pat}_ksp
  
  bart resize -c 0 320 ${pat}_col_os ${pat}_col
done
```

+++ {"id": "sPRPXJP6XMK4"}

### 4.3 Coil-combined Reference

We combine the fully-sampled coil images using the `fmac` command to obtain the coil-combined reference and visualize the results:

```{code-cell} ipython3
:id: gFcihG9jVm5n

%%bash

for pat in $DAT
do
    pat=$WDIR/$pat
    bart fmac -s$(bart bitmask 3) -C ${pat}_cim ${pat}_col ${pat}_ref
done
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 531
  referenced_widgets: [8e4cdcf933e347918f12b2b975916627, ee97bee8aefc4bc08290153d071c2de4,
    b6e030c64f6047dfaf365586d60f8e59, 428f27c1bbcc49c2bf1e38d21d0e3d6f, 3ecbca24e68b4245a5ad7d5bbd3190ae,
    7cf1c169118b4d5b88b272cbdb5373aa, c6b60bed22c644f299841ef91e10f0db, 236451a138674ccca6ca07dbfe6952b4,
    9c76e9e34f7c46d1bed1a8a4ad299c36, 478954a366594928aaf50e2b66d7b280, 441947f9a207447b9343c5fda44204a5,
    da3bd7376fa74a1ca703fbc9ec64efd8, 195406fff3e44f828c9c5f38b044732c, dea00908640e424eab664c1c0c92e444,
    8db2e06475834d4ebd052bb4cb4ece19, fd31e5a17ad24fb7bc97f9cf3a79e5f5, 6dcda2cbc3d14e7ca7f220736ecf5d13,
    3325d9beb7074dfab3c833ed72d43fd8, 066689f2546f4df6a9a7802aa2d9c0d8]
id: XcwZWprRzPOv
outputId: bdd42dae-7ba3-49ee-cccc-b03d71d20e7f
---
plot([os.environ["WDIR"]+"pat_19_col", os.environ["WDIR"]+"pat_19_cim", os.environ["WDIR"]+"pat_19_ref"])
```

+++ {"id": "bwvfzutko61h"}

## 5 Undersampling Pattern

We create an undersampling mask with a regular sampling pattern (R=4) and 24 auto calibration lines.
The script stacks ones and zeros, adds the AC-region and performs binary thresholding. (Python code would probably look nicer):

```{code-cell} ipython3
:id: NTJS7Y4rzSZ0

%%bash

AC=24
R=4

READ=320
PHS1=332
PHS_OS=18

TDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`;
trap 'rm -rf "$TDIR"' EXIT
cd $TDIR

bart ones 1 1 one
bart zeros 1 $((R-1)) zero
bart join 0 one zero pat_0
bart repmat 1 $((PHS1/R)) pat_0 pat_1
bart reshape $(bart bitmask 0 1) 1 $PHS1 pat_1 pat_2

bart ones 2 1 18 one
bart join 1 one pat_2 one pat_3

bart ones 2 1 $AC one
bart resize -c 1 $((PHS1+2*PHS_OS)) one ac

bart saxpy 1 ac pat_3 pat_4

bart threshold -B 0.5 pat_4 pat_5

bart repmat 0 $READ pat_5 $WDIR/pat
```

+++ {"id": "vVLgDjLkqA2m"}

### 5.1 CG-SENSE Baseline

A subsampled k-space is created by multiplication of the k-space and the sampling pattern.
To get a feeling how much undersampling effects the reconstruction result, we create a zero-filled reconstruction and use `pics` tool for a CG-SENSE baseline reconstruction (without any regularization). As for ESPIRiT, we loop over all slices and reconstruct them independently.

Finally, we visualize the results.

```{code-cell} ipython3
:id: Ury7ufIu758G

%%bash

TDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`;
trap 'rm -rf "$TDIR"' EXIT
cd $TDIR

for pat in $VAL_DAT
do
    pat=$WDIR/$pat

    bart fmac ${pat}_ksp $WDIR/pat ksp_ss

    CG=""
    
    for i in $(seq 0 $(($(bart show -d 13 ksp_ss)-1)))
    do
      bart slice 13 $i ksp_ss ksp
      bart slice 13 $i ${pat}_col col
      bart pics -p$WDIR/pat -l2 ksp col cg_$i > /dev/null

      CG+=" cg_$i"
    done

    bart join 13 $CG ${pat}_rec_cg

    bart fft -i -u $(bart bitmask 0 1) ksp_ss cim_ss
    bart fmac -s$(bart bitmask 3) -C cim_ss ${pat}_col ${pat}_rec_zf
done
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 447
  referenced_widgets: [db26c6776c1747619b72d9af094c1eda, 58e4036c4e1741a98b2818c756c4e816,
    885df3f3977f440e8823ff941bcdac4b, e544866d18214569a1866d6ce99c8b56, 83e50f1e2f5849b89a060dc087be4fbd,
    73336d44f3f14335baacf2382acd9a60, b572079ebe2f490f9a380922f04bbfbc, 45935fa7dafb4c739fc3f4acec1803c3,
    9f10cae128534c17b14d4df515d6e8a7, e640a6f800b749b2bccd1f5df5a38fcc, b45171a52ba44af89592fb4d48f68bb1,
    4f50e6e57ee6488f8fb3a2efbf26c374, 6eeafdd2db60466588a8d7b854f9da0f, 2f01b94cc3584c14b7b1ca2e6484b8dd,
    5dcffa3b455242d6b292503151ab79f2, 7811e03060d24efab735166bacdc111c, 41fc6e0c3e9e4dd6bf54b6d4e4943e19,
    ba95fa12a84847768782a8f693b53021, 8ae5091761d94cfe93fa50ebe96a8dd0]
id: MXEwg8r_3uAn
outputId: c082d778-9c84-45e6-d4a3-cfe627b25402
---
plot([os.environ["WDIR"]+"pat", os.environ["WDIR"]+"pat_19_rec_zf", os.environ["WDIR"]+"pat_19_rec_cg", os.environ["WDIR"]+"pat_19_ref"])
```

+++ {"id": "ETOkQ08Yqrec"}

## 6 Joining Training and Validation Datasets

We have prepared all data required to train a neural network with the `reconet` command. To pass the data to the `reconet` command, we join all patients of the training dataset to one file.

+++ {"id": "kuhrku2VcUMq"}

### 6.1 Define Variables for the Joined Datasets

For convincing access of the joined datasets, we store their file pathes in environment variables:

```{code-cell} ipython3
:id: DGqX8FlNSgHD

os.environ["TRN_KSP"]=os.environ['WDIR']+"trn_ksp"
os.environ["TRN_COL"]=os.environ['WDIR']+"trn_col"
os.environ["TRN_REF"]=os.environ['WDIR']+"trn_ref"
os.environ["TRN_PAT"]=os.environ['WDIR']+"pat"

os.environ["VAL_KSP"]=os.environ['WDIR']+"val_ksp"
os.environ["VAL_COL"]=os.environ['WDIR']+"val_col"
os.environ["VAL_REF"]=os.environ['WDIR']+"val_ref"
os.environ["VAL_PAT"]=os.environ['WDIR']+"pat"
```

+++ {"id": "u5YXcovWc59o"}

### 6.2 Join and Reshape

The **reconet** command follows the usual conventions of dimensions in BART defined in **src/misc/mri.h**. Independent datasets should be stacked along the **BATCH** dimension (15).

For training, we interpret the slices of the respective patients as independent datasets, i.e. we stack the data of all patients along the **SLICE** dimension and reshape the **SLICE** dimension to the **BATCH** dimension.

```{code-cell} ipython3
:id: eQOUqyy7TAa7

%%bash

TDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`;
trap 'rm -rf "$TDIR"' EXIT
cd $TDIR

DAT_KSP=""
DAT_COL=""
DAT_REF=""

for d in $TRN_DAT
do
DAT_KSP+=" ${WDIR}/${d}_ksp"
DAT_COL+=" ${WDIR}/${d}_col"
DAT_REF+=" ${WDIR}/${d}_ref"
done

bart join 13 $DAT_KSP tmp
bart reshape $(bart bitmask 13 15) 1 $(bart show -d13 tmp) tmp $TRN_KSP

bart join 13 $DAT_COL tmp
bart reshape $(bart bitmask 13 15) 1 $(bart show -d13 tmp) tmp $TRN_COL

bart join 13 $DAT_REF tmp
bart reshape $(bart bitmask 13 15) 1 $(bart show -d13 tmp) tmp $TRN_REF
```

```{code-cell} ipython3
:id: e35sti9YSgHD

%%bash

TDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`;
trap 'rm -rf "$TDIR"' EXIT
cd $TDIR

DAT_KSP=""
DAT_COL=""
DAT_REF=""

for d in $VAL_DAT
do
DAT_KSP+=" ${WDIR}/${d}_ksp"
DAT_COL+=" ${WDIR}/${d}_col"
DAT_REF+=" ${WDIR}/${d}_ref"
done

bart join 13 $DAT_KSP tmp
bart reshape $(bart bitmask 13 15) 1 $(bart show -d13 tmp) tmp $VAL_KSP

bart join 13 $DAT_COL tmp
bart reshape $(bart bitmask 13 15) 1 $(bart show -d13 tmp) tmp $VAL_COL

bart join 13 $DAT_REF tmp
bart reshape $(bart bitmask 13 15) 1 $(bart show -d13 tmp) tmp $VAL_REF
```

+++ {"id": "y_oLMR1VdRDG"}

### 6.3 Inspect the Datasets

We check the dimensions of the constructed training dataset:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: JCBavhY3uCKu
outputId: 6361aca0-4fe0-44eb-89be-25fd197cbfce
---
%%bash

bart show -m $TRN_KSP
bart show -m $TRN_COL
bart show -m $TRN_REF
```

+++ {"id": "NRe9S-qodZXE"}

And visualize them:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 531
  referenced_widgets: [7156df933ca14e1f973eac2abad28350, 6d981ca5072042c38938180ba042fde5,
    8d09383701af4d57a43cdd3c5d2a1811, 7b652e89ccc44f29a4dc9ac804a1ea94, 6c1f81bfe5564ebe8184899403453158,
    aeccfa03ce9d452d8df5412dbf08c7cb, 7d3a02f1849c463aa46e50444a524b4e, 8af22d88b8b040bfa7d64577e8bee369,
    5b0a8dae58cc49b8b2eb7d92a0d3ee48, 39c4a78d883946e597d0ff4ff35eb818, d21d4bcb2a1d47ec96cbab1f598607ec,
    b4c7baeb97e9428292e5ed498868be94, 1b2e9886993f459ab9cea8f23d7a3dde, 980898c6dd8040cda48a373aaee9dbae,
    9ba31ec6cf634bf7ac913058ca710141, 2c75ac4a33b643de8ce10326bea94434, 8de0aae488eb40f2893ec85a8ead4c42,
    758e0ce954fb465f8e0eacb57ae66a3a, 3ad1c560bc4d410483d8fa1f411da9e7]
id: HNl78pIpSgHD
outputId: fa6f89f7-9eed-437d-de08-32d7ef08a710
---
plot([os.environ["TRN_KSP"], os.environ["TRN_COL"], os.environ["TRN_REF"]])
```

+++ {"id": "V4Pwl5vEz_9E"}

## 7 A First Training - Basic Options

Finally, we have everything prepared to run the `reconet` command and train MoDL or the Variational Network.

+++ {"id": "ueSGY8pPeCX0"}

### 7.1 The Variational Network 

Here, we describe each option we provide the `reconet` command to train the Variational network:

````
bart reconet              \
    --network=varnet -I10 \ #We train the Variational Network with 10 iterations
    --train               \ #Select train mode
    -b2                   \ #Batch size is 2
    --gpu                 \ #Obvious :)
    --normalize           \ #Normalize data such that max A^Hk=1
    --train-algo epochs=1 \ #One epoch
    --pattern=$TRN_PAT    \ #Pattern could be estimated from k-space 
    $TRN_KSP $TRN_COL $WGH $TRN_REF
````

The output of `reconet` in the training mode are the weights `$WGH`, which are provided afterwards in the inference mode to obtain a reconstruction.

To train the neural network, the `reconet`-command constructs an operator comparing the output of the network with the reference. Training the corresponds to minimizing the loss with respect to the weights $\theta$:

$$ \theta^* = \mathrm{argmin}_\theta \sum_i L\left(\text{ref}_i, \mathtt{Net}(A^Hb_i, \mathcal{C}_i, \mathcal{P}_i; \theta)\right) $$

The inputs and output od the loss operator are printed in the training log.

We execute the command:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: QsnjSonvSgHE
outputId: 3d7e5c7a-bfd4-4bea-a386-7aa33f41d9e6
---
!bart reconet --network=varnet -I10 --train -b2 --gpu --normalize --train-algo epochs=1 --pattern=$TRN_PAT $TRN_KSP $TRN_COL wgh_varnet $TRN_REF
```

+++ {"id": "skjgx3brSgHE"}

The trained weights are stored as complex floats in a *.cfl file. We have a look at its header:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: TjBiTlECSgHE
outputId: 243ef17a-2f6b-4f62-fb4d-5683abf5d500
---
%%bash

cat wgh_varnet.hdr
```

+++ {"id": "od6E3o6LJ5kR"}

### 7.2 MoDL

Similarly, we can train MoDL, by selecting `--network=modl`. The authors of MoDL propose to train their network in two steps:

````
bart reconet              \
    --network=modl -I1    \ #We initialize MoDL weights by training with one iteration
    --train               \
    -b2                   \
    --gpu --normalize     \
    --train-algo epochs=5 \ #We use some epochs such that the batch normalization can estimate floating mean/variance
    --pattern=$TRN_PAT    \
    $TRN_KSP $TRN_COL wgh_modl_one $TRN_REF

bart reconet
    --network=modl -I5    \
    --lowmem              \ #Reduce memory by checkpointing but double some computations
    --train               \
    -b2                   \
    --gpu --normalize     \
    --train-algo epochs=1 \
    -l wgh_modl_one       \ #We initialize with the weights from the first step
    --pattern=$TRN_PAT    \
    $TRN_KSP $TRN_COL wgh_modl $TRN_REF
````

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: -jhGHGQsSgHE
outputId: b2178d5d-275d-4ab1-a95f-164c0bfd75c7
---
!bart reconet --network=modl -I1 --train -b2 --gpu --normalize --train-algo epochs=5 --pattern=$TRN_PAT $TRN_KSP $TRN_COL wgh_modl_one $TRN_REF
!bart reconet --network=modl -I5 --lowmem --train -b2 --gpu --normalize --train-algo epochs=1 -l wgh_modl_one --pattern=$TRN_PAT $TRN_KSP $TRN_COL wgh_modl $TRN_REF
```

+++ {"id": "lpEArAz4SgHE"}

## 8 Applying the Networks

To apply the networks, we use the `--apply` option instead of the `--train` option:

```{code-cell} ipython3
:id: ds7X2-ePSgHF

%%bash

bart reconet --network=varnet -I10 --apply  --gpu --normalize --pattern=$VAL_PAT $VAL_KSP $VAL_COL wgh_varnet out_varnet > /dev/null
bart reconet --network=modl   -I5  --apply  --gpu --normalize --pattern=$VAL_PAT $VAL_KSP $VAL_COL wgh_modl   out_modl	 > /dev/null
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 531
  referenced_widgets: [5cae548d75aa4170afe1277f75ba9635, 9b62416336fe4121ae369e4ab8740361,
    a9e1ab86bf2c42ff8e2e7e5903b2313f, 021591edc4c146608a09afa6815fe949, 2ea1a3fed27d4cd09894e070d4f2c283,
    ebe6d94de3d145ac9a5f997bd34dd5cb, daf2b5bbb386402ba4211e5723b9c58b, 6521511a6fd94817ae370b5b6faad9a5,
    6202d9e2f6b647e0bacdabc3b1888983, 6c22190ed0bf4db8b3f2303726dd6c9d, 7e18608588414e72a849e7cfcbe41bcd,
    3d24d6356d784b6798d833296df8cd63, df3335c5344e4a93bc212894fbe15550, f36866556aa04ce084d7dbe7fabbdf44,
    247e812bedeb493e84ff4e3ac77cfcf0, 9a1a08381eaa47c0b5c263c2d27ea9ae, d9b3f41db69e432c933d264003336097,
    3c2a5d9e1ff84d268d0da707f1edf211, 73681a06e42d4eccb6b88d1afb63f0f6]
id: uzo49pLaSgHF
outputId: 29062ebf-4269-4040-bc56-8091aa584482
---
plot([os.environ["VAL_REF"], "out_varnet", "out_modl"])
```

+++ {"id": "PBO7BkTySgHF"}

## 9 Extended Training

To obtain better results, we take some more time and train for more epochs.
We can monitor the training by providing the validation dataset which is evaluated after each epoch. We use the `--valid-data` option:

+++ {"id": "Uc6taFNHfjM2"}

### 9.1 The Variational Network

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: DnTJl5qbOVz9
outputId: ce91cd01-a7b2-4420-ea11-be4fef8d72ee
---
%%bash

mkdir -p pretrained
cd pretrained

if [ $DEMO -eq "1" ]
then
    echo "Training skipped in DEMO mode" 
else
    bart reconet --network=varnet  --gpu --normalize                                     \
        --train -T epochs=30,batchgen-shuffle-data                                       \
        --valid-data pattern=${VAL_PAT},kspace=${VAL_KSP},coil=${VAL_COL},ref=${VAL_REF} \
        --pattern=$TRN_PAT $TRN_KSP $TRN_COL wgh_varnet $TRN_REF                         \
        >log_varnet.log
fi


cat log_varnet.log
```

+++ {"id": "3tYAOa6HfsHK"}

### 9.2 MoDL

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: iNvLnsVaSgHF
outputId: 6a8819ec-ccee-4e15-94da-df9def46638c
---
%%bash

mkdir -p pretrained
cd pretrained

if [ $DEMO -eq "1" ]
then
    echo "Training skipped in DEMO mode" 
else
    bart reconet --network=modl -I1                                      \
        --train -T e=50,batchgen-shuffle-data  --gpu --normalize         \
        --lowmem                                                         \
        --valid-data pattern=${VAL_PAT},kspace=${VAL_KSP},coil=${VAL_COL},ref=${VAL_REF} \
        --pattern=$TRN_PAT $TRN_KSP $TRN_COL wgh_modl_one $TRN_REF        \
        >log_modl.log

    bart reconet --network=modl -I5                                      \
        --train -T e=50,batchgen-shuffle-data  --gpu --normalize         \
        --lowmem -lwgh_modl_one                                      \
        --valid-data pattern=${VAL_PAT},kspace=${VAL_KSP},coil=${VAL_COL},ref=${VAL_REF} \
        --pattern=$TRN_PAT $TRN_KSP $TRN_COL wgh_modl $TRN_REF            \
        >>log_modl.log
fi

cat log_modl.log
```

+++ {"id": "GeR-egXxfyoR"}

### 9.3 Visualization

```{code-cell} ipython3
:id: gm7gEbPSSgHF

%%bash

bart reconet --network=varnet -I10 --apply  --gpu --normalize --pattern=$VAL_PAT $VAL_KSP $VAL_COL pretrained/wgh_varnet out_varnet > /dev/null
bart reconet --network=modl   -I5  --apply  --gpu --normalize --pattern=$VAL_PAT $VAL_KSP $VAL_COL pretrained/wgh_modl   out_modl   > /dev/null
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 531
  referenced_widgets: [9beee1e97a4b4f35a842ecbf3dda32d7, 8ffcea02ab144f6e8c14b4d513790baf,
    43927d4001044835bafc4ebf20b2cf02, 76013866acb2453b83a6de9b0e27f1c6, ef87bcaa804b4825aa0f32ca5902da9e,
    c9be09df3ada4028ba482726a009bd9d, b8c786cf277c457a93279cbb963a5e7d, 913ad139530d4c1db976c0551448c600,
    5826a70f19b247f5994acb4f7cb92774, b4df4d4f69c24bd8b2f90be4b1c10889, a78d084afd22472b8b152b6e834b6c9b,
    afff64e877a0400cb47929e9ee99cb40, 36c64a61d58b4b6dba92ed527423df68, d0a74d862452431d844dac1dd7d6e360,
    e8f928ecf1a8440b9a5a765517b4e200, 1a82325484a94f4db004840a77d3eee6, 8629e18cd1d644a1b49c3fbb18054e58,
    527619ceabd845a8b0bf62fabb4582a0, 92e7f5ae1ce649b4b15dba4c29a20cc4]
id: WnwLWvQKSgHG
outputId: f4e3d415-0060-4f7a-db55-a7c228c27c52
---
plot([os.environ["VAL_REF"], "out_varnet", "out_modl"])
```

+++ {"id": "L1OSEFqwSgHG"}

## 10 Evaluation

To obtain a quantitative evaluation of the trained networks, we can pass the `--eval` option to first apply the trained network and compare with a reference.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: JRwv4Y4TSgHG
outputId: 5e13bd38-bed8-457b-90bc-ad3f2b6ef28a
---
%%bash

bart reconet --network=varnet -I10 --eval  --gpu --normalize --pattern=$VAL_PAT $VAL_KSP $VAL_COL pretrained/wgh_varnet $VAL_REF
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: tcPjqH6nSgHG
outputId: 074b3c84-4b86-4092-ddc9-7c3d27f295aa
---
%%bash

bart reconet --network=modl   -I5  --eval  --gpu --normalize --pattern=$VAL_PAT $VAL_KSP $VAL_COL pretrained/wgh_modl $VAL_REF
```

+++ {"id": "gtoC22sPykR8"}

# Extensions: Some Options

The BART implementation of `reconet` implements many options for the training process. A list of all options with a short description is available by the help function.

In this section, we present some of them:

+++ {"id": "ZZoVHq6LzkYy"}

### 11.1 Training Algorithm

The training algorithm can be configured with the `--train-algo` option. We have already used this option to configure the number of epochs and to shuffle the training data after each epoch:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: j-5xMQZqzrf9
outputId: 2f004848-b1c9-4a81-e344-f18795ab72e2
---
%%bash

bart reconet --train-algo h
```

+++ {"id": "5ue7XuxgSgHG"}

### 11.2 Loss configuration

The training loss can be configured using the `--train-loss` option.
Different losses can be combined by setting their weighting. By default, MoDL uses `mse` and the Variational Network uses `mse-magnitude`.

If coil images are provided as reference data, losses acting on magnitude images automatically apply on the rss of the coil images.

Note that not all losses are suitable for images.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 9yAov-dESgHG
outputId: e416b51a-b770-4791-d573-a88802274626
---
%%bash

bart reconet --train-loss h
```

+++ {"id": "uO2RFU_V0l7Y"}

### 11.3 Initial Reconstruction / Data-Consistency

The data-consistency term can be configured using the respective option. By default, MoDL uses `proximal-mapping` and the Variational Network uses `gradient-step`.

By default, both networks are initialized with a zero-filled (i.e. adjoint) reconstruction. We can change that to a CG-SENSE reconstruction `--initial-reco tickhonov`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: HPqINzJB08nT
outputId: 7ecd410e-7b3e-4626-d0f1-22af8d27f856
---
%%bash

bart reconet --data-consistency h
bart reconet --initial-reco h
```

+++ {"id": "romDchs31clE"}

### 11.4 Network Parameter

Further, we can configure the network parameter of the residual block in MoDL or the variational block in the Variational Network.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: MSm9NOk11gek
outputId: 3bb9417b-9f56-4fe8-8ecf-0db38d8be7d4
---
%%bash

bart reconet --varnet-block h
bart reconet --resnet-block h
```

+++ {"id": "N2010mTezaFE"}

### 11.5 Retrain MoDL
We can retrain MoDL using some different options

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: K6Ja-OEBy_xV
outputId: 6d48557b-6ee0-4448-81a5-582ccb028e11
---
%%bash

mkdir -p pretrained
cd pretrained

if [ $DEMO -eq "1" ]
then
    echo "Training skipped in DEMO mode" 
else
    bart reconet --network=modl -I1                                      \
        --train -T e=50,batchgen-shuffle-data  --gpu --normalize         \
        --train -T learning-rate=0.0001                                  \
        --train-loss mse=1.,mad-magnitude=0.01                           \
        --initial-reco tickhonov                                         \
        --lowmem                                                         \
        --valid-data pattern=${VAL_PAT},kspace=${VAL_KSP},coil=${VAL_COL},ref=${VAL_REF} \
        --pattern=$TRN_PAT $TRN_KSP $TRN_COL wgh_modl_one_mad $TRN_REF   \
        >log_modl_mad.log

    bart reconet --network=modl -I5                                      \
        --train -T e=50,batchgen-shuffle-data  --gpu --normalize         \
        --train -T learning-rate=0.0001                                  \
        --train-loss mse=1.,mad-magnitude=0.01                           \
        --initial-reco tickhonov                                         \
        --lowmem -lwgh_modl_one_mad                                      \
        --valid-data pattern=${VAL_PAT},kspace=${VAL_KSP},coil=${VAL_COL},ref=${VAL_REF} \
        --pattern=$TRN_PAT $TRN_KSP $TRN_COL wgh_modl_mad $TRN_REF       \
        >>log_modl_mad.log
fi

cat log_modl_mad.log
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: luapACsvY8LO
outputId: 76193f70-04d8-47ee-c3ab-745d5daa36eb
---
%%bash

bart reconet --network=modl                          -I5 --eval --gpu --normalize --pattern=$VAL_PAT $VAL_KSP $VAL_COL pretrained/wgh_modl $VAL_REF
bart reconet --network=modl --initial-reco tickhonov -I5 --eval --gpu --normalize --pattern=$VAL_PAT $VAL_KSP $VAL_COL pretrained/wgh_modl_mad $VAL_REF
```

```{code-cell} ipython3
:id: 8izRiAZySgHG

%%bash

bart reconet --network=modl -I5  --apply  --gpu --normalize --pattern=$VAL_PAT $VAL_KSP $VAL_COL pretrained/wgh_modl_mad out_modl_mad > /dev/null
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 531
  referenced_widgets: [da3e4a7fdb52475f8ed71a25bbf547b3, 8f692dd6f0af49c2800cfdb7ae5521fd,
    c25cd9171e4d459dba81bcd3a5c9d8b2, 9c32ea8ac1c24485af788f8888148e0b, dd4b5afba8bd4dc8963297aa0a7336a1,
    1fe3bb729cd74cf2b06a3b6031c2119b, 743552fe51e14d6688bb5103a71d5fbb, 64e9a30e3454409b8025f59fe2d099e0,
    b35e3be103be43e9bc5d24a258295085, 698fa20f1d21402baca1ae10297f5aba, 1970da636a2c4e4f92a9082150dee7a4,
    10aaeaa67d7c48d79ca9b72f1094d49d, 34686fa7b10943859752e46e894fc10d, 3744bb538ba34937b5b203797473f8c1,
    9840efe5ee434853b0d0492bfe8c1b41, 94fe492874754012bc4cc4acab68a6ee, 8a1c7207a2334b65a899f4899080794e,
    585407985a384f23bb20c5b306b47905, d68afe363433498f95cd82b5ae4ea545]
id: xitJr0sKSgHH
outputId: 5b4f72cc-7cb4-4b0d-dafb-36cd476717a2
---
plot([os.environ["VAL_REF"], "out_modl_mad", "out_modl"])
```

+++ {"id": "CXLOAsLAyzXc"}

# Extensions: Non-Cartesian Trajectories

Finally, we present a reconstruction using non-Cartesian trajectories.
The trajectory can be passed with the `--trajectory` option to the reconet command.

In this example, we simulate non-Cartesian k-space data based on the fully-sampled coil images:

+++ {"id": "-7Qbv7qp5SKr"}

## 12.1 Simulate Non-Cartesian Dataset and Generate Training Data

```{code-cell} ipython3
:id: THsxL1jQuXYO

os.environ["TRJ"]=os.environ['WDIR']+"trj"

os.environ["TRN_KSP"]=os.environ['WDIR']+"trn_ksp_rad"
os.environ["VAL_KSP"]=os.environ['WDIR']+"val_ksp_rad"
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: X3mGBO2luXYO
outputId: a9b87859-7a76-4c68-c087-d138c7382519
---
%%bash

bart traj -x 368 -y 30 -r $TRJ

for pat in $DAT
do 
    bart nufft $TRJ $WDIR/${pat}_cim $WDIR/${pat}_ksp_rad
done
```

```{code-cell} ipython3
:id: NhuvhriZuXYO

%%bash

TDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`;
trap 'rm -rf "$TDIR"' EXIT
cd $TDIR

DAT_KSP=""

for d in $TRN_DAT
do
DAT_KSP+=" ${WDIR}/${d}_ksp_rad"
done

bart join 13 $DAT_KSP tmp
bart reshape $(bart bitmask 13 15) 1 $(bart show -d13 tmp) tmp $TRN_KSP

DAT_KSP=""

for d in $VAL_DAT
do
DAT_KSP+=" ${WDIR}/${d}_ksp_rad"
done

bart join 13 $DAT_KSP tmp
bart reshape $(bart bitmask 13 15) 1 $(bart show -d13 tmp) tmp $VAL_KSP
```

+++ {"id": "8MhjvkM2uXYO"}

## 12.2 Training MoDL

To train MoDL, we use again the `--initial-reco tickhonov` option to initialize the network with the CG-SENSE reconstruction. We further increase the number of CG iterations and fix the regularization of the initial reconstruction to be not trained. Thus, the data is normalized such that the maximum magnitude of the CG-SENSE reconstruction is one.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: DA-9_OgguXYP
outputId: e4927d18-48ea-4f87-da30-9d2553227b27
---
%%bash

if [ $DEMO -eq "1" ]
then
    echo Skipped
else

    bart reconet --network=modl -I1                                      \
        --train -T e=50,batchgen-shuffle-data,r=0.01  --gpu --normalize  \
        --initial-reco tickhonov,fix-lambda=0.1,max-cg-iter=30           \
        --data-consistency max-cg-iter=30                                \
        --valid-data trajectory=${TRJ},kspace=${VAL_KSP},coil=${VAL_COL},ref=${VAL_REF} \
        --trajectory=${TRJ} $TRN_KSP $TRN_COL pretrained/wgh_modl_one_rad $TRN_REF      \
        >pretrained/log_modl_rad.log

    bart reconet --network=modl -I5                                      \
        --train -T e=50,batchgen-shuffle-data  --gpu --normalize         \
        --lowmem -lpretrained/wgh_modl_one_rad                           \
        --initial-reco tickhonov,fix-lambda=0.1,max-cg-iter=30           \
        --data-consistency max-cg-iter=30                                \
        --valid-data trajectory=${TRJ},kspace=${VAL_KSP},coil=${VAL_COL},ref=${VAL_REF} \
        --trajectory=$TRJ $TRN_KSP $TRN_COL pretrained/wgh_modl_rad $TRN_REF            \
        >>pretrained/log_modl_rad.log

fi
cat pretrained/log_modl_rad.log
```

+++ {"id": "MlV5yVJDuXYP"}

## 12.3 L1-Wavelet Baseline Reconstruction

For reference, we perform a l1-wavelet baseline reconstruction.

```{code-cell} ipython3
:id: qM_Cw2wTuXYP

%%bash

TDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`;
trap 'rm -rf "$TDIR"' EXIT
cd $TDIR

for pat in $VAL_DAT
do
    pat=$WDIR/$pat

    CG=""
    
    for i in $(seq 0 $(($(bart show -d 13 ${pat}_ksp_rad)-1)))
    do
      bart slice 13 $i ${pat}_ksp_rad ksp
      bart slice 13 $i ${pat}_col col
      bart pics -r0.0002 -l1 -e -i100 -t$TRJ ksp col cg_$i > /dev/null

      CG+=" cg_$i"
    done

    bart join 13 $CG ${pat}_rec_rad_l1
done
```

+++ {"id": "pQ9RVN1UuXYP"}

## 12.4 Apply Non-Cartesian MoDL

Finally, we apply the non-Cartesian MoDL and visualize the reconstruction

```{code-cell} ipython3
:id: b_jl_ikVuXYQ

%%bash

bart reconet --network=modl -I5 --apply --gpu --normalize                                     \
    --initial-reco tickhonov,fix-lambda=0.1,max-cg-iter=30 --data-consistency max-cg-iter=30  \
    --trajectory=$TRJ $VAL_KSP $VAL_COL pretrained/wgh_modl_rad out_noncart > /dev/null
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 531
  referenced_widgets: [3cdf935b442d42009d970cdabf4138fd, 50e2a72e383644e68200b0e505e65beb,
    aab0aa74db7b4bb6a162c64276a0eee9, 4c2384773cce4b92b29b9af16a664c0c, f2c1e0b25cf04002bb1b726b5eb2c142,
    650fb1fae2524b3fa8fe74f715bcef89, 0876ce949764491183d36e2f082667f1, 778019efd43f4314ac8d311b21322ab4,
    1de2552496d845a6a10d00091fc23c84, 303c81e89b914600a3c1515a9f994c7f, 81bb589866644921be2bf32f1299d210,
    d734e4d036e54400b12fd9811ee032e9, 58f9e78928d4430f9ae7663c4cae6fee, 2ee7f44aff174b99b6ebc38e0aabf510,
    b68bd1fd8acb4499a487675bd214c66a, c9e89dc6103447089d4df38487faac33, b87de7760e1042e9a54fea93929439fb,
    d2637c13ad804323944bd6f28dcff3b8, 8cb23871f7f24d3095ff651407882b20]
id: 1hxbOHMLuXYQ
outputId: 35a92c1c-9baf-42ee-a359-c078c4487a05
---
plot([os.environ["WDIR"]+"pat_19_ref", os.environ["WDIR"]+"pat_19_rec_rad_l1", "out_noncart"])
```

+++ {"id": "OoynXO1G578C"}

# Thank you for your attention

```{code-cell} ipython3
:id: SVqsXZL79DPp


```

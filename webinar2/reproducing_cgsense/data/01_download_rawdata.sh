#!/bin/bash

[[ -f rawdata_brain_radial_96proj_12ch.h5 ]] || curl --output rawdata_brain_radial_96proj_12ch.h5 'https://zenodo.org/record/3975887/files/rawdata_brain_radial_96proj_12ch.h5'
[[ -f rawdata_heart_radial_55proj_34ch.h5 ]] || curl --output rawdata_heart_radial_55proj_34ch.h5 'https://zenodo.org/record/3975887/files/rawdata_heart_radial_55proj_34ch.h5'
[[ -f rawdata_spiral_ETH.h5 ]] || curl --output rawdata_spiral_ETH.h5 'https://zenodo.org/record/3975887/files/rawdata_spiral_ETH.h5'
[[ -f cardiac_radial_KI.h5 ]] || curl --output cardiac_radial_KI.h5 'https://zenodo.org/record/3975887/files/cardiac_radial_KI.h5'

#wget 'https://zenodo.org/record/3975887/files/rawdata_brain_radial_96proj_12ch.h5'
#wget 'https://zenodo.org/record/3975887/files/rawdata_heart_radial_55proj_34ch.h5'
#wget 'https://zenodo.org/record/3975887/files/rawdata_spiral_ETH.h5'
#wget 'https://zenodo.org/record/3975887/files/cardiac_radial_KI.h5'

md5sum -c zenodo.md5

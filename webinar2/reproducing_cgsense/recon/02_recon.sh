#!/bin/bash
set -euo pipefail
set -B

#export TOOLBOX_PATH=../../bart_for_rrsg

export PATH=$TOOLBOX_PATH:$PATH

if [ ! -e $TOOLBOX_PATH/bart ] ; then
        echo "\$TOOLBOX_PATH is not set correctly!" >&2
        exit 1
fi


# check for CUDA support, and only use GPU if it is supported
CUDA=$(bart version -V | grep CUDA | awk -F "=" '{print $2}')

if [[ $CUDA -eq 1 ]]; then
	GPU='-g'
else
	GPU=''
fi

echo $GPU

declare -A UNDERS
UNDERS[brain_radial_96proj_12ch_cfl]="96 48 32 24"
UNDERS[heart_radial_55proj_34ch_cfl]="55 33 22 11"

for dir in "brain_radial_96proj_12ch_cfl" "heart_radial_55proj_34ch_cfl"
do
	UNDER=${UNDERS[$dir]}
	LARGEST=$(echo "${UNDER%% *}")
	echo -n "${dir} "
	for und in ${UNDER}
	do
		echo ${und}
		udir=${dir}/${und}
		bart fmac ${udir}/rl_${und} ${udir}/ksp_${und} ${udir}/ksprl_${und}
		bart nufft -a ${udir}/traj_${und} ${udir}/ksprl_${und} ${udir}/nufft_${und}

		bart scale 1e4 ${udir}/ksp_${und} ${udir}/ksp_${und}_scaled
#		bart scale 1e0 ${udir}/ksp_${und} ${udir}/ksp_${und}_scaled

		PICS_OPTS="-d5 -w1. -S -p ${udir}/sqrt_rl_${und} -RQ:0.5 $GPU -t ${udir}/traj_${und} ${udir}/ksp_${und}_scaled ${dir}/Coils"

		if [[ $und -ne ${LARGEST} ]]; then
			ADDITIONAL_OPTS="-T ${dir}/${LARGEST}/pics_u_${LARGEST}"
		else
			ADDITIONAL_OPTS=""
		fi

		bart pics -i30 $ADDITIONAL_OPTS $PICS_OPTS ${udir}/pics_u_${und} | tee ${dir}/log_pics_u_${und}
                bart fmac ${dir}/InScale_inv ${udir}/pics_u_${und} ${dir}/pics_u_${und}_InScale

		# single iteration
		bart pics -i1 $ADDITIONAL_OPTS $PICS_OPTS ${udir}/pics_u_${und}_it_1
                bart fmac ${dir}/InScale_inv ${udir}/pics_u_${und}_it_1 ${dir}/pics_u_${und}_it_1_InScale
	done
done


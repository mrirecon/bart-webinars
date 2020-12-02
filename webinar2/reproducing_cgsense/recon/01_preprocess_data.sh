#!/bin/bash
set -euo pipefail
set -B

# Download raw data and convert to .cfl
(
	cd ../data
	./01_download_rawdata.sh
	./02_hdf5_to_cfl.py
)

#export TOOLBOX_PATH=../../bart_for_rrsg

export PATH=$TOOLBOX_PATH:$PATH

if [ ! -e $TOOLBOX_PATH/bart ] ; then
        echo "\$TOOLBOX_PATH is not set correctly!" >&2
        exit 1
fi


datadir="../data"

declare -A UNDERS
UNDERS[brain_radial_96proj_12ch_cfl]="96 48 32 24"
UNDERS[heart_radial_55proj_34ch_cfl]="55 33 22 11"

for dir in ${!UNDERS[@]}
do
	mkdir -p ${dir}
	
	#InScale
	bart squeeze ${datadir}/${dir}/InScale ${dir}/InScale
	bart invert ${dir}/InScale ${dir}/InScale_inv
	
	# Coils
	bart transpose 0 3 ${datadir}/${dir}/Coils ${dir}/tmp_Coils
	bart resize -c  1 $(bart show -d 0 ${dir}/InScale) 2 $(bart show -d 1 ${dir}/InScale) ${dir}/tmp_Coils ${dir}/tmp_Coils2
	bart reshape 7 $(bart show -d 0 ${dir}/InScale) $(bart show -d 1 ${dir}/InScale) 1 ${dir}/tmp_Coils2 ${dir}/Coils
	
	
	# Undersample ksp and traj, generate RamLak
	fullksp=${datadir}/${dir}/rawdata
	fulltraj=${datadir}/${dir}/trajectory
	UNDER=${UNDERS[$dir]}
	for und in ${UNDER}
	do
		udir=${dir}/${und}
		mkdir -p ${udir}

		# Undersampling is different for brain and heart
		# Brain needs every 2nd, every 3rd and every 4th spoke
		# Heart just takes elements from the beginning

		if [[ "$dir" == "brain_radial_96proj_12ch_cfl" ]]; then

		        full=$(bart show -d2 ${fullksp})
#		        elems=()
#		        for ((i = 0 ; i < $und ; i++)); do
#		                elem=$(echo "scale=0; $i*$full/$und" | bc -l)
#		                elems+=($elem)
#		        done
#		        echo ${elems[@]}
#		
#		        bart pick 2 ${elems[@]} ${fullksp} ${udir}/ksp_${und}
#		        bart pick 2 ${elems[@]} ${fulltraj} ${udir}/traj_${und}
			acc=$(echo "scale=0;$full / $und" | bc -l)
			echo $acc
		        ../undersample.py 2 $acc ${fullksp} ${udir}/ksp_${und}
		        ../undersample.py 2 $acc ${fulltraj} ${udir}/traj_${und}
		elif [[ "$dir" == "heart_radial_55proj_34ch_cfl" ]]; then

			bart extract 2 0 $und ${fullksp} ${udir}/ksp_${und}
			bart extract 2 0 $und ${fulltraj} ${udir}/traj_${und}
		else

			echo "Not implemented!"
			exit 1
		fi
	
		#Generate RamLak

		bart rss 1 ${udir}/traj_${und} ${udir}/tmp_rl_${und}
		# Scale RamLak between to between 0 and 1
	        imsize=$(bart show -d 1 ${dir}/Coils)
		rlscale=$(echo "2/${imsize}" | bc -l)
		echo $rlscale
		bart scale $rlscale ${udir}/tmp_rl_${und} ${udir}/rl_${und}
		bart spow 0.5 ${udir}/rl_${und} ${udir}/sqrt_rl_${und}
	done
done

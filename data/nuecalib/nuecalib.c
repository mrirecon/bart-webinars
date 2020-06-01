/* 
 * Copyright 2020 Jon Tamir.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2020		Jon Tamir <jtamir@utexas.edu>
 */

#include <complex.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/mmio.h" // used for input-output operations, like load_cfl
#include "misc/mri.h" // contains defs for dimensions, like COIL_DIM
#include "misc/debug.h" // used for printing debug info
#include "misc/opts.h" // for command-line argument parsing

#include "num/multind.h" // multi-dim array functions
#include "num/flpmath.h" // used for CFL_SIZE def
#include "num/fft.h" // fast Fourier transform functions
#include "num/init.h" // CPU/GPU intialization routines

#include "calib/calib.h" // library containing routines for ESPIRiT calibration

#include "linops/linop.h" // library containing linear operator routines

#include "iter/lsqr.h" // library containing least squares solver

#include "noncart/nufft.h" // library containing non-uniform FFT


/*
 * usage_str will display when the command is typed without arguments or with incorrect arguments
 * and should be indicate the basic input/output use.
 */
static const char usage_str[] = "<kspace> <traj> <sensitivities>";

/*
 * help_str will display with the -h argument, and should explain the tool usage in more detail
 */
static const char help_str[] =
		"Estimate coil sensitivities from non-Cartesian data using\n"
		"ESPIRiT calibration.\n"
		"This tool is for demonstration purposes only and does not\n"
		"have the full functionality of `nufft` or `ecalib`.";


/*
 * Each tool should have a main function with the following signature,
 * and with the naming convention main_<toolname>
 */
int main_nuecalib(int argc, char* argv[])
{
	/*
	 * variables that should be accessible as command-line arguments
	 * are defined here. This includes config structs
	 */

	unsigned int maps = 1;

	struct nufft_conf_s nuconf = nufft_conf_defaults;
	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;
	struct ecalib_conf econf = ecalib_defaults;

	const struct opt_s opts[] = {

		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_UINT('m', &maps, "maps", "Number of maps to compute."),
		OPT_FLOAT('t', &econf.threshold, "threshold", "This determines the size of the null-space."),
		OPT_FLOAT('c', &econf.crop, "crop_value", "Crop the sensitivities if the eigenvalue is smaller than {crop_value}."),
	};

	/*
	 * The cmdline function creates and parses the commandline arguments.
	 * Typically, the only values that should be changed are the third and fourth
	 * arguments that indicate the minimum and maximum number of required arguments
	 */
	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	/*
	 * num_init() initializes the program for multi-CPU. There is also a version for GPU,
	 * called num_init_gpu()
	 */
	num_init();

	/*
	 * load the input data with memory-mapped files. Based on our usage string,
	 * kspace data will be in argv[1] and trajectory data in argv[2].
	 * To load data, we allocate an array to hold its dimensions. In the
	 * call to load_cfl, both the data and the array will be populated
	 */

	// load trajectory
	long traj_dims[DIMS];
	complex float* traj = load_cfl(argv[1], DIMS, traj_dims);

	// load kspace data
	long ksp_dims[DIMS];
	complex float* ksp = load_cfl(argv[2], DIMS, ksp_dims);

	/*
	 * Step 1: perform iterative NUFFT.
	 * We will use the noncart library. The sub-steps are the following:
	 * 1. Estimate image dimensions based on trajectory
	 * 2. Construct an NUFFT linear operator
	 * 3. Perform least squares fit to iteratively compute the inverse NUFFT 
	 */

	
	// estimate multi-coil image dimensions based on the trajectory
	long cimg_dims[DIMS] = { 0 };
	estimate_im_dims(DIMS, FFT_FLAGS, cimg_dims, traj_dims, traj);
	md_copy_dims(DIMS - 3, cimg_dims + 3, ksp_dims + 3);
	debug_print_dims(DP_INFO, DIMS, cimg_dims);

	/*
	 * debug_printf can be used to print messages to standard out and standard error/
	 * The amount of output shown depends on the debug level
	 */
	debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", cimg_dims[0], cimg_dims[1], cimg_dims[2]);


	// allocate and clear memory for coil image.
	// We specify the data dimensions and the data type (complex float)
	complex float* cimg = md_alloc(DIMS, cimg_dims, CFL_SIZE);
	md_clear(DIMS, cimg_dims, cimg, CFL_SIZE);

	// create NUFFT linear operator (used by lsqr)
	const struct linop_s* nufft_op = nufft_create(DIMS, ksp_dims, cimg_dims, traj_dims, traj, NULL, nuconf);

	// solve for NUFFT inverse using LSQR and store in cimg	
	lsqr(DIMS, &(struct lsqr_conf){ 0., false }, iter_conjgrad, CAST_UP(&cgconf), nufft_op, NULL, cimg_dims, cimg, ksp_dims, ksp, NULL);

	// free linear operator memory after we finish solving
	linop_free(nufft_op);

	/* 
	 * Step 2: estimate Sensitivity maps using ESPIRiT.
	 * We will use the calib library. The sub-steps are the following:
	 * 1. Take the FFT of the coil-image, cimg, and store in ksp_grid
	 * 2. Extract a calibration region from the center of k-space
	 * 3. Run the ESPIRiT algorithm
	 */

	// allocate memory and compute Cartesian kspace using centered unitary FFT.
	complex float* ksp_grid = md_alloc(DIMS, cimg_dims, CFL_SIZE);
	fftuc(DIMS, cimg_dims, FFT_FLAGS, ksp_grid, cimg);

	// free the memory for the coil image since we are done with it
	md_free(cimg);

	// extract the calibration data from ksp_grid and store in cal_data
	long cal_dims[DIMS];
	long calsize[3] = { 24, 24, 24 };
	complex float* cal_data = extract_calib(cal_dims, calsize, cimg_dims, ksp_grid, false);
	md_free(ksp_grid);

	// compute the dimensions for the sensitivity maps
	long sens_dims[DIMS];
	md_copy_dims(DIMS, sens_dims, cimg_dims);
	sens_dims[MAPS_DIM] = maps;

	// we use the create_cfl function to create a memory-mapped file.
	// We use arv[3] for the name of the file
	complex float* sens = create_cfl(argv[3], DIMS, sens_dims);

	// compute some things used for ESPIRiT. See paper for full details
	long emaps_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, emaps_dims, sens_dims);
	complex float* emaps = md_alloc(DIMS, emaps_dims, CFL_SIZE);

	long channels = cal_dims[COIL_DIM];
	for (int i = 0; i < 3; i++)
		if (1 == cimg_dims[i])
			econf.kdims[i] = 1;
	unsigned int K = econf.kdims[0] * econf.kdims[1] * econf.kdims[2] * channels;
	float svals[K];

	// run ESPIRiT calibration
	calib(&econf, sens_dims, sens, emaps, K, svals, cal_dims, cal_data); 

	/*
	 * Step 3: final cleanup.
	 * Free remaining memory and unmap files
	 */

	md_free(cal_data);
	md_free(emaps);

	unmap_cfl(DIMS, ksp_dims, ksp);
	unmap_cfl(DIMS, traj_dims, traj);
	unmap_cfl(DIMS, sens_dims, sens);


	printf("Done.\n");
	return 0;
}



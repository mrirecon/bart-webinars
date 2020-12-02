

# compare to nufft, fft, and ecalib

tests/test-nuecalib: traj scale phantom nufft fft ecalib nuecalib nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -G -x128 -y128 traj2.ra						;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra						;\
	$(TOOLDIR)/phantom -k -s 4 -t traj.ra ksp_noncart.ra				;\
	$(TOOLDIR)/nufft -i traj.ra ksp_noncart.ra cimg.ra				;\
	$(TOOLDIR)/fft -u 3 cimg.ra ksp_grid.ra						;\
	$(TOOLDIR)/ecalib -m2 -c .5 -t .002 ksp_grid.ra sens.ra				;\
	$(TOOLDIR)/nuecalib -m2 -c .5 -t .002 traj.ra ksp_noncart.ra sens2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 sens.ra sens2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-nuecalib 

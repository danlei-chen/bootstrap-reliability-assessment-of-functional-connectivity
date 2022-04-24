#!/bin/tcsh
setenv DATA /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled
setenv SCRIPTPATH /autofs/cluster/iaslab/users/danlei/FSMAP/scripts
setenv IMAGE /autofs/cluster/iaslab/users/jtheriault/singularity_images/jtnipyutil/jtnipyutil-2019-01-03-4cecb89cb1d9.simg
setenv PROJNAME emoAvd_CSUSnegneu_trial
# setenv PROJNAME painAvd_CSUS1snegneu_trial
setenv SINGULARITY /usr/bin/singularity
setenv OUTPUT /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/$PROJNAME/connectivity_analyses/level2_iteration_resutls_scrambled_data/subject_SC_mask_corrcoef_FisherZ
setenv PORTION $1 #1-8

mkdir -p /scratch/$USER/$PROJNAME/wrkdir/
mkdir -p /scratch/$USER/$PROJNAME/output/
mkdir -p /scratch/$USER/$PROJNAME/data
mkdir -p $OUTPUT

rsync /autofs/cluster/iaslab/users/danlei/roi/WholeBrain/mni_icbm152_gm_tal_nlin_asym_09a_thresh10.nii /scratch/$USER/$PROJNAME/wrkdir/

# rsync -ra $SCRIPTPATH/model/search_region.nii /scratch/$USER/$PROJNAME/wrkdir/
rsync $SCRIPTPATH/model/2.build_null_distribution/* /scratch/$USER/$PROJNAME/wrkdir/
chmod a+rwx /scratch/$USER/$PROJNAME/wrkdir/*_startup.sh
cd /scratch/$USER

$SINGULARITY exec\
--bind "$DATA/$PROJNAME/connectivity_analyses:/scratch/data" \
--bind "/scratch/$USER/$PROJNAME/output:/scratch/output" \
--bind "/scratch/$USER/$PROJNAME/wrkdir:/scratch/wrkdir" \
$IMAGE\
/scratch/wrkdir/2.build_null_distribution_startup.sh

# rsync -r /scratch/$USER/$PROJNAME/output/* /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/$PROJNAME/connectivity_analyses/

# cd /autofs/cluster/iaslab/users/danlei/FSMAP/scripts/model/
# chmod -R a+rwx *

rm -r /scratch/$USER/$PROJNAME/
exit


# scp -r * /autofs/cluster/iaslab/users/danlei/test/
# scp dz609@door.nmr.mgh.harvard.edu:/autofs/cluster/iaslab/users/danlei/test/sub-014_emo3_combined_trial_copes_neg_smoothed3mm_masked.nii.gz /Users/chendanlei/Desktop/

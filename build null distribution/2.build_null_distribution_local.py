# !/usr/bin/env python
# coding: utf-8

# python3 /Volumes/GoogleDrive/My\ Drive/U01/AffPainTask_connectivity/analysis/connectivity/level2/level2_iteration/level2_iteration_scrambled_data/2.build_null_distribution.py

#schedule the script to run at a certain time
# import datetime
# import time
# run_now=0
# while run_now == 0:
#     currentDT = datetime.datetime.now()
#     print (str(currentDT))
#     if int(currentDT.hour) > 3 and int(currentDT.minute) > 30:
#         run_now = 1
#         print("start to run the script at "+(str(currentDT)))
#     time.sleep(1200)

import nibabel as nib
import numpy as np
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import os, glob
from nilearn.glm import threshold_stats_img
from scipy import stats
import shutil
from nilearn.image import resample_img
import matplotlib.pyplot as plt

#### INPUT ####
proj = 'emoAvd_CSUSnegneu_trial'
# proj = 'painAvd_CSUS1snegneu_trial'
iteration_range = range(0, 30)
warping = 'SC_connectivity_results_scrambled'
# file_suffix = '_negneu_4mm_subject_SC_mask_corrcoef_FisherZ'
flameo_run_mode = 'ols'
seed_suffix = 'subject_SC_mask_corrcoef_FisherZ'
smoothing = '4mm'
negneu = 'negneu'

# #set tstats
# stats_map = 'tstat1'
# df=19
# pval_thresh = 0.05
# tscore_list = np.arange(1, 6, 0.01)
# pval_list = np.array([])
# for tt in tscore_list:
#     pval_list = np.append(pval_list, stats.t.sf(np.abs(tt), df-1))
# fdr_thresh = tscore_list[np.where(pval_list == min(pval_list, key=lambda x:abs(x-pval_thresh)))][0]
# fdr_thresh = float(str(fdr_thresh)[0:4])
# print('df: '+str(df))
# print('tscore threshold: '+str(fdr_thresh))

####################################################################
####################################################################
#group level 
####################################################################
####################################################################
#set up new base directory to run level copes
data_dir_base = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/connectivity/level2/results/level2_iteration_results/'+seed_suffix+'/'
scrambeld_data_dir_base = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/connectivity/level2/results/level2_iteration_resutls_scrambled_data/'+seed_suffix+'/'

data_dir = data_dir_base+proj+'_iter0-999_'+smoothing+'/'+proj+'_iter0-999_'+smoothing+'_overlap_iter_perc.nii.gz'
data_img = nib.load(data_dir)
data = data_img.get_fdata()
mask = '/Volumes/GoogleDrive/My Drive/fMRI/atlas/MNI/mni_icbm152_nlin_asym_09a_nifti/mni_icbm152_nlin_asym_09a/mni_icbm152_gm_tal_nlin_asym_09a_thresh10.nii'
def apply_resampling(template_dir, target_dir, interpolation):
    mask_data = nib.load(template_dir)
    cope_data = nib.load(target_dir)
    cope_data_resampled = resample_img(cope_data,
        target_affine=mask_data.affine,
        target_shape=mask_data.shape[0:3],
        interpolation=interpolation)
    return cope_data_resampled
mask_data_resampled = apply_resampling(data_dir, mask, 'nearest').get_fdata()
data_masked = data[mask_data_resampled!=0]

# read and combine scrambered data 
scrambled_perc_dir = glob.glob(scrambeld_data_dir_base+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'/scrambled_*/'+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_perc'+'_scrambled_*.nii.gz')
scrambled_perc_dir.sort()
scramble_num = [int(i.split('/')[-2].split('_')[-1]) for i in scrambled_perc_dir]
scramble_num.sort()

combined_output_array = np.empty((len(mask_data_resampled[mask_data_resampled!=0]),len(scramble_num)))
for scramble_n,scramble_iter in enumerate(scramble_num):
    scrambled_perc_iter_img = nib.load(glob.glob(scrambeld_data_dir_base+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'/scrambled_'+str(scramble_iter)+'/'+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_perc'+'_scrambled_'+str(scramble_iter)+'.nii.gz')[0])
    scrambled_perc_iter_data = scrambled_perc_iter_img.get_fdata()
    # scrambled_perc_iter_vox_data = scrambled_perc_iter_img.dataobj[x[0],y[0],z[0]]
    
    combined_output_array[:,scramble_n] = scrambled_perc_iter_data[mask_data_resampled!=0]

# draw p-value in brain
pValue_data = mask_data_resampled.copy()
pValue_data[:] = 0

v=14010
v=0
v=1197247
v=1673
v=92
v=1197396

#see the percentile of a number in an array
v_pValue = stats.percentileofscore(combined_output_array[v,:], data_masked[v])/100
# np.percentile(combined_output_array[v,:], 95) #get score of 95 percentile of an array
pValue_data[np.where(mask_data_resampled!=0)[0][v], np.where(mask_data_resampled!=0)[1][v], np.where(mask_data_resampled!=0)[2][v]] = v_pValue
plt.hist(combined_output_array[v,:],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.axvline(x=data_masked[v],color='r')
plt.axvline(x=np.percentile(combined_output_array[v,:], 95),color='g',linestyle='dashed')
plt.title('voxel: '+str(v))
plt.show()
    
    
# for v in range(combined_output_array.shape[0]):
#     #v=14010
    
#     #see the percentile of a number in an array
#     v_pValue = stats.percentileofscore(combined_output_array[v,:], data_masked[v])/100
#     # np.percentile(combined_output_array[v,:], 95) #get score of 95 percentile of an array
#     pValue_data[np.where(mask_data_resampled!=0)[0][v], np.where(mask_data_resampled!=0)[1][v], np.where(mask_data_resampled!=0)[2][v]] = v_pValue
#     # plt.hist(combined_output_array[v,:],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#     # plt.axvline(x=data_masked[v],color='r')
#     # plt.axvline(x=np.percentile(combined_output_array[v,:], 95),color='g',linestyle='dashed')
#     # plt.show()
    
#     if v%1000==0:
#         print(str(v)+' in '+str(combined_output_array.shape[0]))
#         pValue_img = nib.Nifti1Image(pValue_data, data_img.affine, data_img.header)
#         # nib.save(pValue_img, '/Users/chendanlei/Desktop/x.nii.gz')
#         # nib.save(pValue_img, scrambeld_data_dir_base+data_dir.split('/')[-1].split('.nii.gz')[0]+'_pValue.nii.gz')
#         nib.save(pValue_img, data_dir.split('.nii.gz')[0]+'_pValue.nii.gz')

    
    


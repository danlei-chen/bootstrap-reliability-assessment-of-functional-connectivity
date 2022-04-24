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
# from nilearn.glm import threshold_stats_img
from scipy import stats
from nilearn.image import resample_img
import matplotlib.pyplot as plt
from datetime import date

#### INPUT ####
proj = os.environ['PROJNAME']
# proj = 'painAvd_CSUS1snegneu_trial'
# iteration_range = range(0, 30)
# warping = 'SC_connectivity_results_scrambled'
# file_suffix = '_negneu_3mm_subject_SC_mask_corrcoef_FisherZ'
# flameo_run_mode = 'ols'
seed_suffix = 'subject_SC_mask_corrcoef_FisherZ'
smoothing = '3mm'
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
path_base = '/scratch/data/'
data_dir_base = path_base+'level2_iteration_results/'+seed_suffix+'/'
scrambeld_data_dir_base = path_base+'/level2_iteration_resutls_scrambled_data/'+seed_suffix+'/'

data_dir = data_dir_base+proj+'_iter0-999_'+smoothing+'/'+proj+'_iter0-999_'+smoothing+'_overlap_iter_perc.nii.gz'
data_img = nib.load(data_dir)
data = data_img.get_fdata()
mask = '/scratch/wrkdir/mni_icbm152_gm_tal_nlin_asym_09a_thresh10.nii'
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
# scrambled_perc_dir = glob.glob(scrambeld_data_dir_base+'scrambled_*/'+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_perc'+'_scrambled_*.nii.gz')
scrambled_perc_dir = glob.glob(scrambeld_data_dir_base+'scrambled_*/'+proj+'_iter*_'+smoothing+'_overlap_iter_perc'+'_scrambled_*.nii.gz')
scrambled_perc_dir.sort()
scramble_num = [int(i.split('/')[-2].split('_')[-1]) for i in scrambled_perc_dir]
scramble_num.sort()
print('Total number of scrambles: '+str(len(scramble_num)))

combined_output_array = np.empty((len(mask_data_resampled[mask_data_resampled!=0]),len(scramble_num)))
for scramble_n,scramble_iter in enumerate(scramble_num):
    # scrambled_perc_iter_img = nib.load(glob.glob(scrambeld_data_dir_base+'/scrambled_'+str(scramble_iter)+'/'+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_perc'+'_scrambled_'+str(scramble_iter)+'.nii.gz')[0])
    scrambled_perc_iter_img = nib.load(glob.glob(scrambeld_data_dir_base+'/scrambled_'+str(scramble_iter)+'/'+proj+'_iter*_'+smoothing+'_overlap_iter_perc'+'_scrambled_'+str(scramble_iter)+'.nii.gz')[0])
    scrambled_perc_iter_data = scrambled_perc_iter_img.get_fdata()
    # scrambled_perc_iter_vox_data = scrambled_perc_iter_img.dataobj[x[0],y[0],z[0]]

    combined_output_array[:,scramble_n] = scrambled_perc_iter_data[mask_data_resampled!=0]
# print(combined_output_array)

total_portion = 8
vox_range_start = combined_output_array.shape[0]/total_portion*(int(os.environ['PORTION'])-1)
vox_range_end = combined_output_array.shape[0]/total_portion*int(os.environ['PORTION'])
print(range(int(vox_range_start), int(vox_range_end)))
# draw p-value in brain
name_date = date.today().strftime("%b-%d-%Y")
pValue_data = mask_data_resampled.copy()
pValue_data[:] = 0
for v in range(int(vox_range_start), int(vox_range_end)):
    #v=14010
    
    #see the percentile of a number in an array
    v_pValue = stats.percentileofscore(combined_output_array[v,:], data_masked[v])/100
    # np.percentile(combined_output_array[v,:], 95) #get score of 95 percentile of an array
    pValue_data[np.where(mask_data_resampled!=0)[0][v], np.where(mask_data_resampled!=0)[1][v], np.where(mask_data_resampled!=0)[2][v]] = v_pValue
    # plt.hist(combined_output_array[v,:],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # plt.axvline(x=data_masked[v],color='r')
    # plt.axvline(x=np.percentile(combined_output_array[v,:], 95),color='g',linestyle='dashed')
    # plt.show()
    
    if v%1000==0:
        print(str(v)+' in '+str(combined_output_array.shape[0]))
        pValue_img = nib.Nifti1Image(pValue_data, data_img.affine, data_img.header)
        # nib.save(pValue_img, '/Users/chendanlei/Desktop/x.nii.gz')
        # nib.save(pValue_img, scrambeld_data_dir_base+data_dir.split('/')[-1].split('.nii.gz')[0]+'_pValue.nii.gz')
        save_name = data_dir.split('.nii.gz')[0]+'_pValue_'+name_date+'_'+os.environ['PORTION']+'.nii.gz'
        nib.save(pValue_img, save_name)
    
    


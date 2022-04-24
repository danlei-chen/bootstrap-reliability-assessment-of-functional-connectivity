# !/usr/bin/env python
# coding: utf-8

# rm -r /Users/chendanlei/Desktop/U01/fc_results/*/SC_connectivity_results_scrambled/*
# python3 /Volumes/GoogleDrive/My\ Drive/U01/AffPainTask_connectivity/analysis/connectivity/level2/scripts/level2_iteration/level2_iteration_scrambled_data/0.scramble_subj_conn_map+1.flameo_conn_2groups_iterations_overlap_scrambled_data.py

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
import shutil

#### INPUT ####
#4mm stopped here: scramble_num_range = list(range(907, 915))
########################################
############ scramble data ############
########################################
# proj_list = ['emoAvd_CSUSnegneu_trial','painAvd_CSUS1snegneu_trial']
proj_list = ['emoAvd_CSUSnegneu_trial']
# scramble_num_range = range(725, 754)
scramble_num_range = [729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750]
# warping = 'SC_connectivity_results'
file_suffix = '_negneu_3mm_subject_SC_mask_corrcoef_FisherZ'
########################################
############### level 2 ################
########################################
# proj_list = ['emoAvd_CSUSnegneu_trial','painAvd_CSUS1snegneu_trial']
# proj_list = ['emoAvd_CSUSnegneu_trial']
# proj_list = ['painAvd_CSUS1snegneu_trial']
# scramble_num_range = range(770, 780)
# scramble_num_range = range(200, 225)
iteration_range = range(0, 25)
# warping = 'SC_connectivity_results_scrambled'
# file_suffix = '_negneu_4mm_subject_SC_mask_corrcoef_FisherZ'
flameo_run_mode = 'ols'
seed_suffix = 'subject_SC_mask_corrcoef_FisherZ'
smoothing = '3mm'
negneu = 'negneu'

#set stats
stats_map = 'tstat1'
df=19
pval_thresh = 0.05
tscore_list = np.arange(1, 6, 0.01)
pval_list = np.array([])
for tt in tscore_list:
    pval_list = np.append(pval_list, stats.t.sf(np.abs(tt), df-1))
fdr_thresh = tscore_list[np.where(pval_list == min(pval_list, key=lambda x:abs(x-pval_thresh)))][0]
fdr_thresh = float(str(fdr_thresh)[0:4])
print('df: '+str(df))
print('tscore threshold: '+str(fdr_thresh))

fixed_fx = pe.Workflow(name='fixedfx')

copemerge = pe.MapNode(
    interface=fsl.Merge(dimension='t'),
    iterfield=['in_files'],
    name='copemerge')

if flameo_run_mode == 'flame1':
    varcopemerge = pe.MapNode(
        interface=fsl.Merge(dimension='t'),
        iterfield=['in_files'],
        name='varcopemerge')

level2model = pe.Node(interface=fsl.L2Model(), name='l2model')

if flameo_run_mode == 'ols':
    flameo = pe.MapNode(
        interface=fsl.FLAMEO(run_mode='ols'),
        name='flameo',
        iterfield=['cope_file'])
    fixed_fx.connect([
        (copemerge, flameo, [('merged_file', 'cope_file')]),
        (level2model, flameo, [('design_mat', 'design_file'),
                               ('design_con', 't_con_file'),
                               ('design_grp', 'cov_split_file')]),
    ])
elif flameo_run_mode == 'flame1':
    flameo = pe.MapNode(
        interface=fsl.FLAMEO(run_mode='flame1'),
        name='flameo',
        iterfield=['cope_file', 'var_cope_file'])
    fixed_fx.connect([
        (copemerge, flameo, [('merged_file', 'cope_file')]),
        (varcopemerge, flameo, [('merged_file', 'var_cope_file')]),
        (level2model, flameo, [('design_mat', 'design_file'),
                               ('design_con', 't_con_file'),
                               ('design_grp', 'cov_split_file')]),
    ])

def run_flameo(seed_suffix, work_dir, path_base, save_dir, subjs):
    fixed_fx.base_dir = save_dir
    if not os.path.exists(fixed_fx.base_dir):
        os.makedirs(fixed_fx.base_dir)
    print('working on seed:', seed_suffix)
    # get files
    cope_list = glob.glob(path_base)
    cope_list = [i for i in cope_list if i.split('/')[-2] in subjs]
    cope_list.sort()
    print(str(len(cope_list))+' files')
    # cope_list=[i for i in cope_list if 'sub-037' not in i and 'sub-039' not in i and 'sub-053' not in i and 'sub-060' not in i and 'sub-066' not in i and 'sub-119' not in i]
    # build mask.
    # print('mask')
    mask = np.mean(np.array([nib.load(f).get_fdata() for f in cope_list]), axis=0)
    # print('mask finished')
    mask[np.isnan(mask)] = 0
    mask[mask != 0] = 1
    nib.save(nib.Nifti1Image(mask, nib.load(cope_list[0]).affine, nib.load(cope_list[0]).header), os.path.join(work_dir, 'mask_'+seed_suffix.split(".")[0]+'.nii.gz'))
    mask_file = os.path.join(work_dir, 'mask_'+seed_suffix.split(".")[0]+'.nii.gz')
    fixed_fx.inputs.flameo.mask_file = mask_file
    fixed_fx.inputs.copemerge.in_files = cope_list
    fixed_fx.inputs.l2model.num_copes = len(cope_list)
    try:
        fixed_fx.run()
    except:
        pass

for proj in proj_list:
    print('************************************************')
    print('************************************************')  
    print(proj)  
    print('************************************************')
    print('************************************************')

    #read iterative subject groups
    subj_group_all = np.load('/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/connectivity/level2/scripts/level2_iteration/'+proj+'_subj_2groups_iter100000.npy')
    subj_group_all = subj_group_all[:,iteration_range,:]

    for scramble_iter in scramble_num_range:
        print('************************************************')
        print('scrambled iteration: '+str(scramble_iter))
        print('************************************************')

        ########################################
        ############ scramble data ############
        ########################################
        print('...scrambling data...')
        
        data_dir_base = '/Users/chendanlei/Desktop/U01/fc_results/'
        # data_dir_base = '/Volumes/GoogleDrive/My\ Drive/U01/AffPainTask_connectivity/analysis/connectivity/level1/results/'
        save_dir_base = data_dir_base+proj+'/SC_connectivity_results_scrambled/'+'scrambled_'+str(scramble_iter)
        # save_dir_base = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/connectivity/level1/results/'+proj+'/SC_connectivity_results_scrambled/'+'scrambled_'+str(scramble_iter)
        if not os.path.isdir(save_dir_base):
            os.mkdir(save_dir_base)

        all_subj = [i.split('/')[-1] for i in glob.glob(data_dir_base+proj+'/SC_connectivity_results/sub*')]
        all_subj.sort()

        for subj in all_subj:
            file = glob.glob(data_dir_base+proj+'/SC_connectivity_results/'+subj+'/'+subj+'_*'+file_suffix+'.nii.gz')[0]
            file_img = nib.load(file)
            file_data = file_img.get_fdata()
            # print(np.nansum(file_data))

            voxels_to_scramble = np.where((file_data!=0.0) & (~np.isnan(file_data)))
            voxel_scrambled_order = np.array(range(voxels_to_scramble[0].shape[0]))
            
            np.random.shuffle(voxel_scrambled_order)
            voxels_scrambled = (voxels_to_scramble[0][voxel_scrambled_order], voxels_to_scramble[1][voxel_scrambled_order], voxels_to_scramble[2][voxel_scrambled_order])
            file_data[voxels_to_scramble] = file_data[voxels_scrambled]
            # print(np.nansum(file_data))
        
            save_dir_subj = save_dir_base+'/'+subj+'/'
            if not os.path.isdir(save_dir_subj):
                os.mkdir(save_dir_subj)
            scrambled_img = nib.Nifti1Image(file_data, file_img.affine, file_img.header)
            nib.save(scrambled_img, save_dir_subj+file.split('/')[-1].split('.nii.gz')[0]+'_scrambled_'+str(scramble_iter)+'.nii.gz')

        ########################################
        ############### level 2 ################
        ########################################
        print('...running group analysis...')

        group_level = True
        # data_dir_base = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/connectivity/level1/results/'
        data_dir_base = '/Users/chendanlei/Desktop/U01/fc_results/'
        work_dir_base = '/Users/chendanlei/Desktop/U01/fc_results/workdir/'
        save_dir_base = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/connectivity/level2/results/level2_iteration_resutls_scrambled_data/'+seed_suffix+'/'

        result_dir = save_dir_base+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'/scrambled_'+str(scramble_iter)+'/'
        result_sum_map = result_dir+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_sum'+'_scrambled_'+str(scramble_iter)+'.nii.gz'
        result_perc_map = result_dir+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_perc'+'_scrambled_'+str(scramble_iter)+'.nii.gz'
    
        for iter_n, iter_name in enumerate(iteration_range):
            print(iter_n)  
        
            subj_group1 = subj_group_all[0][iter_n]
            subj_group2 = subj_group_all[1][iter_n]

            temp_work_dir = '2groups_iter_temp_'+proj+'_iter'+str(min(iteration_range))+'-'+str(max(iteration_range))+'_'+smoothing+'_scrambled_'+str(scramble_iter)+'/'
            work_dir = work_dir_base+temp_work_dir
            if not os.path.isdir(work_dir):
                os.mkdir(work_dir)
            path_base = data_dir_base+proj+'/SC_connectivity_results_scrambled/scrambled_'+str(scramble_iter)+'/sub*'+'/*_'+negneu+'_'+smoothing+'_'+seed_suffix+'_scrambled_'+str(scramble_iter)+'.nii.gz'
                        
            #the code below will repeatedly save to the same local directory, we'll upload them to the cluster later
            #group1
            save_dir = os.path.join(work_dir, seed_suffix+'_group1_'+smoothing)
            run_flameo(seed_suffix, work_dir, path_base, save_dir, subj_group1)
            # np.savetxt(os.path.join(work_dir, seed_suffix+'_group1_'+smoothing, 'subj_group1.txt'), subj_group1, fmt='%s')
            #group2
            save_dir = os.path.join(work_dir, seed_suffix+'_group2_'+smoothing)
            run_flameo(seed_suffix, work_dir, path_base, save_dir, subj_group2)
            # np.savetxt(os.path.join(work_dir, seed_suffix+'_group2_'+smoothing, 'subj_group2.txt'), subj_group2, fmt='%s')
            
            group1_map_name = work_dir+'/'+seed_suffix+'_group1_'+smoothing+'/fixedfx/flameo/mapflow/_flameo0/stats/tstat1.nii.gz'
            group2_map_name = work_dir+'/'+seed_suffix+'_group2_'+smoothing+'/fixedfx/flameo/mapflow/_flameo0/stats/tstat1.nii.gz'
            group1_map = nib.load(group1_map_name).get_fdata()
            group2_map = nib.load(group2_map_name).get_fdata()
    
            group1_map[group1_map<fdr_thresh] = 0
            group2_map[group2_map<fdr_thresh] = 0
    
            group1_map[group1_map!=0] = 1
            group2_map[group2_map!=0] = 1
    
            overlap_iter = group1_map+group2_map
            overlap_iter[overlap_iter!=2] = 0
            overlap_iter[overlap_iter!=0] = 1
    
            if os.path.exists(result_sum_map):
                overlap_sum = nib.load(result_sum_map).get_fdata()
                overlap_sum = overlap_sum + overlap_iter
            else:
                os.makedirs(result_dir)
                overlap_sum = overlap_iter.copy()
    
            overlap_all_img = nib.Nifti1Image(overlap_sum, nib.load(group1_map_name).affine, nib.load(group1_map_name).header)
            nib.save(overlap_all_img, result_sum_map)
            
            if not os.path.isdir(result_dir+'iterations/'):
                os.mkdir(result_dir+'iterations/')
            overlap_iter_img_name = result_dir+'iterations/'+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter'+str(iter_name)+'_scrambled_'+str(scramble_iter)+'.nii.gz'
            overlap_iter_img = nib.Nifti1Image(overlap_iter, nib.load(group1_map_name).affine, nib.load(group1_map_name).header)
            nib.save(overlap_iter_img, overlap_iter_img_name)
    
        overlap_perc = overlap_sum/np.max(overlap_sum)
        overlap_perc_img = nib.Nifti1Image(overlap_perc, nib.load(group1_map_name).affine, nib.load(group1_map_name).header)
        nib.save(overlap_perc_img, result_perc_map)
        
        shutil.rmtree(work_dir)
        # shutil.rmtree(data_dir_base+proj+'/SC_connectivity_results_scrambled/scrambled_'+str(scramble_iter))



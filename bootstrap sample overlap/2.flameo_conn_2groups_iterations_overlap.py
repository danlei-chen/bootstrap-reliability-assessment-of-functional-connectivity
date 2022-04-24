# !/usr/bin/env python
# coding: utf-8

# python /Volumes/GoogleDrive/My\ Drive/U01/AffPainTask_connectivity/analysis/connectivity/level2/level2_iteration/2.flameo_conn_2groups_iterations_overlap.py

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

#### INPUT ####
# proj_list = ['painAvd_CSUS1snegneu_trial','emoAvd_CSUSnegneu_trial']
proj_list = ['painAvd_CSUS1snegneu_trial']
warping = 'SC_connectivity_results'
# seed_dir_suffix = ['subject_SC_mask_corrcoef_FisherZ', 'group_emo_SC_mask_corrcoef_FisherZ', 'group_pain_SC_mask_corrcoef_FisherZ']
seed_suffix = 'subject_SC_mask_corrcoef_FisherZ'
# seed_suffix = 'Vision_group_SC_mask_corrcoef_FisherZ_right'
# seed_suffix = 'Somato_group_SC_mask_corrcoef_FisherZ_right'
# seed_suffix = 'subject_SC_mask_corrcoef_FisherZ_right'
# seed_suffix = 'subject_SC_mask_corrcoef_FisherZ_left'
flameo_run_mode = 'ols'
iteration_range = range(0, 1000)
smoothing = '3mm'
negneu = 'negneu'

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

####################################################################
####################################################################
#group level 
####################################################################
####################################################################
#set up new base directory to run level copes
group_level = True
data_dir_base = '/Users/chendanlei/Desktop/U01/fc_results/'
work_dir_base = '/Users/chendanlei/Desktop/U01/fc_results/workdir/'
save_dir_base = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/connectivity/level2_iteration_results/'+seed_suffix+'/'
# data_dir_base = '/scratch/workdir/'
# work_dir_base = '/scratch/workdir/'

for proj in proj_list:
    print('************************************************')
    print('************************************************')  
    print(proj)  
    print('************************************************')
    print('************************************************')

    #read iterative subject groups
    subj_group_all = np.load('/Volumes/GoogleDrive/My Drive//U01/AffPainTask_connectivity/analysis/connectivity/level2/level2_iteration/'+proj+'_subj_2groups_iter100000.npy')
    subj_group_all = subj_group_all[:,iteration_range,:]
 
    # result_sum_map = save_dir_base+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_sum.nii.gz'
    # result_perc_map = save_dir_base+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_perc.nii.gz'
    result_dir = save_dir_base+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'/'
    result_sum_map = result_dir+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_sum.nii.gz'
    result_perc_map = result_dir+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter_perc.nii.gz'

    for iter_n, iter_name in enumerate(iteration_range):
        print('************************************************')
        print(iter_n)  
        print(iter_name)  
        print('************************************************')

        subj_group1 = subj_group_all[0][iter_n]
        subj_group2 = subj_group_all[1][iter_n]

        temp_work_dir = '2groups_iter_temp_'+proj+'_iter'+str(min(iteration_range))+'-'+str(max(iteration_range))+'_'+smoothing
        work_dir = work_dir_base+temp_work_dir
        if not os.path.isdir(work_dir):
            os.mkdir(work_dir)
        path_base = data_dir_base+proj+'/'+warping+'/sub*'+'/*_'+negneu+'_'+smoothing+'_'+seed_suffix+'.nii.gz'
                    
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
        overlap_iter_img_name = result_dir+'iterations/'+proj+'_iter'+str(np.min(iteration_range))+'-'+str(np.max(iteration_range))+'_'+smoothing+'_overlap_iter'+str(iter_name)+'.nii.gz'
        overlap_iter_img = nib.Nifti1Image(overlap_iter, nib.load(group1_map_name).affine, nib.load(group1_map_name).header)
        nib.save(overlap_iter_img, overlap_iter_img_name)

    overlap_perc = overlap_sum/np.max(overlap_sum)
    overlap_perc_img = nib.Nifti1Image(overlap_perc, nib.load(group1_map_name).affine, nib.load(group1_map_name).header)
    nib.save(overlap_perc_img, result_perc_map)




            # #move some of the results to the cluster for storage
            # #group1
            # move_from = work_dir+'/'+seed_suffix+'_group1_'+smoothing
            # move_to = 'dz609@door.nmr.mgh.harvard.edu:/autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/'+proj+'/connectivity_analyses/group_analyses_iteration_'+smoothing+'/iteration_'+str(iter_name)+'/group1/'
            # # files_to_move = glob.glob(move_from + '/fixedfx/flameo/mapflow/_flameo0/stats/*')
            # files_to_move = glob.glob(move_from + '/fixedfx/flameo/mapflow/_flameo0/stats/tstat1.nii.gz')
            # files_to_move = files_to_move + glob.glob(move_from+'/subj_group1.txt')
            # # files_to_move = [i for i in files_to_move if 'mean_random_effects_var1.nii.gz' not in i]
            # # files_to_move = [i for i in files_to_move if 'res4d.nii.gz' not in i]
            # # files_to_move = [i for i in files_to_move if 'weights1.nii.gz' not in i]
            # # files_to_move = [i for i in files_to_move if 'tdof_t1.nii.gz' not in i]
            # # files_to_move = [i for i in files_to_move if 'zflame1' not in i]
            # # files_to_move = [i for i in files_to_move if 'mask.nii.gz' not in i]
            # for file in files_to_move:
            #     os.system("rsync -a --ignore-existing "+ file + " " + move_to)
            # #group2
            # move_from = work_dir+'/'+seed_suffix+'_group2_'+smoothing
            # move_to = 'dz609@door.nmr.mgh.harvard.edu:/autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/'+proj+'/connectivity_analyses/group_analyses_iteration_'+smoothing+'/iteration_'+str(iter_name)+'/group2/'
            # # files_to_move = glob.glob(move_from + '/fixedfx/flameo/mapflow/_flameo0/stats/*')
            # files_to_move = glob.glob(move_from + '/fixedfx/flameo/mapflow/_flameo0/stats/tstat1.nii.gz')
            # files_to_move = files_to_move + glob.glob(move_from+'/subj_group2.txt')
            # # files_to_move = [i for i in files_to_move if 'mean_random_effects_var1.nii.gz' not in i]
            # # files_to_move = [i for i in files_to_move if 'res4d.nii.gz' not in i]
            # # files_to_move = [i for i in files_to_move if 'weights1.nii.gz' not in i]
            # # files_to_move = [i for i in files_to_move if 'tdof_t1.nii.gz' not in i]
            # # files_to_move = [i for i in files_to_move if 'zflame1' not in i]
            # # files_to_move = [i for i in files_to_move if 'mask.nii.gz' not in i]
            # for file in files_to_move:
            #     os.system("rsync -a --ignore-existing "+ file + " " + move_to)            
            
# foreach number ( `seq 0 1000` )
#     echo $number
#     mkdir -p /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/emoAvd_CSUSnegneu_trial/connectivity_analyses/group_analyses_iteration_3mm/iteration_${number}/group1
#     mkdir -p /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/emoAvd_CSUSnegneu_trial/connectivity_analyses/group_analyses_iteration_3mm/iteration_${number}/group2
#     mkdir -p /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/painAvd_CSUS1snegneu_trial/connectivity_analyses/group_analyses_iteration_3mm/iteration_${number}/group1
#     mkdir -p /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/painAvd_CSUS1snegneu_trial/connectivity_analyses/group_analyses_iteration_3mm/iteration_${number}/group2
# end
            
# foreach number ( `seq 0 1000` )
#     set f=/autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/emoAvd_CSUSnegneu_trial/connectivity_analyses/group_analyses_iteration/iteration_${number}/group1/tstat1.nii.gz
#     if ( ! -f $f ) then
#         echo 'emo_iter'${number}
#     endif

#     set f=/autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/painAvd_CSUS1snegneu_trial/connectivity_analyses/group_analyses_iteration/iteration_${number}/group1/tstat1.nii.gz
#     if ( ! -f $f ) then
#         echo 'pain_iter'${number}
#     endif
# end




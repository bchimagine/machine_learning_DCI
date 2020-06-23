#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:03:37 2019

@author: davood
"""






from __future__ import division

import numpy as np
import os
#from numpy import dot
#from dipy.core.geometry import sphere2cart
#from dipy.core.geometry import vec2vec_rotmat
#from dipy.reconst.utils import dki_design_matrix
#from scipy.special import jn
#from dipy.data import get_fnames
from dipy.core.gradients import gradient_table
#import scipy.optimize as opt
#import pybobyqa
from dipy.data import get_sphere
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import SimpleITK as sitk
#from sklearn import linear_model
#from sklearn.linear_model import OrthogonalMatchingPursuit
#import spams
#import dipy.core.sphere as dipysphere
from tqdm import tqdm
import crl_aux
#import crl_dti
import crl_dci
#from scipy.stats import f
#from importlib import reload
import h5py
#import dipy.reconst.sfm as sfm
import dipy.data as dpd
#from dipy.viz import window, actor
#import dipy.direction.peaks as dpp
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import auto_response
#from dipy.reconst.forecast import ForecastModel
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
#from dipy.denoise.nlmeans import nlmeans
#from dipy.denoise.noise_estimate import estimate_sigma
#from dipy.denoise.pca_noise_estimate import pca_noise_estimate
#from dipy.denoise.localpca import localpca
#from dipy.denoise.localpca import mppca
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
#from dipy.viz import has_fury
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
#from dipy.viz import colormap
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
import nibabel as nib
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.data import small_sphere
from dipy.direction import ProbabilisticDirectionGetter
#from dipy.direction import ClosestPeakDirectionGetter
from dipy.io.streamline import load_trk
import dipy.tracking.life as life
import dk_aux
import pandas as pd
from os import listdir
from os.path import isdir, join
from dipy.data.fetcher import get_two_hcp842_bundles
from dipy.data.fetcher import (fetch_target_tractogram_hcp,
                               fetch_bundle_atlas_hcp842,
                               get_bundle_atlas_hcp842,
                               get_target_tractogram_hcp)
#import numpy as np
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from fury import actor, window
#from dipy.io.stateful_tractogram import Space, StatefulTractogram
#from dipy.io.streamline import load_trk, save_trk
from dipy.io.utils import create_tractogram_header
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
#from scipy.ndimage.morphology import binary_dilation
from dipy.tracking.utils import length
#from dipy.tracking.metrics import downsample
#from dipy.tracking.distances import approx_polygon_track
import dipy.core.optimize as dipy_opt
from dipy.tracking.streamline import cluster_confidence
from dipy.direction.peaks import peak_directions
import matplotlib.patches as patches
import tensorflow as tf
import dk_model



PI= np.pi
R2D= 180/np.pi







dhcp_dir= '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/dHCP/data/'
res_dir=  '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/dHCP/results/'



subj_info= pd.read_csv( dhcp_dir + 'participants.tsv', delimiter= '\t')

n_anat_sess= np.zeros(len(subj_info))
n_dmri_sess= np.zeros(len(subj_info))
ages_anat=   np.zeros((len(subj_info),2))
ages_dmri=   np.zeros((len(subj_info),2))
i= -1

for subj in subj_info['participant_id']:
    
    anat_dir= dhcp_dir + 'dhcp_anat_pipeline/sub-' + subj
    anat_sess = [d for d in listdir(anat_dir) if isdir(join(anat_dir, d))]
    
    dmri_dir= dhcp_dir + 'dhcp_dmri_pipeline/sub-' + subj
    dmri_sess = [d for d in listdir(dmri_dir) if isdir(join(dmri_dir, d))]
    
    sess_tsv= anat_dir + '/sub-' + subj + '_sessions.tsv'
    sess_info= pd.read_csv( sess_tsv , delimiter= '\t')
    ages= np.array(sess_info['scan_age'])
    
    i+= 1
    n_anat_sess[i]= len(anat_sess)
    n_dmri_sess[i]= len(dmri_sess)
    ages_anat[i,:len(ages)]= ages
    
    i_temp= -1
    for j in range(len(sess_info)):
        for k in range(len(dmri_sess)):
            if sess_info.loc[j, 'session_id']== int(dmri_sess[k].split('ses-')[1]):
                i_temp+= 1
                ages_dmri[i,i_temp]= sess_info.loc[j, 'scan_age']

temp= ages_dmri[ages_dmri>0]
bins= np.arange( np.floor(temp.min()), np.ceil(temp.max()) )
plt.figure(), plt.hist(temp, bins= bins, weights=np.ones(len(temp)) / len(temp)*100 );




###############################################################################

###   Create JHU atlas
'''
jhu_dir= '/media/nerossd2/atlases/JHU_neonate_SS/'

bundles_2_read= ['00_cc']
pr_lb= 50

t2_jhu= sitk.ReadImage( jhu_dir + 'JHU_neonate_nonlinear_t2ss.img')
t2_jhu.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
t2_jhu_np= sitk.GetArrayFromImage(t2_jhu)
t2_jhu_np= np.transpose(t2_jhu_np, [2,1,0])

lb_jhu_np= np.zeros(t2_jhu_np.shape)
bundle_count= -1
for bundle_2_read in bundles_2_read:
    bundle_count+= 1
    temp_img= sitk.ReadImage( jhu_dir + '/bundles/' + bundles_2_read[bundle_count] + '.img')
    temp_img_np= sitk.GetArrayFromImage(temp_img)
    temp_img_np= np.transpose(temp_img_np, [2,1,0])
    lb_jhu_np[temp_img_np>pr_lb]= bundle_count+1

lb_jhu_np= np.transpose(lb_jhu_np, [2,1,0])
lb_jhu= sitk.GetImageFromArray(lb_jhu_np)
lb_jhu.SetDirection(t2_jhu.GetDirection())
lb_jhu.SetOrigin(t2_jhu.GetOrigin())
lb_jhu.SetSpacing(t2_jhu.GetSpacing())

sitk.WriteImage(lb_jhu, jhu_dir + '/bundles/'  + 'lb_jhu.mhd')

tn_jhu= sitk.ReadImage( jhu_dir + '/tensors/Dall2.img')
tn_jhu_np= sitk.GetArrayFromImage(tn_jhu)
tn_jhu_np= np.transpose(tn_jhu_np, [3,2,1,0])
'''

###############################################################################




save_data_thumbs= True


CC_stats= np.zeros((500, 24))
CC_i= -1

min_age= 0
max_age= 100

subsample_g_table= False
subsample_g_table_mode= 'random'
b_keep= [0,1000] 
fraction_del= 0.75

dwi_signal_mean= np.zeros((500, 4))
dwi_signal_mean_i= -1

for subj in subj_info['participant_id']:
    
    anat_dir= dhcp_dir + 'dhcp_anat_pipeline/sub-' + subj
    dmri_dir= dhcp_dir + 'dhcp_dmri_pipeline/sub-' + subj
    
    dmri_sess = [d for d in listdir(dmri_dir) if isdir(join(dmri_dir, d))]
    
    sess_tsv= anat_dir + '/sub-' + subj + '_sessions.tsv'
    sess_info= pd.read_csv( sess_tsv , delimiter= '\t')
    
    for j in range(len(sess_info)):
        for k in range(len(dmri_sess)):
            if sess_info.loc[j, 'session_id']== int(dmri_sess[k].split('ses-')[1]):
                
                age_c= sess_info.loc[j, 'scan_age']
                
                if age_c>=min_age and age_c<=max_age:
                    
                    print('\n'*0, 'Processing subject: ', subj, ',   session:', sess_info.loc[j, 'session_id'], ',    age:', age_c)
                    
                    subject= 'sub-' + subj
                    session= 'ses-'+ str(sess_info.loc[j, 'session_id'])
                    
                    dwi_dir = dmri_dir + '/' + session + '/' + 'dwi/'
                    ant_dir = anat_dir + '/' + session + '/' + 'anat/'
                    dav_dir = '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/dHCP/results/' + subject + '/' + session + '/'
                    os.makedirs(dav_dir, exist_ok=True)
                    
                    file_name= subject + '_' + session + '_desc-preproc_dwi.nii.gz'
                    d_img= sitk.ReadImage( dwi_dir + file_name )
                    d_img= sitk.GetArrayFromImage(d_img)
                    d_img= np.transpose( d_img, [3,2,1,0] )
                    
                    file_name= subject + '_' + session + '_desc-preproc_dwi'
                    b_vals= np.loadtxt( dwi_dir + file_name + '.bval' )
                    b_vecs= np.loadtxt( dwi_dir + file_name + '.bvec' )
                    
                    file_name= subject + '_' + session + '_desc-preproc_space-dwi_brainmask.nii.gz'
                    brain_mask_img= sitk.ReadImage( dwi_dir + file_name )
                    sitk.WriteImage(brain_mask_img, dav_dir + 'brain_mask_img.mhd')
                    brain_mask= sitk.GetArrayFromImage(brain_mask_img)
                    brain_mask= np.transpose( brain_mask, [2,1,0] )
                    
                    ref_dir= brain_mask_img.GetDirection()
                    ref_org= brain_mask_img.GetOrigin()
                    ref_spc= brain_mask_img.GetSpacing()
                    
                    file_name= subject + '_' + session + '_desc-restore_T2w.nii.gz'
                    t2_img= sitk.ReadImage( ant_dir + file_name )
                    t2_img= dk_aux.resample_imtar_to_imref(t2_img, brain_mask_img, sitk.sitkBSpline, False)
                    sitk.WriteImage(t2_img, dav_dir + 't2_img.mhd')
                    t2_img= sitk.GetArrayFromImage(t2_img)
                    t2_img= np.transpose( t2_img, [2,1,0] )
                    
                    '''if save_data_thumbs:
                        file_name= dhcp_dir + 'thumbs/' + str(int(age_c*1000)) + '.png'
                        if not os.path.exists(file_name):
                            temp=t2_img[:,:,32].T
                            temp[temp<0]=0
                            temp= (temp- temp.min())/temp.max()*256
                            temp= temp.astype(np.uint8)
                            im = Image.fromarray(temp)
                            im.save(file_name)'''
                    
                    file_name= subject + '_' + session + '_desc-drawem87_space-T2w_dseg.nii.gz'
                    pr_img= sitk.ReadImage( ant_dir + file_name )
                    pr_img= dk_aux.resample_imtar_to_imref(pr_img, brain_mask_img, sitk.sitkNearestNeighbor, False)
                    sitk.WriteImage(pr_img, dav_dir + 'pr_img.mhd')
                    pr_img= sitk.GetArrayFromImage(pr_img)
                    pr_img= np.transpose( pr_img, [2,1,0] )
                    
                    file_name= subject + '_' + session + '_desc-drawem9_space-T2w_dseg.nii.gz'
                    ts_img= sitk.ReadImage( ant_dir + file_name )
                    ts_img= dk_aux.resample_imtar_to_imref(ts_img, brain_mask_img, sitk.sitkNearestNeighbor, False)
                    sitk.WriteImage(ts_img, dav_dir + 'ts_img.mhd')
                    ts_img= sitk.GetArrayFromImage(ts_img)
                    ts_img= np.transpose( ts_img, [2,1,0] )
                    
                    
                    
                    
                    
                    '''cc_label= 48
                    mask_cc_part= pr_img==cc_label
                    
                    mean_signal = np.mean(d_img[mask_cc_part], axis=0)
                    
                    mask_noise = binary_dilation(brain_mask, iterations=10)
                    #mask_noise[..., :mask_noise.shape[-1]//2] = 1
                    mask_noise = ~mask_noise
                    
                    noise_std = np.std(d_img[mask_noise, :])'''
                    
                    
                    
                    
                    
                    if subsample_g_table:
                        
                        b_vals, b_vecs, keep_ind= crl_aux.subsample_g_table(b_vals, b_vecs, mode=subsample_g_table_mode, b_keep=b_keep, fraction_del=fraction_del)
                        
                        d_img= d_img[:,:,:,keep_ind]
                    
                    gtab = gradient_table(b_vals, b_vecs)
                    
                    
                    sx, sy, sz, _= d_img.shape
                    
                    
                    
                    
                    #skull= crl_aux.create_rough_skull_mask(d_img[:,:,:,0]>0, closing_radius= 4, radius= 6.0)
                    skull= crl_aux.skull_from_brain_mask(brain_mask, radius= 2.0)
                    skull_img= np.transpose(skull, [2,1,0])
                    skull_img= sitk.GetImageFromArray(skull_img)
                    skull_img.SetDirection(ref_dir)
                    skull_img.SetOrigin(ref_org)
                    skull_img.SetSpacing(ref_spc)
                    sitk.WriteImage(skull_img, dav_dir + 'skull.mhd')
                    
                    skull_thick= crl_aux.skull_from_brain_mask(brain_mask, radius= 5.0)
                    skull_img= np.transpose(skull_thick, [2,1,0])
                    skull_img= sitk.GetImageFromArray(skull_img)
                    skull_img.SetDirection(ref_dir)
                    skull_img.SetOrigin(ref_org)
                    skull_img.SetSpacing(ref_spc)
                    sitk.WriteImage(skull_img, dav_dir + 'skull_thick.mhd')
                    
                    
                    ###   Denoising
                    #
                    #method= 'nlm'
                    #
                    #if method=='nlm':
                    #    
                    #    sig_est= np.zeros(d_img.shape[-1])
                    #    d_img_den= np.zeros(d_img.shape)
                    #    
                    #    for i_vol in tqdm(range(d_img.shape[-1])):
                    #        vol= d_img[:,:,:,i_vol]
                    #        sigma = estimate_sigma(vol, N=0)
                    #        sig_est[i_vol]= sigma
                    #        den = nlmeans(vol, sigma=sigma, mask=vol>0, patch_radius= 2, block_radius = 2, rician= True)
                    #        d_img_den[:,:,:,i_vol]= den.copy()
                    #        
                    #elif method=='lpca':dwi_mean
                    #    
                    #    sigma = pca_noise_estimate(d_img, gtab, correct_bias=True, smooth=3)
                    #    d_img_den = localpca(d_img, sigma, tau_factor=2.3, patch_radius=2)
                    #    
                    #elif method=='mppca':
                    #    
                    #    d_img_den = mppca(d_img, patch_radius=2)
                    #
                    #d_img= d_img_den
                    
                    ###   DTI
                    
                    tenmodel = dti.TensorModel(gtab)
                    
                    tenfit = tenmodel.fit(d_img, brain_mask)
                    
                    FA = fractional_anisotropy(tenfit.evals)
                    FA[np.isnan(FA)] = 0
                    FA[skull==1]= 0
                    
                    FA_img= np.transpose(FA, [2,1,0])
                    FA_img= sitk.GetImageFromArray(FA_img)
                    FA_img.SetDirection(ref_dir)
                    FA_img.SetOrigin(ref_org)
                    FA_img.SetSpacing(ref_spc)
                    sitk.WriteImage(FA_img, dav_dir + 'FA.mhd')
                    sitk.WriteImage(FA_img, dav_dir + 'FA.nii.gz' )
                    
                    RGB = color_fa(FA, tenfit.evecs)
                    
                    GFA_img= np.transpose(RGB, [2,1,0,3])
                    GFA_img= sitk.GetImageFromArray(GFA_img)
                    GFA_img.SetDirection(ref_dir)
                    GFA_img.SetOrigin(ref_org)
                    GFA_img.SetSpacing(ref_spc)
                    sitk.WriteImage(GFA_img, dav_dir + 'GFA.mhd')
                    
                    
                    
                    
                    
                    
                    
                    FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
                    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
                    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
                    
                    #affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
                    #                     [0.0, 1.0, 0.0, 0.0],
                    #                     [0.0, 0.0, 1.0, 0.0],
                    #                     [0.0, 0.0, 0.0, 1.0],] )
                    affine= FA_img_nii.affine
                    
                    
                    
                    
                    ten_img= tenfit.lower_triangular()
                    ten_img= ten_img[:,:,:,(0,1,3,2,4,5)]
                    array_img = nib.Nifti1Image(ten_img, affine)
                    nib.save(array_img, dav_dir + 'TN_img.nii.gz' )
                    
                    
                    '''dwi_mean= np.zeros(len(b_vals))
                    dwi_std = np.zeros(len(b_vals))
                    for i_dwi in range(len(b_vals)):
                        temp= d_img[:,:,:,i_dwi]
                        temp= temp[brain_mask>0]
                        dwi_mean[i_dwi]= temp.mean()
                        dwi_std[i_dwi] = temp.std()
                    
                    dwi_signal_mean_i+= 1
                    dwi_signal_mean[dwi_signal_mean_i,:]= dwi_mean[b_vals==0].mean(), dwi_mean[b_vals==400].mean(), dwi_mean[b_vals==1000].mean(), dwi_mean[b_vals==2600].mean()
                    '''
                    
                    
                    ### CC
                    
                    cc_label= 48
                    cc_pixels= np.where(pr_img==cc_label)
                    cc_x_min, cc_x_max, cc_y_min, cc_y_max, cc_z_min, cc_z_max= \
                            cc_pixels[0].min(), cc_pixels[0].max(), cc_pixels[1].min(), cc_pixels[1].max(), cc_pixels[2].min(), cc_pixels[2].max()
                    
                    slc= (cc_z_min+cc_z_max)//2
                    
                    ###   response estimation
                    
                    sphere = get_sphere('repulsion724')
                    
                    response, ratio = auto_response(gtab, d_img[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max,:], roi_radius=300, fa_thr=0.7)
                    
                    '''CC_i+= 1
                    CC_stats[CC_i,0]= ratio
                    CC_stats[CC_i,1:4]= response[0]
                    CC_stats[CC_i,4]= response[1]
                    CC_stats[CC_i,5]= age_c
                    fa_cc= FA[pr_img==cc_label]
                    CC_stats[CC_i,6]= fa_cc.mean()
                    CC_stats[CC_i,7]= fa_cc.std()
                    CC_stats[CC_i,8]= fa_cc.max()
                    CC_stats[CC_i,9]= np.percentile(fa_cc, 50)
                    CC_stats[CC_i,10]= np.percentile(fa_cc, 90)
                    CC_stats[CC_i,11]= np.percentile(fa_cc, 95)
                    CC_stats[CC_i,12:16]= dwi_mean[b_vals==0].mean(), dwi_mean[b_vals==400].mean(), dwi_mean[b_vals==1000].mean(), dwi_mean[b_vals==2600].mean()
                    CC_stats[CC_i,16:20]= dwi_std[b_vals==0].mean(), dwi_std[b_vals==400].mean(), dwi_std[b_vals==1000].mean(), dwi_std[b_vals==2600].mean()
                    '''
                    
                    
                    
                    #################    dk_sD
                    
                    n_fib= 5
                    
                    fibr_img= np.zeros( (sx, sy, sz, 3, n_fib) )
                    resp_img= np.zeros( (sx, sy, sz, n_fib) )
                    resp_csf= np.zeros( (sx, sy, sz) )
                    
                    #sphere = get_sphere('symmetric724')
                    #sphere = get_sphere('repulsion724')
                    v, _ = sphere.vertices, sphere.faces
                    
                    fodf_sd= np.zeros( (sx, sy, sz, len(v)) )
                    
                    
                    #Lam= np.array( [0.0019, 0.0004 ])
                    Lam= response[0][:2]
                    d_iso=  0.003
                    
                    semi_sphere= False
                    with_iso= True
                    if semi_sphere:
                        H= crl_dci.sd_matrix_half(Lam, d_iso, b_vals, b_vecs.T, v, with_iso= with_iso)
                        v= v[:len(v)//2,:]
                    else:
                        H= crl_dci.sd_matrix(Lam, d_iso, b_vals, b_vecs.T, v, with_iso= with_iso)
                    
                    #mask= b0_img[:,:,:,0]>0
                    mask= d_img[:,:,:,0]>10
                    
                    for ix in tqdm(range(sx), ascii=True):
                        for iy in range(sy):
                            for iz in range(sz):#(slc,slc+1):
                                
                                if mask[ix, iy, iz]:
                                    
                                    #s= b1_img[ix, iy, iz,:]/b0_img[ix, iy, iz,0]
                                    s= d_img[ix, iy, iz,:]
                                    
                                    f_0= np.ones( H.shape[1] ) / H.shape[1]
                                    f_n= crl_dci.RL_deconv(H, s, f_0, n_iter= 150)
                                    #f_n= crl_dci.dRL_deconv(H, s, f_0, n_iter= 1000)
                                    
                                    fibers , responses = crl_dci.find_dominant_fibers(v, f_n[:len(v)], min_angle= np.pi/6, n_fib=n_fib)
                                    
                                    fibr_img[ix, iy, iz,:,:]= fibers
                                    resp_img[ix, iy, iz,:]= responses
                                    resp_csf[ix, iy, iz]  = f_n[-1]
                                    
                                    fodf_sd[ix, iy, iz,:]  = f_n[:-1]
                    
                    one_mask=   np.logical_and( resp_csf[:,:,:]<35, mask)
                    two_mask=   resp_img[:,:,:,1] / (resp_img[:,:,:,0]+1e-7)>0.75
                    two_mask=   np.logical_and( one_mask, two_mask )
                    thr_mask=   resp_img[:,:,:,2] / (resp_img[:,:,:,0]+1e-7)>0.85
                    thr_mask=   np.logical_and( thr_mask, two_mask )
                    
                    mose_sd= np.zeros((sx,sy,sz))
                    mose_sd[one_mask]= 1
                    mose_sd[two_mask]= 2
                    mose_sd[thr_mask]= 3
                    
                    my_list= [t2_img, pr_img, ts_img,
                              d_img[:,:,:,0], FA, RGB,
                              mose_sd]
                    
                    n_rows, n_cols = 3, 3
                    
                    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
                    
                    for i in range(n_rows*n_cols):
                        plt.subplot(n_rows, n_cols, i+1)
                        if i<len(my_list):
                            plt.imshow( my_list[i] [:, :, slc ])
                            if i>5:
                                plt.imshow( my_list[i] [:, :, slc ] , vmin=0, vmax=3 )
                        plt.axis('off');
                    
                    plt.tight_layout()
                    
                    fig.savefig(dav_dir + 'summary_sd.png')
                    
                    plt.close(fig)
                    
                    mose_sd= np.transpose(mose_sd, [2,1,0])
                    mose_sd= sitk.GetImageFromArray(mose_sd)
                    mose_sd.SetDirection(ref_dir)
                    mose_sd.SetOrigin(ref_org)
                    mose_sd.SetSpacing(ref_spc)
                    sitk.WriteImage(mose_sd, dav_dir + 'MOSE.mhd')
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    ##########   MDL ODF
                    
                    ######
                    N= 15
                    M= 15
                    
                    gpu_ind= 2
                    
                    n_feat_vec= np.array([2*N+2, 30, 60, 80, 80, 60, 30, 1])
                    
                    X = tf.placeholder("float32", [None, n_feat_vec[0]])
                    Y = tf.placeholder("float32", [None, n_feat_vec[-1]])
                    
                    p_keep_hidden = tf.placeholder("float")
                    
                    Y_p = dk_model.davood_reg_net(X, n_feat_vec, p_keep_hidden, bias_init=0.001)
                    
                    cost= tf.reduce_mean( tf.pow(( Y - Y_p ), 2) )
                    cost2= tf.reduce_mean( tf.div( tf.pow(( Y - Y_p ), 2) , tf.pow(( Y + 5.0 ), 3) ) )
                    
                    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
                    
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
                    
                    saver = tf.train.Saver(max_to_keep=50)
                    
                    sess = tf.Session()
                    sess.run(tf.global_variables_initializer())
                    
                    temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_diff_gtable_9.ckpt'
                    saver.restore(sess, temp_path)
                    
                    ######
                    
                    v, _ = sphere.vertices, sphere.faces
                    
                    b_vecs_test= b_vecs[:,b_vals==1000]
                    b1_img= d_img[:,:,:,b_vals==1000]
                    b0_img= d_img[:,:,:,b_vals==0]
                    b0_img= np.mean(b0_img, axis=-1)
                    
                    n_fib= 5
                    fibr_mag= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_mag= np.zeros( (sx, sy, sz, n_fib) )
                    
                    fodf_ml= np.zeros( (sx, sy, sz, len(v)) )
                    fodf_ml2= np.zeros( (sx, sy, sz, len(v)) )
                    
                    mose_ml= np.zeros( (sx, sy, sz) )
                    
                    mask= b0_img>0
                    
                    for ix in tqdm(range(sx), ascii=True):
                        for iy in range(sy):
                            for iz in range(sz):#range(sz):
                                
                                if mask[ix, iy, iz]:
                                    
                                    s= b1_img[ix, iy, iz,:]/ b0_img[ix, iy, iz]
                                    
                                    #Ang= np.zeros( (v.shape[0], 2) )
                                    
                                    batch_x= crl_dci.compute_angular_feature_vector_cont_full_sphere(s, b_vecs_test, v, N= N, M= M, full_circ= False)
                                    
                                    y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})[:,0]
                                    
                                    #Ang[:,0:1]= y_pred
                                    
                                    # ang_s= crl_aux.smooth_spherical(v.T, y_pred, n_neighbor=5, n_outlier=0, power=20,  method='1', div=60, s=1.0)
                                    ang_s= crl_aux.smooth_spherical_fast(v.T, y_pred, n_neighbor=5, n_outlier=0, power=20,  method='1', div=60, s=1.0)
                                    
                                    #ang_t, ang_p= Ang[:,1], Ang[:,0]
                                    #pred_fibers, pred_resps= crl_dci.find_dominant_fibers_dipy_way(sphere, 1/y_pred, 45, 2, peak_thr=.01, optimize=False, Psi= None, opt_tol=1e-7)
                                    #to_keep= np.where(1/pred_resps<25)[0]
                                    #pred_fibers= pred_fibers[:,to_keep]
                                    #pred_resps= pred_resps[to_keep]
                                    
                                    #n_pred= pred_fibers.shape[1]
                                    
                                    #mose_ml[ix,iy,iz]= n_pred
                                    
                                    #if pred_fibers.shape[1]>n_fib:
                                    #    pred_fibers= pred_fibers[:,:n_fib]
                                    #    pred_resps=  pred_resps[:n_fib]
                                    
                                    #fibr_mag[ix, iy, iz,:n_pred,:]= pred_fibers.T
                                    #resp_mag[ix, iy, iz,:n_pred]= pred_resps
                                    
                                    fodf_ml[ix,iy,iz,:]= 1/y_pred
                                    fodf_ml2[ix,iy,iz,:]= 1/ang_s
                                    
                    '''h5f = h5py.File(dav_dir + 'fodf_ml.h5','w')
                    h5f['fodf_ml2']= fodf_ml2
                    h5f['fodf_ml']= fodf_ml
                    h5f.close()'''
                    h5f = h5py.File(dav_dir + 'fodf_ml.h5','r')
                    fodf_ml2= h5f['fodf_ml2'][:]
                    fodf_ml = h5f['fodf_ml'][:]
                    h5f.close()
                    
                    ###########################################################
                    ###########################################################
                    # My tracking
                    pmf = fodf_ml2.copy()
                    pmf[pmf>1.0]= 1.0
                    pmf= pmf**20
                    prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                                    sphere=sphere)
                    mask= d_img[:,:,:,0]>10
                    csa_model = CsaOdfModel(gtab, sh_order=6)
                    gfa = csa_model.fit(d_img, mask=mask).gfa
                    stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
                    seed_mask= FA>0.10
                    seed_mask= seed_mask *  (1-skull)
                    FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
                    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
                    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
                    affine= FA_img_nii.affine
                    seeds = utils.seeds_from_mask(seed_mask, affine, density=2)
                    streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                                         affine=affine, step_size=.5)
                    streamlines = Streamlines(streamline_generator)
                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                    save_trk(sft, dav_dir+"tractogram_ml_2_20_FA10_seed2.trk",bbox_valid_check=False)
                    ###########################################################
                    # DIPY deterministic
                    mask= d_img[:,:,:,0]>10
                    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
                    csd_fit = csd_model.fit(d_img, mask=mask)
                    mask= d_img[:,:,:,0]>10
                    csa_model = CsaOdfModel(gtab, sh_order=6)
                    gfa = csa_model.fit(d_img, mask=mask).gfa
                    stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
                    seed_mask= FA>0.10
                    seed_mask= seed_mask *  (1-skull)
                    FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
                    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
                    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
                    affine= FA_img_nii.affine
                    seeds = utils.seeds_from_mask(seed_mask, affine, density=2)
                    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
                                            csd_fit.shm_coeff, max_angle=30., sphere=default_sphere)
                    streamline_generator = LocalTracking(detmax_dg, stopping_criterion, seeds,
                                                         affine, step_size=.5)
                    streamlines = Streamlines(streamline_generator)
                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                    save_trk(sft, dav_dir+"tractogram_deterministic_dipy.trk",bbox_valid_check=False)
                    ###########################################################
                    # DIPY probabilistic
                    mask= d_img[:,:,:,0]>10
                    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
                    csd_fit = csd_model.fit(d_img, mask=mask)
                    mask= d_img[:,:,:,0]>10
                    csa_model = CsaOdfModel(gtab, sh_order=6)
                    gfa = csa_model.fit(d_img, mask=mask).gfa
                    stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
                    seed_mask= FA>0.10
                    seed_mask= seed_mask *  (1-skull)
                    FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
                    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
                    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
                    affine= FA_img_nii.affine
                    seeds = utils.seeds_from_mask(seed_mask, affine, density=2)
                    fod = csd_fit.odf(small_sphere)
                    pmf = fod.clip(min=0)
                    prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                                    sphere=small_sphere)
                    streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                                         affine, step_size=.5)
                    streamlines = Streamlines(streamline_generator)
                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                    save_trk(sft, dav_dir+"tractogram_probabilistic_dipy.trk",bbox_valid_check=False)
                    ###########################################################
                    ###########################################################
                    
                    
                    
                    tract_address= dav_dir+"tractogram_ml_2_20_FA10_seed2.trk"
                    
                    all_sl = load_trk( tract_address , FA_img_nii)
                    #all_sl.to_vox()
                    all_sl = all_sl.streamlines
                    
                    all_sl= Streamlines(all_sl)
                    
                    
                    #  Keep long ones
                    
                    lengths = list(length(all_sl))
                    long_sl = Streamlines()
                    n_long= 0
                    for i, sl in enumerate(all_sl):
                        if i % 10000==0:
                            print(i)
                        if lengths[i] > 20:
                            long_sl.append(sl)
                            n_long+= sl.shape[0]
                    
                    '''long_data= long_sl.data
                    long_data= long_data[:n_long,:]
                    long_data_streamlines= Streamlines(long_data)'''
                    
                    sft = StatefulTractogram(long_sl, FA_img_nii, Space.RASMM)
                    save_trk(sft, dav_dir+"tractogram_ml_2_20_FA10_seed2_long_long.trk")
                    
                    
                    
                    # Keep confident ones
                    
                    cci = cluster_confidence(long_sl)
                    
                    long_conf_sl = Streamlines()
                    for i, sl in enumerate(long_sl):
                        if i % 10000==0:
                            print(i)
                        if cci[i] >= 100:
                            long_conf_sl.append(sl)
                    
                    sft= StatefulTractogram(long_conf_sl, FA_img_nii, Space.RASMM)
                    save_trk(sft, dav_dir+'tractogram_ml_2_20_FA10_seed2_long_long_conf.trk')
                    
                    
                    
                    
                    
                    
                    ####  Begin LiFE
                    
                    tract_address= dav_dir+"tractogram_ml_2_20_FA10_seed2_stop20_long.trk"
                    
                    life_sl = load_trk( tract_address , FA_img_nii)
                    life_sl = life_sl.streamlines
                    
                    #life_sl= Streamlines(life_sl)
                    
                    inv_affine = np.linalg.inv(FA_img_nii.affine)
                    life_affine= inv_affine
                    #life_affine= np.eye(4)
                    
                    fiber_model = life.FiberModel(gtab)
                    
                    fiber_fit = fiber_model.fit(d_img, life_sl, affine=life_affine)
                    
                    model_predict = fiber_fit.predict()
                    
                    model_error = model_predict - fiber_fit.data
                    model_rmse = np.sqrt(np.mean(model_error[:, 1:] ** 2, -1))
                    
                    beta_baseline = np.zeros(fiber_fit.beta.shape[0])
                    pred_weighted = np.reshape(dipy_opt.spdot(fiber_fit.life_matrix, beta_baseline),
                                               (fiber_fit.vox_coords.shape[0],
                                                np.sum(~gtab.b0s_mask)))
                    mean_pred = np.empty((fiber_fit.vox_coords.shape[0], gtab.bvals.shape[0]))
                    S0 = fiber_fit.b0_signal
                    
                    mean_pred[..., gtab.b0s_mask] = S0[:, None]
                    mean_pred[..., ~gtab.b0s_mask] =\
                        (pred_weighted + fiber_fit.mean_signal[:, None]) * S0[:, None]
                    mean_error = mean_pred - fiber_fit.data
                    mean_rmse = np.sqrt(np.mean(mean_error ** 2, -1))
                    
                    ####  end LiFE
                                        
                    
                    
                    
                    
                    ###  Begin RecoBundle
                    
                    bundle_names = [ 'CC', 'IFOF_R', 'AF_L', 'AF_L', 'CST_L', 'C_R' ]
                    bundle_dir   = '/home/davood/Documents/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/'
                    
                    target_file= dav_dir+"tractogram_ml_2_20_FA10_seed2_stop20.trk"
                    
                    atlas_file, _  = get_bundle_atlas_hcp842()
                    sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=False)
                    atlas = sft_atlas.streamlines
                    atlas_header = create_tractogram_header(atlas_file,
                                                            *sft_atlas.space_attributes)
                    
                    sft_target = load_trk(target_file, "same", bbox_valid_check=False)
                    target = sft_target.streamlines
                    target_header = create_tractogram_header(atlas_file,
                                                             *sft_atlas.space_attributes)
                    
                    moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
                        atlas, target, x0='affine', verbose=True, progressive=True,
                        rng=np.random.RandomState(1984))
                    
                    connectome_moved = StatefulTractogram(moved, target_header,
                                                   Space.RASMM)
                    save_trk(connectome_moved, dav_dir + 'reco_bund/' + 'connectome_moved.trk', bbox_valid_check=False)
                    
                    for bundle_name in bundle_names:
                        
                        sft_bundle = load_trk(bundle_dir+bundle_name+'.trk', "same", bbox_valid_check=False)
                        model_bundle = sft_bundle.streamlines
                        
                        rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(0))
                        
                        recognized_bundle, bundle_labels = rb.recognize(model_bundle=model_bundle,
                                                                    model_clust_thr=5.,
                                                                    reduction_thr=10,
                                                                    reduction_distance='mam',
                                                                    slr=True,
                                                                    slr_metric='asymmetric',
                                                                    pruning_distance='mam')
                        
                        reco_bundle = StatefulTractogram(target[bundle_labels], target_header,
                                                       Space.RASMM)
                        save_trk(reco_bundle, dav_dir + 'reco_bund/' + bundle_name + '.trk', bbox_valid_check=False)
                    
                    ###  End RecoBundle
                    
                    
                    
                    
                    
                    
                    ###  Begin visualize ODF
                    
                    slc_viz= 32
                    
                    fodf_pred= fodf_ml2[:, :, slc_viz:slc_viz+1, :]
                    fodf_pred_99= np.percentile(fodf_pred, 99)
                    fodf_pred[fodf_pred>fodf_pred_99]= fodf_pred_99
                    fodf_pred= fodf_pred**3
                    fodf_pred_sum= np.sum(fodf_pred, axis= -1)
                    for i in range(fodf_pred.shape[0]):
                        for j in range(fodf_pred.shape[1]):
                            for k in range(fodf_pred.shape[2]):
                                fodf_pred[i,j,k,:]/= (fodf_pred_sum[i,j,k]+1e-10)
                    
                    odf_actor = actor.odf_slicer(fodf_pred, sphere=sphere,
                             colormap='plasma', norm=True, scale=0.4)
                    odf_actor.display(z=0)
                    odf_actor.RotateZ(-90)
                    ren = window.Renderer()
                    ren.add(odf_actor)
                    ren.reset_camera_tight()
                    window.record(ren, out_path=dav_dir+'fODF_pred_notfiltered.png', size=(600, 600), magnification=4)
                    
                    
                    fodf_pred= fodf_ml2[:, :, :, :]
                    fodf_pred_99= np.percentile(fodf_pred, 99)
                    fodf_pred[fodf_pred>fodf_pred_99]= fodf_pred_99
                    fodf_pred= fodf_pred**3
                    fodf_pred_sum= np.sum(fodf_pred, axis= -1)
                    for i in range(fodf_pred.shape[0]):
                        for j in range(fodf_pred.shape[1]):
                            for k in range(fodf_pred.shape[2]):
                                fodf_pred[i,j,k,:]/= (fodf_pred_sum[i,j,k]+1e-10)
                                
                    ml_mask= b0_img>0
                    n_fib= 5
                    fibr_ml= np.zeros( (sx, sy, sz, 3, n_fib) )
                    resp_ml= np.zeros( (sx, sy, sz, n_fib) )
                    for ix in range(sx):
                        for iy in range(sy):
                            for iz in range(sz):
                                if ml_mask[ix,iy,iz]:
                                    fibers_temp, resps_temp, _  = peak_directions(fodf_pred[ix,iy,iz,:], sphere, 
                                                               relative_peak_threshold=.5, min_separation_angle=25)
                                    n_pred= len(resps_temp)
                                    if n_pred>n_fib:
                                        n_pred= n_fib
                                        fibers_temp= fibers_temp[:n_fib,:]
                                        resps_temp= resps_temp[:n_fib]
                                    fibr_ml[ix,iy,iz,:,:n_pred]= fibers_temp.T
                                    resp_ml[ix,iy,iz,:n_pred]= resps_temp
                    
                    window.clear(ren)
                    fodf_peaks = actor.peak_slicer(np.transpose(fibr_ml,[0,1,2,4,3]), resp_ml*150)
                    fodf_peaks.display(z=slc_viz)
                    fodf_peaks.RotateZ(-90)
                    ren.add(fodf_peaks)
                    
                    window.record(ren, out_path=dav_dir+'ml_peaks10b.png', size=(600, 600), magnification=4)
                    
                    #crl_aux.show_fibers(t2_img, ml_mask, fibr_ml, resp_ml, slc_viz, 150, scale_with_response=True, direction='z', colored= False)
                    crl_aux.show_fibers(t2_img, ml_mask, fibr_ml, resp_ml, slc_viz, 0.2, scale_with_response=False, direction='z', colored= False)
                    
                    slc_viz= 64
                    #crl_aux.show_fibers(t2_img, ml_mask, fibr_ml, resp_ml, slc_viz, 50, scale_with_response=True, direction='y', colored= False)
                    crl_aux.show_fibers(t2_img, ml_mask, fibr_ml, resp_ml, slc_viz, 0.25, scale_with_response=False, direction='y', colored= False)
                    
                    
                    
                    slc_viz= 32
                    
                    csa_mask= np.logical_and(brain_mask, b0_img>0)
                    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
                    csd_fit = csd_model.fit(d_img, csa_mask )
                    
                    csd_odf = csd_fit.odf(sphere)
                    
                    odf_actor = actor.odf_slicer(csd_odf[:,:,slc_viz:slc_viz+1, :], sphere=sphere,
                             colormap='plasma', norm=True, scale=0.4)
                    odf_actor.display(z=0)
                    odf_actor.RotateZ(-90)
                    ren = window.Renderer()
                    ren.add(odf_actor)
                    ren.reset_camera_tight()
                    window.record(ren, out_path=dav_dir+'fODF_csd.png', size=(600, 600), magnification=4)
                    
                    
                    n_fib= 5
                    fibr_csd= np.zeros( (sx, sy, sz, 3, n_fib ) )
                    resp_csd= np.zeros( (sx, sy, sz, n_fib) )
                    for ix in range(sx):
                        for iy in range(sy):
                            for iz in range(sz):
                                if csa_mask[ix,iy,iz]:
                                    fibers_temp, resps_temp, _  = peak_directions(csd_odf[ix,iy,iz,:], sphere, 
                                                               relative_peak_threshold=.5, min_separation_angle=25)
                                    n_pred= len(resps_temp)
                                    if n_pred>n_fib:
                                        n_pred= n_fib
                                        fibers_temp= fibers_temp[:n_fib,:]
                                        resps_temp= resps_temp[:n_fib]
                                    fibr_csd[ix,iy,iz,:,:n_pred]= fibers_temp.T
                                    resp_csd[ix,iy,iz,:n_pred]= resps_temp
                    
                    window.clear(ren)
                    fodf_peaks = actor.peak_slicer(np.transpose(fibr_csd,[0,1,2,4,3]), resp_csd)
                    fodf_peaks.display(z=slc_viz)
                    fodf_peaks.RotateZ(-90)
                    ren.add(fodf_peaks)
                    
                    window.record(ren, out_path=dav_dir+'csd_peaks.png', size=(600, 600), magnification=4)
                    
                    #crl_aux.show_fibers(t2_img, ml_mask, fibr_csd, resp_csd, slc_viz, 150, scale_with_response=True, direction='z', colored= False)
                    crl_aux.show_fibers(t2_img, ml_mask, fibr_csd, resp_csd, slc_viz, 0.2, scale_with_response=False, direction='z', colored= False)
                    
                    slc_viz= 64
                    #crl_aux.show_fibers(t2_img, ml_mask, fibr_csd, resp_csd, slc_viz, 50, scale_with_response=True, direction='y', colored= False)
                    crl_aux.show_fibers(t2_img, ml_mask, fibr_csd, resp_csd, slc_viz, 0.25, scale_with_response=False, direction='y', colored= False)
                    
                    #crl_aux.show_fibers(t2_img, csa_mask, fibr_csd, resp_csd, 32, 0.7, scale_with_response=True, direction='z', colored= False)
                    
                    
                    
                    
                    #  For MICCAI
                    
                    
                    fodf_pred= fodf_ml2.copy()
                    fodf_pred_99= np.percentile(fodf_pred, 99)
                    fodf_pred[fodf_pred>fodf_pred_99]= fodf_pred_99
                    fodf_pred= fodf_pred**2
                    fodf_pred_sum= np.sum(fodf_pred, axis= -1)
                    for i in range(fodf_pred.shape[0]):
                        for j in range(fodf_pred.shape[1]):
                            for k in range(fodf_pred.shape[2]):
                                fodf_pred[i,j,k,:]/= (fodf_pred_sum[i,j,k]+1e-10)
                    
                    csa_mask= np.logical_and(brain_mask, b0_img>0)
                    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
                    csd_fit = csd_model.fit(d_img, csa_mask )
                    csd_odf = csd_fit.odf(sphere)
                    csd_pred_99= np.percentile(csd_odf, 99)
                    csd_odf[csd_odf>csd_pred_99]= csd_pred_99
                    csd_odf_sum= np.sum(csd_odf, axis= -1)
                    for i in range(csd_odf.shape[0]):
                        for j in range(csd_odf.shape[1]):
                            for k in range(csd_odf.shape[2]):
                                csd_odf[i,j,k,:]/= (csd_odf_sum[i,j,k]+1e-10)
                    
                    i_saved= 0
                    
                    for slc_viz in [30, 32, 34]:
                        
                        for xmin in [40, 50, 60, 70, 80]:
                            
                            for ymin in [20, 30, 40, 50, 60, 80]:
                                
                                xmax, ymax= xmin+20, ymin+20
                                
                                if ts_img[xmin+10, ymin+10, slc_viz]==3:
                                    
                                    odf_actor = actor.odf_slicer(fodf_pred[xmin:xmax, ymin: ymax, slc_viz:slc_viz+1, :], sphere=sphere,
                                             colormap=None, norm=True, scale=0.4)
                                    odf_actor.display(z=0)
                                    odf_actor.RotateZ(-90)
                                    ren = window.renderer((256,256,256))
                                    #ren = window.Renderer()
                                    ren.add(odf_actor)
                                    ren.reset_camera_tight()
                                    window.record(ren, out_path=dav_dir+'miccai/fODF_pred.png', size=(600, 600), magnification=1)
                                    
                                    odf_actor = actor.odf_slicer(csd_odf[xmin:xmax, ymin:ymax, slc_viz:slc_viz+1, :], sphere=sphere,
                                             colormap=None, norm=True, scale=0.4)
                                    odf_actor.display(z=0)
                                    odf_actor.RotateZ(-90)
                                    ren = window.renderer((256,256,256))
                                    #ren = window.Renderer()
                                    ren.add(odf_actor)
                                    ren.reset_camera_tight()
                                    window.record(ren, out_path=dav_dir+'miccai/fODF_csd.png', size=(600, 600), magnification=1)
                                    
                                    fODF_pred= plt.imread(dav_dir+'miccai/fODF_pred.png')
                                    fODF_csd= plt.imread(dav_dir+'miccai/fODF_csd.png')
                                    
                                    fig, ax = plt.subplots(figsize=(18,8), nrows=1, ncols=3)
                                    
                                    ax[0].imshow( t2_img [:, :, slc_viz ], cmap='gray')
                                    rect5 = patches.Rectangle((ymin,xmin),ymax-ymin,xmax-xmin,linewidth=2,edgecolor='r',facecolor='None')
                                    ax[0].add_patch(rect5)
                                    ax[0].axis('off');
                                    ax[1].imshow( fODF_pred )
                                    ax[1].axis('off');
                                    ax[2].imshow( fODF_csd )
                                    ax[2].axis('off');
                                    
                                    i_saved+= 1
                                    
                                    plt.tight_layout()
                                    fig.savefig(dav_dir+'miccai/all_' + str(i_saved) + '.png')
                                    plt.close(fig)
                    
                    
                    
                    i_saved= 0
                    
                    for slc_viz in [40, 50, 60, 70, 80]:
                        
                        for xmin in [20, 30, 40, 50, 60, 70, 80]:
                            
                            for zmin in [10, 20, 30, 40]:
                                
                                if ts_img[xmin+10, slc_viz, zmin+10]==3:
                                    
                                    xmax, zmax= xmin+20, zmin+20
                                    zdraw= 64- zmax
                                    
                                    odf_actor = actor.odf_slicer(fodf_pred[xmin:xmax, slc_viz:slc_viz+1, zmin:zmax], sphere=sphere,
                                             colormap=None, norm=True, scale=0.4)
                                    odf_actor.display(y=0)
                                    odf_actor.RotateX(-90)
                                    ren = window.renderer((256,256,256))
                                    #ren = window.Renderer()
                                    ren.add(odf_actor)
                                    ren.reset_camera_tight()
                                    window.record(ren, out_path=dav_dir+'miccai/fODF_pred.png', size=(600, 600), magnification=1)
                                    
                                    odf_actor = actor.odf_slicer(csd_odf[xmin:xmax, slc_viz:slc_viz+1, zmin:zmax], sphere=sphere,
                                             colormap=None, norm=True, scale=0.4)
                                    odf_actor.display(y=0)
                                    odf_actor.RotateX(-90)
                                    ren = window.renderer((256,256,256))
                                    #ren = window.Renderer()
                                    ren.add(odf_actor)
                                    ren.reset_camera_tight()
                                    window.record(ren, out_path=dav_dir+'miccai/fODF_csd.png', size=(600, 600), magnification=1)
                                    
                                    fODF_pred= plt.imread(dav_dir+'miccai/fODF_pred.png')
                                    fODF_csd= plt.imread(dav_dir+'miccai/fODF_csd.png')
                                    
                                    fig, ax = plt.subplots(figsize=(18,8), nrows=1, ncols=3)
                                    ax[0].imshow( t2_img [:, slc_viz, ::-1 ].T, cmap='gray')
    #                                rect5 = patches.Rectangle((zmin,xmin),zmax-zmin,xmax-xmin,linewidth=2,edgecolor='r',facecolor='None')
                                    rect5 = patches.Rectangle((xmin,zdraw),xmax-xmin,zmax-zmin,linewidth=2,edgecolor='r',facecolor='None')
                                    ax[0].add_patch(rect5)
                                    ax[0].axis('off');
                                    ax[1].imshow( fODF_pred )
                                    ax[1].axis('off');
                                    ax[2].imshow( fODF_csd )
                                    ax[2].axis('off');
                                    
                                    i_saved+= 1
                                    
                                    plt.tight_layout()
                                    fig.savefig(dav_dir+'miccai/all_' + str(i_saved) + '.png')
                                    plt.close(fig)
                                
                    ###   Y
                    slc_viz= 64
                    
                    fodf_pred= fodf_ml2[:, slc_viz:slc_viz+1, :, :]**10
                    fodf_pred_sum= np.sum(fodf_pred, axis= -1)
                    for i in range(fodf_pred.shape[0]):
                        for j in range(fodf_pred.shape[1]):
                            for k in range(fodf_pred.shape[2]):
                                fodf_pred[i,j,k,:]/= fodf_pred_sum[i,j,k]
                    
                    fodf_pred= np.transpose(fodf_pred, [0,2,1,3])
                    
                    odf_actor = actor.odf_slicer(fodf_pred, sphere=sphere,
                             colormap='plasma', norm=True, scale=0.4)
                    odf_actor.display(z=0)
                    ren = window.Renderer()
                    ren.add(odf_actor)
                    ren.reset_camera_tight()
                    window.record(ren, out_path=dav_dir+'fODF_pred_y10.png', size=(600, 600), magnification=4)
                    
                    
                    fodf_pred= fodf_ml2[:, :, :, :]**10
                    fodf_pred_sum= np.sum(fodf_pred, axis= -1)
                    for i in range(fodf_pred.shape[0]):
                        for j in range(fodf_pred.shape[1]):
                            for k in range(fodf_pred.shape[2]):
                                fodf_pred[i,j,k,:]/= fodf_pred_sum[i,j,k]
                    ml_mask= b0_img>0
                    n_fib= 5
                    fibr_ml= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_ml= np.zeros( (sx, sy, sz, n_fib) )
                    for ix in range(sx):
                        for iy in range(sy):
                            for iz in range(sz):
                                if ml_mask[ix,iy,iz]:
                                    fibers_temp, resps_temp, _  = peak_directions(fodf_pred[ix,iy,iz,:], sphere, 
                                                               relative_peak_threshold=.5, min_separation_angle=25)
                                    n_pred= len(resps_temp)
                                    if n_pred>n_fib:
                                        n_pred= n_fib
                                        fibers_temp= fibers_temp[:n_fib,:]
                                        resps_temp= resps_temp[:n_fib]
                                    fibr_ml[ix,iy,iz,:n_pred,:]= fibers_temp
                                    resp_ml[ix,iy,iz,:n_pred]= resps_temp
                    
                    fibr_ml_tr= np.transpose(fibr_ml, [0,2,1,3,4])
                    resp_ml_tr= np.transpose(resp_ml, [0,2,1,3])
                    
                    window.clear(ren)
                    fodf_peaks = actor.peak_slicer(fibr_ml_tr, resp_ml_tr*5)
                    fodf_peaks.display(z=slc_viz)
                    ren.add(fodf_peaks)
                    
                    window.record(ren, out_path=dav_dir+'ml_peaks_y10.png', size=(600, 600), magnification=4)
                    
                    crl_aux.show_fibers(t2_img, ml_mask, fibr_ml, resp_ml, slc_viz, 0.2, scale_with_response=False, direction='y', colored= False)
                    
                    
                    csa_mask= np.logical_and(brain_mask, d_img[:,:,:,0]>20)
                    
                    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
                    
                    csd_fit = csd_model.fit(d_img, csa_mask )
                    
                    csd_odf = csd_fit.odf(sphere)
                    
                    odf_actor = actor.odf_slicer(csd_odf[:,:,slc_viz:slc_viz+1, :], sphere=sphere,
                             colormap='plasma', norm=True, scale=0.4)
                    odf_actor.display(z=0)
                    odf_actor.RotateZ(-90)
                    ren = window.Renderer()
                    ren.add(odf_actor)
                    ren.reset_camera_tight()
                    window.record(ren, out_path=dav_dir+'fODF_csd.png', size=(600, 600), magnification=4)
                    
                    n_fib= 5
                    fibr_csd= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_csd= np.zeros( (sx, sy, sz, n_fib) )
                    for ix in range(sx):
                        for iy in range(sy):
                            for iz in range(sz):
                                if csa_mask[ix,iy,iz]:
                                    fibers_temp, resps_temp, _  = peak_directions(csd_odf[ix,iy,iz,:], sphere, 
                                                               relative_peak_threshold=.5, min_separation_angle=25)
                                    n_pred= len(resps_temp)
                                    if n_pred>n_fib:
                                        n_pred= n_fib
                                        fibers_temp= fibers_temp[:n_fib,:]
                                        resps_temp= resps_temp[:n_fib]
                                    fibr_csd[ix,iy,iz,:n_pred,:]= fibers_temp
                                    resp_csd[ix,iy,iz,:n_pred]= resps_temp
                    
                    window.clear(ren)
                    fodf_peaks = actor.peak_slicer(fibr_csd, resp_csd)
                    fodf_peaks.display(z=32)
                    fodf_peaks.RotateZ(-90)
                    ren.add(fodf_peaks)
                    
                    window.record(ren, out_path=dav_dir+'csd_peaks.png', size=(600, 600), magnification=4)
                    
                    crl_aux.show_fibers(t2_img, csa_mask, fibr_csd, resp_csd, 32, 0.7, scale_with_response=True, direction='z', colored= False)
                    
                    
                    ###  End visualize ODF
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    ###  begin subsampling effects
                    
                    ###  My method
                    
                    v, _ = sphere.vertices, sphere.faces
                    
                    subsample_g_table= True
                    subsample_g_table_mode= 'keep_bs'
#                    subsample_g_table_mode= 'random_keep_bs'
                    b_keep= [0,1000] 
                    fraction_del= 0.5
                    
                    if subsample_g_table:
                        
                        b_vals_ML, b_vecs_ML, keep_ind_ML= crl_aux.subsample_g_table(b_vals, b_vecs, mode=subsample_g_table_mode,
                                                                                        b_keep=b_keep, fraction_del=fraction_del)
                        
                        d_img_ML= d_img[:,:,:,keep_ind_ML].copy()
                    
                    b0_img= d_img[:,:,:,b_vals==0]
                    b0_img= np.mean(b0_img, axis=-1)
                    mask= np.logical_and( b0_img>0, ts_img==3 )
                    
                    b_vecs_test= b_vecs_ML[:,b_vals_ML>0]
                    b1_img= d_img[:,:,:,keep_ind_ML[b_vals_ML==1000]]
                    
                    n_fib= 3
                    fibr_mag_1= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_mag_1= np.zeros( (sx, sy, sz, n_fib) )
                    fibr_mag_2= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_mag_2= np.zeros( (sx, sy, sz, n_fib) )
                    
                    mose_ml_1= np.zeros( (sx, sy, sz) )
                    mose_ml_2= np.zeros( (sx, sy, sz) )
                    
                    fodf_ml_1= np.zeros( (sx, sy, sz, len(v)) )
                    fodf_ml_2= np.zeros( (sx, sy, sz, len(v)) )
                    
                    theta_thr= 25
                    ang_res= 12.5
                    power= 20
                    
                    for ix in tqdm(range(sx), ascii=True):
                        for iy in range(sy):
                            for iz in range(slc, slc+1):#range(sz):
                                
                                if mask[ix, iy, iz]:
                                    
                                    s= b1_img[ix, iy, iz,:]/ b0_img[ix, iy, iz]
                                    
                                    batch_x= crl_dci.compute_angular_feature_vector_cont_full_sphere(s, b_vecs_test, v, N= N, M= M, full_circ= False)
                                    
                                    y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})[:,0]
                                    
                                    ang_s= crl_aux.smooth_spherical_fast(v.T, y_pred, n_neighbor=5, n_outlier=0, power=power,  method='1', div=60, s=1.0)
                                    
                                    ######
                                    v_sel, labels, n_pred, pred_fibers= crl_aux.spherical_clustering(v, ang_s, theta_thr=theta_thr, ang_res=ang_res, 
                                                                                                     max_n_cluster=n_fib, symmertric_sphere=False)
                                    
                                    n_pred= pred_fibers.shape[1]
                                    
                                    if n_pred>n_fib:
                                        pred_fibers= pred_fibers[:,:n_fib]
                                        n_pred= n_fib
                                        
                                    mose_ml_1[ix,iy,iz]= n_pred
                                    fibr_mag_1[ix,iy,iz,:n_pred,:]= pred_fibers.T
                                    
                                    ######
                                    pred_fibers, pred_resps= crl_dci.find_dominant_fibers_dipy_way(sphere, 1/ang_s, theta_thr, n_fib, peak_thr=.01, optimize=False, 
                                                                                                   Psi= None, opt_tol=1e-7)
                                    to_keep= np.where(1/pred_resps<theta_thr)[0]
                                    pred_fibers= pred_fibers[:,to_keep]
                                    pred_resps= pred_resps[to_keep]
                                    
                                    n_pred= pred_fibers.shape[1]
                                    
                                    mose_ml_2[ix,iy,iz]= n_pred
                                    
                                    if n_pred>n_fib:
                                        pred_fibers= pred_fibers[:,:n_fib]
                                        pred_resps=  pred_resps[:n_fib]
                                        n_pred= n_fib
                                    
                                    fibr_mag_2[ix, iy, iz,:n_pred,:]= pred_fibers.T
                                    resp_mag_2[ix, iy, iz,:n_pred]= pred_resps
                                    
                                    fodf_ml_1[ix,iy,iz,:]= 1/y_pred
                                    fodf_ml_2[ix,iy,iz,:]= 1/ang_s
                    
                    fodf_ml= fodf_ml_1.copy()
                    fodf_ml[fodf_ml>1]= 1
                    '''fodf_ml_99= np.percentile(fodf_ml[:,:,slc,:], 99)
                    fodf_ml[fodf_ml>fodf_ml_99]= fodf_ml_99'''
                    fodf_ml_sum= np.sum(fodf_ml, axis= -1)
                    for i in range(fodf_ml.shape[0]):
                        for j in range(fodf_ml.shape[1]):
                            for k in range(fodf_ml.shape[2]):
                                fodf_ml[i,j,k,:]/= (fodf_ml_sum[i,j,k]+1e-10)
                    fodf_ml_1= fodf_ml.copy()
                    
                    fodf_ml= fodf_ml_2.copy()
                    fodf_ml[fodf_ml>1]= 1
                    '''fodf_ml_99= np.percentile(fodf_ml[:,:,slc,:], 99)
                    fodf_ml[fodf_ml>fodf_ml_99]= fodf_ml_99'''
                    fodf_ml_sum= np.sum(fodf_ml, axis= -1)
                    for i in range(fodf_ml.shape[0]):
                        for j in range(fodf_ml.shape[1]):
                            for k in range(fodf_ml.shape[2]):
                                fodf_ml[i,j,k,:]/= (fodf_ml_sum[i,j,k]+1e-10)
                    fodf_ml_2= fodf_ml.copy()
                    
                    fibr_ml_1_1000_88= fibr_mag_1.copy()
                    resp_ml_1_1000_88= resp_mag_1.copy()
                    fodf_ml_1_1000_88= fodf_ml_1[:,:,slc:slc+1,:].copy()
                    fibr_ml_2_1000_88= fibr_mag_2.copy()
                    resp_ml_2_1000_88= resp_mag_2.copy()
                    fodf_ml_2_1000_88= fodf_ml_2[:,:,slc:slc+1,:].copy()
                    
                    
                    
                    
                    
                    
                    v, _ = sphere.vertices, sphere.faces
                    
                    subsample_g_table= True
#                    subsample_g_table_mode= 'keep_bs'
                    subsample_g_table_mode= 'random_keep_bs'
                    b_keep= [0,1000] 
                    fraction_del= 0.25
                    
                    if subsample_g_table:
                        
                        b_vals_ML, b_vecs_ML, keep_ind_ML= crl_aux.subsample_g_table(b_vals, b_vecs, mode=subsample_g_table_mode,
                                                                                        b_keep=b_keep, fraction_del=fraction_del)
                        
                        d_img_ML= d_img[:,:,:,keep_ind_ML].copy()
                    
                    b0_img= d_img[:,:,:,b_vals==0]
                    b0_img= np.mean(b0_img, axis=-1)
                    mask= np.logical_and( b0_img>0, ts_img==3 )
                    
                    b_vecs_test= b_vecs_ML[:,b_vals_ML>0]
                    b1_img= d_img[:,:,:,keep_ind_ML[b_vals_ML==1000]]
                    
                    n_fib= 3
                    fibr_mag_1= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_mag_1= np.zeros( (sx, sy, sz, n_fib) )
                    fibr_mag_2= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_mag_2= np.zeros( (sx, sy, sz, n_fib) )
                    
                    mose_ml_1= np.zeros( (sx, sy, sz) )
                    mose_ml_2= np.zeros( (sx, sy, sz) )
                    
                    fodf_ml_1= np.zeros( (sx, sy, sz, len(v)) )
                    fodf_ml_2= np.zeros( (sx, sy, sz, len(v)) )
                    
                    theta_thr= 25
                    ang_res= 12.5
                    power= 20
                    
                    for ix in tqdm(range(sx), ascii=True):
                        for iy in range(sy):
                            for iz in range(slc, slc+1):#range(sz):
                                
                                if mask[ix, iy, iz]:
                                    
                                    s= b1_img[ix, iy, iz,:]/ b0_img[ix, iy, iz]
                                    
                                    batch_x= crl_dci.compute_angular_feature_vector_cont_full_sphere(s, b_vecs_test, v, N= N, M= M, full_circ= False)
                                    
                                    y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})[:,0]
                                    
                                    ang_s= crl_aux.smooth_spherical_fast(v.T, y_pred, n_neighbor=5, n_outlier=0, power=power,  method='1', div=60, s=1.0)
                                    
                                    ######
                                    v_sel, labels, n_pred, pred_fibers= crl_aux.spherical_clustering(v, ang_s, theta_thr=theta_thr, ang_res=ang_res, 
                                                                                                     max_n_cluster=n_fib, symmertric_sphere=False)
                                    
                                    n_pred= pred_fibers.shape[1]
                                    
                                    if n_pred>n_fib:
                                        pred_fibers= pred_fibers[:,:n_fib]
                                        n_pred= n_fib
                                        
                                    mose_ml_1[ix,iy,iz]= n_pred
                                    fibr_mag_1[ix,iy,iz,:n_pred,:]= pred_fibers.T
                                    
                                    ######
                                    pred_fibers, pred_resps= crl_dci.find_dominant_fibers_dipy_way(sphere, 1/ang_s, theta_thr, n_fib, peak_thr=.01, optimize=False, 
                                                                                                   Psi= None, opt_tol=1e-7)
                                    to_keep= np.where(1/pred_resps<theta_thr)[0]
                                    pred_fibers= pred_fibers[:,to_keep]
                                    pred_resps= pred_resps[to_keep]
                                    
                                    n_pred= pred_fibers.shape[1]
                                    
                                    mose_ml_2[ix,iy,iz]= n_pred
                                    
                                    if n_pred>n_fib:
                                        pred_fibers= pred_fibers[:,:n_fib]
                                        pred_resps=  pred_resps[:n_fib]
                                        n_pred= n_fib
                                    
                                    fibr_mag_2[ix, iy, iz,:n_pred,:]= pred_fibers.T
                                    resp_mag_2[ix, iy, iz,:n_pred]= pred_resps
                                    
                                    fodf_ml_1[ix,iy,iz,:]= 1/y_pred
                                    fodf_ml_2[ix,iy,iz,:]= 1/ang_s
                    
                    fodf_ml= fodf_ml_1.copy()
                    fodf_ml[fodf_ml>1]= 1
                    '''fodf_ml_99= np.percentile(fodf_ml[:,:,slc,:], 99)
                    fodf_ml[fodf_ml>fodf_ml_99]= fodf_ml_99'''
                    fodf_ml_sum= np.sum(fodf_ml, axis= -1)
                    for i in range(fodf_ml.shape[0]):
                        for j in range(fodf_ml.shape[1]):
                            for k in range(fodf_ml.shape[2]):
                                fodf_ml[i,j,k,:]/= (fodf_ml_sum[i,j,k]+1e-10)
                    fodf_ml_1= fodf_ml.copy()
                    
                    fodf_ml= fodf_ml_2.copy()
                    fodf_ml[fodf_ml>1]= 1
                    '''fodf_ml_99= np.percentile(fodf_ml[:,:,slc,:], 99)
                    fodf_ml[fodf_ml>fodf_ml_99]= fodf_ml_99'''
                    fodf_ml_sum= np.sum(fodf_ml, axis= -1)
                    for i in range(fodf_ml.shape[0]):
                        for j in range(fodf_ml.shape[1]):
                            for k in range(fodf_ml.shape[2]):
                                fodf_ml[i,j,k,:]/= (fodf_ml_sum[i,j,k]+1e-10)
                    fodf_ml_2= fodf_ml.copy()
                    
                    fibr_ml_1_1000_66= fibr_mag_1.copy()
                    resp_ml_1_1000_66= resp_mag_1.copy()
                    fodf_ml_1_1000_66= fodf_ml_1[:,:,slc:slc+1,:].copy()
                    fibr_ml_2_1000_66= fibr_mag_2.copy()
                    resp_ml_2_1000_66= resp_mag_2.copy()
                    fodf_ml_2_1000_66= fodf_ml_2[:,:,slc:slc+1,:].copy()
                    
                    
                    
                    
                    
                    
                    v, _ = sphere.vertices, sphere.faces
                    
                    subsample_g_table= True
#                    subsample_g_table_mode= 'keep_bs'
                    subsample_g_table_mode= 'random_keep_bs'
                    b_keep= [0,1000] 
                    fraction_del= 0.5
                    
                    if subsample_g_table:
                        
                        b_vals_ML, b_vecs_ML, keep_ind_ML= crl_aux.subsample_g_table(b_vals, b_vecs, mode=subsample_g_table_mode,
                                                                                        b_keep=b_keep, fraction_del=fraction_del)
                        
                        d_img_ML= d_img[:,:,:,keep_ind_ML].copy()
                    
                    b0_img= d_img[:,:,:,b_vals==0]
                    b0_img= np.mean(b0_img, axis=-1)
                    mask= np.logical_and( b0_img>0, ts_img==3 )
                    
                    b_vecs_test= b_vecs_ML[:,b_vals_ML>0]
                    b1_img= d_img[:,:,:,keep_ind_ML[b_vals_ML==1000]]
                    
                    n_fib= 3
                    fibr_mag_1= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_mag_1= np.zeros( (sx, sy, sz, n_fib) )
                    fibr_mag_2= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_mag_2= np.zeros( (sx, sy, sz, n_fib) )
                    
                    mose_ml_1= np.zeros( (sx, sy, sz) )
                    mose_ml_2= np.zeros( (sx, sy, sz) )
                    
                    fodf_ml_1= np.zeros( (sx, sy, sz, len(v)) )
                    fodf_ml_2= np.zeros( (sx, sy, sz, len(v)) )
                    
                    theta_thr= 25
                    ang_res= 12.5
                    power= 20
                    
                    for ix in tqdm(range(sx), ascii=True):
                        for iy in range(sy):
                            for iz in range(slc, slc+1):#range(sz):
                                
                                if mask[ix, iy, iz]:
                                    
                                    s= b1_img[ix, iy, iz,:]/ b0_img[ix, iy, iz]
                                    
                                    batch_x= crl_dci.compute_angular_feature_vector_cont_full_sphere(s, b_vecs_test, v, N= N, M= M, full_circ= False)
                                    
                                    y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})[:,0]
                                    
                                    ang_s= crl_aux.smooth_spherical_fast(v.T, y_pred, n_neighbor=5, n_outlier=0, power=power,  method='1', div=60, s=1.0)
                                    
                                    ######
                                    v_sel, labels, n_pred, pred_fibers= crl_aux.spherical_clustering(v, ang_s, theta_thr=theta_thr, ang_res=ang_res, 
                                                                                                     max_n_cluster=n_fib, symmertric_sphere=False)
                                    
                                    n_pred= pred_fibers.shape[1]
                                    
                                    if n_pred>n_fib:
                                        pred_fibers= pred_fibers[:,:n_fib]
                                        n_pred= n_fib
                                        
                                    mose_ml_1[ix,iy,iz]= n_pred
                                    fibr_mag_1[ix,iy,iz,:n_pred,:]= pred_fibers.T
                                    
                                    ######
                                    pred_fibers, pred_resps= crl_dci.find_dominant_fibers_dipy_way(sphere, 1/ang_s, theta_thr, n_fib, peak_thr=.01, optimize=False, 
                                                                                                   Psi= None, opt_tol=1e-7)
                                    to_keep= np.where(1/pred_resps<theta_thr)[0]
                                    pred_fibers= pred_fibers[:,to_keep]
                                    pred_resps= pred_resps[to_keep]
                                    
                                    n_pred= pred_fibers.shape[1]
                                    
                                    mose_ml_2[ix,iy,iz]= n_pred
                                    
                                    if n_pred>n_fib:
                                        pred_fibers= pred_fibers[:,:n_fib]
                                        pred_resps=  pred_resps[:n_fib]
                                        n_pred= n_fib
                                    
                                    fibr_mag_2[ix, iy, iz,:n_pred,:]= pred_fibers.T
                                    resp_mag_2[ix, iy, iz,:n_pred]= pred_resps
                                    
                                    fodf_ml_1[ix,iy,iz,:]= 1/y_pred
                                    fodf_ml_2[ix,iy,iz,:]= 1/ang_s
                    
                    fodf_ml= fodf_ml_1.copy()
                    fodf_ml[fodf_ml>1]= 1
                    '''fodf_ml_99= np.percentile(fodf_ml[:,:,slc,:], 99)
                    fodf_ml[fodf_ml>fodf_ml_99]= fodf_ml_99'''
                    fodf_ml_sum= np.sum(fodf_ml, axis= -1)
                    for i in range(fodf_ml.shape[0]):
                        for j in range(fodf_ml.shape[1]):
                            for k in range(fodf_ml.shape[2]):
                                fodf_ml[i,j,k,:]/= (fodf_ml_sum[i,j,k]+1e-10)
                    fodf_ml_1= fodf_ml.copy()
                    
                    fodf_ml= fodf_ml_2.copy()
                    fodf_ml[fodf_ml>1]= 1
                    '''fodf_ml_99= np.percentile(fodf_ml[:,:,slc,:], 99)
                    fodf_ml[fodf_ml>fodf_ml_99]= fodf_ml_99'''
                    fodf_ml_sum= np.sum(fodf_ml, axis= -1)
                    for i in range(fodf_ml.shape[0]):
                        for j in range(fodf_ml.shape[1]):
                            for k in range(fodf_ml.shape[2]):
                                fodf_ml[i,j,k,:]/= (fodf_ml_sum[i,j,k]+1e-10)
                    fodf_ml_2= fodf_ml.copy()
                    
                    
                    fibr_ml_1_1000_44= fibr_mag_1.copy()
                    resp_ml_1_1000_44= resp_mag_1.copy()
                    fodf_ml_1_1000_44= fodf_ml_1[:,:,slc:slc+1,:].copy()
                    fibr_ml_2_1000_44= fibr_mag_2.copy()
                    resp_ml_2_1000_44= resp_mag_2.copy()
                    fodf_ml_2_1000_44= fodf_ml_2[:,:,slc:slc+1,:].copy()
                    
                    
                    
                    
                    h5f = h5py.File(dav_dir + 'downsample_ml2.h5','w')
                    h5f['fibr_ml_1_1000_88']= fibr_ml_1_1000_88
                    h5f['resp_ml_1_1000_88']= resp_ml_1_1000_88
                    h5f['fodf_ml_1_1000_88']= fodf_ml_1_1000_88
                    h5f['fibr_ml_2_1000_88']= fibr_ml_2_1000_88
                    h5f['resp_ml_2_1000_88']= resp_ml_2_1000_88
                    h5f['fodf_ml_2_1000_88']= fodf_ml_2_1000_88
                    h5f['fibr_ml_1_1000_66']= fibr_ml_1_1000_66
                    h5f['resp_ml_1_1000_66']= resp_ml_1_1000_66
                    h5f['fodf_ml_1_1000_66']= fodf_ml_1_1000_66
                    h5f['fibr_ml_2_1000_66']= fibr_ml_2_1000_66
                    h5f['resp_ml_2_1000_66']= resp_ml_2_1000_66
                    h5f['fodf_ml_2_1000_66']= fodf_ml_2_1000_66
                    h5f['fibr_ml_1_1000_44']= fibr_ml_1_1000_44
                    h5f['resp_ml_1_1000_44']= resp_ml_1_1000_44
                    h5f['fodf_ml_1_1000_44']= fodf_ml_1_1000_44
                    h5f['fibr_ml_2_1000_44']= fibr_ml_2_1000_44
                    h5f['resp_ml_2_1000_44']= resp_ml_2_1000_44
                    h5f['fodf_ml_2_1000_44']= fodf_ml_2_1000_44
                    h5f.close()
                    
                    
                    '''WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(fibr_ml_2_1000_88[:,:,slc:slc+1].copy(), 
                                                                          fibr_ml_2_1000_44[:,:,slc:slc+1].copy(), 
                                                                          mask[:,:,slc:slc+1])'''
                    
                    
                    fodf_pred= fodf_ml_2_1000_88.copy()
                    fodf_pred[np.isnan(fodf_pred)]= 0
                    fodf_pred= fodf_pred**3
                    fodf_pred_sum= np.sum(fodf_pred, axis= -1)
                    for i in range(fodf_pred.shape[0]):
                        for j in range(fodf_pred.shape[1]):
                            for k in range(fodf_pred.shape[2]):
                                fodf_pred[i,j,k,:]/= (fodf_pred_sum[i,j,k]+1e-10)
                    
                    fodf_pred[np.isnan(fodf_pred)]= 0
                    
                    ml_mask= b0_img>0
                    n_fib= 5
                    fibr_ml= np.zeros( (sx, sy, 1, 3, n_fib) )
                    resp_ml= np.zeros( (sx, sy, 1, n_fib) )
                    for ix in range(sx):
                        for iy in range(sy):
                            for iz in range(1):
                                if ml_mask[ix,iy,slc+iz]:
                                    fibers_temp, resps_temp, _  = peak_directions(fodf_pred[ix,iy,iz,:], sphere, 
                                                                                   relative_peak_threshold=.5, min_separation_angle=25)
                                    n_pred= len(resps_temp)
                                    if n_pred>n_fib:
                                        n_pred= n_fib
                                        fibers_temp= fibers_temp[:n_fib,:]
                                        resps_temp= resps_temp[:n_fib]
                                    fibr_ml[ix,iy,iz,:,:n_pred]= fibers_temp.T
                                    resp_ml[ix,iy,iz,:n_pred]= resps_temp
                    
                    peaks88= fibr_ml
                    
                    
                    
                    fodf_pred= fodf_ml_2_1000_66.copy()
                    fodf_pred[np.isnan(fodf_pred)]= 0
                    fodf_pred= fodf_pred**3
                    fodf_pred_sum= np.sum(fodf_pred, axis= -1)
                    for i in range(fodf_pred.shape[0]):
                        for j in range(fodf_pred.shape[1]):
                            for k in range(fodf_pred.shape[2]):
                                fodf_pred[i,j,k,:]/= (fodf_pred_sum[i,j,k]+1e-10)
                    
                    fodf_pred[np.isnan(fodf_pred)]= 0
                    
                    ml_mask= b0_img>0
                    n_fib= 5
                    fibr_ml= np.zeros( (sx, sy, 1, 3, n_fib) )
                    resp_ml= np.zeros( (sx, sy, 1, n_fib) )
                    for ix in range(sx):
                        for iy in range(sy):
                            for iz in range(1):
                                if ml_mask[ix,iy,slc+iz]:
                                    fibers_temp, resps_temp, _  = peak_directions(fodf_pred[ix,iy,iz,:], sphere, 
                                                                                   relative_peak_threshold=.5, min_separation_angle=25)
                                    n_pred= len(resps_temp)
                                    if n_pred>n_fib:
                                        n_pred= n_fib
                                        fibers_temp= fibers_temp[:n_fib,:]
                                        resps_temp= resps_temp[:n_fib]
                                    fibr_ml[ix,iy,iz,:,:n_pred]= fibers_temp.T
                                    resp_ml[ix,iy,iz,:n_pred]= resps_temp
                    
                    peaks66= fibr_ml
                    
                    
                    
                    fodf_pred= fodf_ml_2_1000_44.copy()
                    fodf_pred[np.isnan(fodf_pred)]= 0
                    fodf_pred= fodf_pred**3
                    fodf_pred_sum= np.sum(fodf_pred, axis= -1)
                    for i in range(fodf_pred.shape[0]):
                        for j in range(fodf_pred.shape[1]):
                            for k in range(fodf_pred.shape[2]):
                                fodf_pred[i,j,k,:]/= (fodf_pred_sum[i,j,k]+1e-10)
                    
                    fodf_pred[np.isnan(fodf_pred)]= 0
                    
                    ml_mask= b0_img>0
                    n_fib= 5
                    fibr_ml= np.zeros( (sx, sy, 1, 3, n_fib) )
                    resp_ml= np.zeros( (sx, sy, 1, n_fib) )
                    for ix in range(sx):
                        for iy in range(sy):
                            for iz in range(1):
                                if ml_mask[ix,iy,slc+iz]:
                                    fibers_temp, resps_temp, _  = peak_directions(fodf_pred[ix,iy,iz,:], sphere, 
                                                                                   relative_peak_threshold=.5, min_separation_angle=25)
                                    n_pred= len(resps_temp)
                                    if n_pred>n_fib:
                                        n_pred= n_fib
                                        fibers_temp= fibers_temp[:n_fib,:]
                                        resps_temp= resps_temp[:n_fib]
                                    fibr_ml[ix,iy,iz,:,:n_pred]= fibers_temp.T
                                    resp_ml[ix,iy,iz,:n_pred]= resps_temp
                    
                    peaks44= fibr_ml
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    ###  CSD
                    
                    subsample_g_table= True
                    subsample_g_table_mode= 'keep_bs'
#                    subsample_g_table_mode= 'random_keep_bs'
                    b_keep= [0,1000] 
                    fraction_del= 0.5
                    
                    if subsample_g_table:
                        
                        b_vals_CSD, b_vecs_CSD, keep_ind_CSD= crl_aux.subsample_g_table(b_vals, b_vecs, mode=subsample_g_table_mode,
                                                                                        b_keep=b_keep, fraction_del=fraction_del)
                        
                        d_img_CSD= d_img[:,:,:,keep_ind_CSD].copy()
                    
                    gtab_CSD = gradient_table(b_vals_CSD, b_vecs_CSD)
                    
                    b0_img= d_img[:,:,:,b_vals==0]
                    b0_img= np.mean(b0_img, axis=-1)
                    mask= np.logical_and( b0_img>0, ts_img==3 )
                    
                    par= 0.3
                    
                    csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response)
                    
                    csd_peaks = peaks_from_model(model=csd_model, data=d_img_CSD, mask= mask, sphere=sphere,
                                                 relative_peak_threshold=0.5, min_separation_angle=30, parallel=False, return_odf=True, npeaks=3)
                    
                    resp_mag_0= csd_peaks.peak_values
                    fibr_mag_0= csd_peaks.peak_dirs
                    
                    img_mos_dipy= np.sum(resp_mag_0>par, axis=-1)
                    
                    img_mos_dipy[mask==0]= 0
                    
                    csd_odf= csd_peaks.odf.copy()
                    csd_pred_99= np.percentile(csd_odf, 99)
                    csd_odf[csd_odf>csd_pred_99]= csd_pred_99
                    csd_odf_sum= np.sum(csd_odf, axis= -1)
                    for i in range(csd_odf.shape[0]):
                        for j in range(csd_odf.shape[1]):
                            for k in range(csd_odf.shape[2]):
                                csd_odf[i,j,k,:]/= (csd_odf_sum[i,j,k]+1e-10)
                    
                    fibr_csd_1000_88= fibr_mag_0.copy()
                    resp_csd_1000_88= resp_mag_0.copy()
                    fodf_csd_1000_88= csd_odf[:,:,slc:slc+1,:].copy()
                    
                    
                    
                    
                    
                    subsample_g_table= True
#                    subsample_g_table_mode= 'keep_bs'
                    subsample_g_table_mode= 'random_keep_bs'
                    b_keep= [0,1000] 
                    fraction_del= 0.25
                    
                    if subsample_g_table:
                        
                        b_vals_CSD, b_vecs_CSD, keep_ind_CSD= crl_aux.subsample_g_table(b_vals, b_vecs, mode=subsample_g_table_mode,
                                                                                        b_keep=b_keep, fraction_del=fraction_del)
                        
                        d_img_CSD= d_img[:,:,:,keep_ind_CSD].copy()
                    
                    gtab_CSD = gradient_table(b_vals_CSD, b_vecs_CSD)
                    
                    b0_img= d_img[:,:,:,b_vals==0]
                    b0_img= np.mean(b0_img, axis=-1)
                    mask= np.logical_and( b0_img>0, ts_img==3 )
                    
                    par= 0.3
                    
                    csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response)
                    
                    csd_peaks = peaks_from_model(model=csd_model, data=d_img_CSD, mask= mask, sphere=sphere,
                                                 relative_peak_threshold=0.5, min_separation_angle=30, parallel=False, return_odf=True, npeaks=3)
                    
                    resp_mag_0= csd_peaks.peak_values
                    fibr_mag_0= csd_peaks.peak_dirs
                    
                    img_mos_dipy= np.sum(resp_mag_0>par, axis=-1)
                    
                    img_mos_dipy[mask==0]= 0
                    
                    csd_odf= csd_peaks.odf.copy()
                    csd_pred_99= np.percentile(csd_odf, 99)
                    csd_odf[csd_odf>csd_pred_99]= csd_pred_99
                    csd_odf_sum= np.sum(csd_odf, axis= -1)
                    for i in range(csd_odf.shape[0]):
                        for j in range(csd_odf.shape[1]):
                            for k in range(csd_odf.shape[2]):
                                csd_odf[i,j,k,:]/= (csd_odf_sum[i,j,k]+1e-10)
                    
                    fibr_csd_1000_66= fibr_mag_0.copy()
                    resp_csd_1000_66= resp_mag_0.copy()
                    fodf_csd_1000_66= csd_odf[:,:,slc:slc+1,:].copy()
                    
                    
                    
                    
                    subsample_g_table= True
#                    subsample_g_table_mode= 'keep_bs'
                    subsample_g_table_mode= 'random_keep_bs'
                    b_keep= [0,1000] 
                    fraction_del= 0.5
                    
                    if subsample_g_table:
                        
                        b_vals_CSD, b_vecs_CSD, keep_ind_CSD= crl_aux.subsample_g_table(b_vals, b_vecs, mode=subsample_g_table_mode,
                                                                                        b_keep=b_keep, fraction_del=fraction_del)
                        
                        d_img_CSD= d_img[:,:,:,keep_ind_CSD].copy()
                    
                    gtab_CSD = gradient_table(b_vals_CSD, b_vecs_CSD)
                    
                    b0_img= d_img[:,:,:,b_vals==0]
                    b0_img= np.mean(b0_img, axis=-1)
                    mask= np.logical_and( b0_img>0, ts_img==3 )
                    
                    par= 0.3
                    
                    csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response)
                    
                    csd_peaks = peaks_from_model(model=csd_model, data=d_img_CSD, mask= mask, sphere=sphere,
                                                 relative_peak_threshold=0.5, min_separation_angle=30, parallel=False, return_odf=True, npeaks=3)
                    
                    resp_mag_0= csd_peaks.peak_values
                    fibr_mag_0= csd_peaks.peak_dirs
                    
                    img_mos_dipy= np.sum(resp_mag_0>par, axis=-1)
                    
                    img_mos_dipy[mask==0]= 0
                    
                    csd_odf= csd_peaks.odf.copy()
                    csd_pred_99= np.percentile(csd_odf, 99)
                    csd_odf[csd_odf>csd_pred_99]= csd_pred_99
                    csd_odf_sum= np.sum(csd_odf, axis= -1)
                    for i in range(csd_odf.shape[0]):
                        for j in range(csd_odf.shape[1]):
                            for k in range(csd_odf.shape[2]):
                                csd_odf[i,j,k,:]/= (csd_odf_sum[i,j,k]+1e-10)
                    
                    fibr_csd_1000_44= fibr_mag_0.copy()
                    resp_csd_1000_44= resp_mag_0.copy()
                    fodf_csd_1000_44= csd_odf[:,:,slc:slc+1,:].copy()
                    
                    
                    
                    
                    
                    h5f = h5py.File(dav_dir + 'downsample_csd.h5','w')
                    h5f['fibr_csd_1000_88']= fibr_csd_1000_88
                    h5f['resp_csd_1000_88']= resp_csd_1000_88
                    h5f['fodf_csd_1000_88']= fodf_csd_1000_88
                    h5f['fibr_csd_1000_66']= fibr_csd_1000_66
                    h5f['resp_csd_1000_66']= resp_csd_1000_66
                    h5f['fodf_csd_1000_66']= fodf_csd_1000_66
                    h5f['fibr_csd_1000_44']= fibr_csd_1000_44
                    h5f['resp_csd_1000_44']= resp_csd_1000_44
                    h5f['fodf_csd_1000_44']= fodf_csd_1000_44
                    h5f.close()
                    
                    
                    
                    common_mask= np.logical_and(  fibr_csd_1000_88[:,:,slc,0,0]!=0 , fibr_csd_1000_66[:,:,slc,0,0]!=0 )
                    common_mask= np.logical_and(  common_mask, fibr_csd_1000_44[:,:,slc,0,0]!=0)
                    common_mask= np.logical_and(  common_mask, peaks88[:,:,0,0,0]!=0)
                    common_mask= np.logical_and(  common_mask, peaks66[:,:,0,0,0]!=0)
                    common_mask= np.logical_and(  common_mask, peaks44[:,:,0,0,0]!=0)
                    common_mask= common_mask[:,:,np.newaxis]
                    
                    
                    
                    
                    Err_mat, WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(np.transpose(peaks88[:,:,:,:,:1],[0,1,2,4,3]).copy(), 
                                                                          np.transpose(peaks44[:,:,:,:,:1],[0,1,2,4,3]).copy(), 
                                                                          common_mask)
                    
                    temp= np.squeeze( Err_mat )
                    temp2= mask[:,:,slc]
                    temp3= temp[temp2]
                    temp3.mean()
                    temp4= temp3[temp3<45]
                    temp4.mean()
                    
                    
                    gold= np.squeeze(fodf_ml_2_1000_88)
                    pred= np.squeeze(fodf_ml_2_1000_44)
                    gold[gold>1]= 1
                    pred[pred>1]= 1
                    
                    diff_norm= np.linalg.norm(pred-gold, axis=-1)
                    gold_norm= np.linalg.norm(gold, axis=-1)
                    err_norm= diff_norm/ (gold_norm+1e-10)
                    err_norm= err_norm[err_norm>0]
                    
                    #plt.figure(), plt.hist(err_norm, bins=100);
                    error_sum= np.array([err_norm.mean(), err_norm.std(), err_norm.min(), err_norm.max()])
                    print( error_sum )
                    
                    err_95= np.percentile(err_norm, 95)
                    err_norm_95= err_norm[err_norm<err_95]
                    #plt.figure(), plt.hist(err_norm_95, bins=100);
                    error_sum_95= np.array([err_norm_95.mean(), err_norm_95.std(), err_norm_95.min(), err_norm_95.max()])
                    print( error_sum_95 )
                    
                    AAA= np.concatenate( (error_sum, error_sum_95) )
                    
                    
                    
                    
                    
                    
                    
                    Err_mat, WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(fibr_csd_1000_88[:,:,slc:slc+1,:1,:], 
                                                                                   fibr_csd_1000_44[:,:,slc:slc+1,:1,:], 
                                                                                   common_mask, penalize_miss=True)
                    
                    temp= np.squeeze( Err_mat )
                    temp2= mask[:,:,slc]
                    temp3= temp[temp2]
                    temp3.mean()
                    temp4= temp3[temp3<45]
                    temp4.mean()
                    
                    
                    gold= np.squeeze(fodf_csd_1000_88)
                    pred= np.squeeze(fodf_csd_1000_44)
                    gold[gold>1]= 1
                    pred[pred>1]= 1
                    
                    diff_norm= np.linalg.norm(pred-gold, axis=-1)
                    gold_norm= np.linalg.norm(gold, axis=-1)
                    err_norm= diff_norm/ (gold_norm+1e-10)
                    err_norm= err_norm[err_norm>0]
                    
                    #plt.figure(), plt.hist(err_norm, bins=100);
                    error_sum= np.array([err_norm.mean(), err_norm.std(), err_norm.min(), err_norm.max()])
                    print( error_sum )
                    
                    err_95= np.percentile(err_norm, 95)
                    err_norm_95= err_norm[err_norm<err_95]
                    #plt.figure(), plt.hist(err_norm_95, bins=100);
                    error_sum_95= np.array([err_norm_95.mean(), err_norm_95.std(), err_norm_95.min(), err_norm_95.max()])
                    print( error_sum_95 )
                    
                    AAA= np.concatenate( (error_sum, error_sum_95) )
                    
                    
                    ###  end  subsampling effects
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    '''fixed_image= FA_img
                    moving_image= sitk.ReadImage( jhu_dir + 'JHU_neonate_SS_fass.img')
                    moving_image.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                    
                    ###
                    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
                                                                          moving_image, 
                                                                          sitk.Euler3DTransform(), 
                                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
                    
                    registration_method = sitk.ImageRegistrationMethod()
                    
                    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
                    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
                    registration_method.SetMetricSamplingPercentage(0.50)
                    
                    registration_method.SetInterpolator(sitk.sitkLinear)
                    
                    registration_method.SetOptimizerAsGradientDescent(learningRate=10.0, numberOfIterations=1000)
                    registration_method.SetOptimizerScalesFromPhysicalShift() 
                    
                    registration_method.SetInitialTransform(initial_transform, inPlace=False)
                    
                    final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                                     sitk.Cast(moving_image, sitk.sitkFloat32))
                    
                    resample = sitk.ResampleImageFilter()
                    resample.SetReferenceImage(fixed_image)
                    
                    resample.SetInterpolator(sitk.sitkNearestNeighbor)  
                    resample.SetTransform(final_transform_v1)
                    ###
                    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
                                                                          moving_image, 
                                                                          sitk.Euler3DTransform(), 
                                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
                    
                    registration_method = sitk.ImageRegistrationMethod()
                    
                    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
                    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
                    registration_method.SetMetricSamplingPercentage(0.50)
                    
                    registration_method.SetInterpolator(sitk.sitkLinear)
                       
                    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100)
                    registration_method.SetOptimizerScalesFromPhysicalShift() 
                    
                    final_transform = sitk.Euler3DTransform(initial_transform)
                    registration_method.SetInitialTransform(final_transform)
                    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
                    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
                    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
                    
                    registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                sitk.Cast(moving_image, sitk.sitkFloat32))
                    
                    final_transform_v22 = sitk.Transform(final_transform)
                    
                    resample = sitk.ResampleImageFilter()
                    resample.SetReferenceImage(fixed_image)
                    
                    resample.SetInterpolator(sitk.sitkLinear)  
                    resample.SetTransform(final_transform_v22)
                    
                    
                    
                    moving_image_3= resample.Execute(moving_image)
                    sitk.WriteImage(moving_image_3 , dav_dir + 'fa_jhu.mha')
                    
                    resample.SetInterpolator(sitk.sitkNearestNeighbor)
                    moving_image_4= resample.Execute(lb_jhu)
                    sitk.WriteImage(moving_image_4 , dav_dir + 'lb_jhu.mha')
                    
                    
                    sitk.WriteTransform(final_transform_v1, dav_dir + 'lb_jhu_transform.tfm')'''





                    
                    
                    ####   F_test
                    '''
                    b0_ind= np.squeeze( np.where(b_vals==0) )
                    b1_ind= np.squeeze( np.where(b_vals>0) )
                    
                    b0_img= d_img[:,:,:,b0_ind]
                    b1_img= d_img[:,:,:,b1_ind]
                    
                    b0_img= np.mean(b0_img, axis=-1)[:,:,:,np.newaxis]
                    
                    b_vals= b_vals[b1_ind]
                    b_vecs= b_vecs[:,b1_ind]
                    
                    m_max= 3
                    
                    img_mos_mf= np.zeros( (sx,sy,sz) )
                    
                    threshold= 20
                    
                    mask= b0_img[:, :,: ,0]>0
                            
                    for ix in tqdm(range(sx), ascii=True):
                        for iy in range(sy):
                            for iz in range(slc,slc+1):
                                
                                if mask[ix, iy, iz]:
                                    
                                    s= b1_img[ix, iy, iz,:]/b0_img[ix, iy, iz,0]
                                    
                                    m_opt= crl_dci.model_selection_f_test(s, b_vals, b_vecs.T, m_max= m_max, threshold= threshold, 
                                                                      condition_mode= 'F_val', model='ball_n_sticks')
                                    
                                    img_mos_mf[ix,iy,iz]= m_opt
                    
                    my_list= [t2_img, pr_img, ts_img,
                              d_img[:,:,:,0], FA, RGB,
                              img_mos_mf]
                    
                    n_rows, n_cols = 3, 3
                    
                    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
                    
                    for i in range(n_rows*n_cols):
                        plt.subplot(n_rows, n_cols, i+1)
                        if i<len(my_list):
                            plt.imshow( my_list[i] [:, :, slc ])
                            if i>5:
                                plt.imshow( my_list[i] [:, :, slc ] , vmin=0, vmax=3 )
                        plt.axis('off');
                    
                    plt.tight_layout()
                    
                    fig.savefig(dav_dir + 'summary_ftest.png')
                    
                    plt.close(fig)
                    '''
                    
                    
                    ###   Reconstruction with the Sparse Fascicle Model
                    
                    '''sf_mask= np.logical_and(brain_mask, d_img[:,:,:,0]>20)
                    
                    sf_model = sfm.SparseFascicleModel(gtab, sphere=sphere,
                                                       l1_ratio=0.5, alpha=0.002,
                                                       response=response[0])
                    
                    sf_fit = sf_model.fit( d_img[:,:,slc:slc+1,:], sf_mask[:,:,slc:slc+1] )
                    
                    sf_odf = sf_fit.odf(sphere)
                    
                    sf_peaks = dpp.peaks_from_model(sf_model, d_img[:,:,slc:slc+1,:], sphere,
                                                    relative_peak_threshold=0.5, min_separation_angle=25, return_sh=False)
                    
                    peak_vals= sf_peaks.peak_values
                    
                    img_mos_sf= np.sum(peak_vals>0, axis=-1)
                    
                    my_list= [t2_img, pr_img, ts_img,
                              d_img[:,:,:,0], FA, RGB,
                              img_mos_sf]
                    
                    n_rows, n_cols = 3, 3
                    
                    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
                    
                    for i in range(n_rows*n_cols):
                        plt.subplot(n_rows, n_cols, i+1)
                        if i<len(my_list):
                            if i<6:
                                plt.imshow( my_list[i] [:, :, slc ])
                            else:
                                plt.imshow( my_list[i] [:, :, 0 ] , vmin=0, vmax=3 )
                        plt.axis('off');
                    
                    plt.tight_layout()
                    
                    fig.savefig(dav_dir + 'summary_sfm.png')
                    
                    plt.close(fig)'''
                    
                    
                    
                    
                    ###   CSA
                    
                    '''csa_mask= np.logical_and(brain_mask, d_img[:,:,:,0]>20)
                    
                    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
                    
                    csd_fit = csd_model.fit(d_img, csa_mask )
                    
                    csd_odf = csd_fit.odf(sphere)
                    
                    csd_peaks = peaks_from_model(model=csd_model, data=d_img[:,:,slc:slc+1,:], mask= csa_mask[:,:,slc:slc+1], sphere=sphere,
                                             relative_peak_threshold=0.5, min_separation_angle=25, parallel=True)
                    
                    peak_vals= csd_peaks.peak_values
                    
                    img_mos_csd= np.sum(peak_vals>0.25, axis=-1)
                    
                    my_list= [t2_img, pr_img, ts_img,
                              d_img[:,:,:,0], FA, RGB,
                              img_mos_csd]
                    
                    n_rows, n_cols = 3, 3
                    
                    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
                    
                    for i in range(n_rows*n_cols):
                        plt.subplot(n_rows, n_cols, i+1)
                        if i<len(my_list):
                            if i<6:
                                plt.imshow( my_list[i] [:, :, slc ])
                            else:
                                plt.imshow( my_list[i] [:, :, 0 ] , vmin=0, vmax=3 )
                        plt.axis('off');
                    
                    plt.tight_layout()
                    
                    fig.savefig(dav_dir + 'summary_csd.png')
                    
                    plt.close(fig)'''
                    
                    
                    
                    ###   ARD
                    
                    '''n_fib= 3
                    fibr_img= np.zeros( (sx, sy, sz, n_fib, 3) )
                    resp_img= np.zeros( (sx, sy, sz, n_fib) )
                    resp_csf= np.zeros( (sx, sy, sz) )
                    
                    separate_b0= True
                    
                    if separate_b0:
                        b0_ind= np.squeeze( np.where(b_vals==0) )
                        b1_ind= np.squeeze( np.where(b_vals>0) )
                        b0_img= d_img[:,:,:,b0_ind]
                        b1_img= d_img[:,:,:,b1_ind]
                        b0_img= np.mean(b0_img, axis=-1)[:,:,:,np.newaxis]
                        b_vals= b_vals[b1_ind]
                        b_vecs= b_vecs[:,b1_ind]
                    
                    mask= d_img[:,:,:,0]>20
                    
                    for ix in tqdm(range(sx), ascii=True):
                        for iy in range(sy):
                            for iz in range(slc,slc+1):
                                
                                if mask[ix, iy, iz]:
                                    
                                    if separate_b0:
                                        s= b1_img[ix, iy, iz,:]/b0_img[ix, iy, iz,0]
                                    else:
                                        s= d_img[ix, iy, iz,:]
                                    
                                    n_fib_init= 1
                                    R_init, bounds_lo, bounds_up = crl_dci.polar_fibers_and_iso_init(n_fib_init, lam_1=Lam[0], lam_2=Lam[1], d_iso=0.003)
                                    
                                    solution = opt.least_squares(crl_dci.polar_fibers_and_iso_resid, R_init,
                                                                                    bounds=(bounds_lo,bounds_up),
                                                                                    args=(n_fib_init, b_vals, b_vecs.T,
                                                                                    s,  s**0.0))
                                    
                                    s_pred= crl_dci.polar_fibers_and_iso_simulate(solution.x, n_fib_init, b_vals, b_vecs.T)
                                    
                                    #fibers= crl_dci.polar_fibers_from_solution(solution.x, n_fib_init)
                                    
                                    #responses= np.array(solution.x)[0:-1:5]
                                    
                                    R_inter, _, _ = crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=Lam[0], lam_2=Lam[1], d_iso=0.003)
                                    R_inter[1:5]= solution.x[1:5]
                                    R_inter[-1]= solution.x[-1]
                                    R_inter[5:-1:5]= 0.00
                                    
                                    sigma= np.std(s_pred-s)*0.5
                                    
                                    R_MCMC, prob_track= crl_dci.MCMC(R_inter, b_vals, b_vecs.T, s, n_fib, sigma, step_f= 0.03, step_lam= 0.00001, step_ang= 0.2, step_sigma= 0.0001*sigma, \
                                                         aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=5000, N_aneal=1000, sample_stride=1)
                                    
                                    fibers= crl_dci.polar_fibers_from_solution(R_MCMC, n_fib)
                                    
                                    responses= np.array(R_MCMC)[0:-1:5]
                                    
                                    fibr_img[ix, iy, iz,:,:]= fibers.T
                                    resp_img[ix, iy, iz,:]= np.sort(responses)[::-1]
                    
                        
                        threshold= 0.20
                        
                        one_mask=   resp_csf<0.75
                        two_mask=   resp_img[:,:,:,1] >threshold
                        two_mask=   np.logical_and( one_mask, two_mask )
                        thr_mask=   resp_img[:,:,:,2] >threshold
                        thr_mask=   np.logical_and( thr_mask, two_mask )
                        
                        mose_ard= np.zeros((sx,sy,sz))
                        mose_ard[one_mask]= 1
                        mose_ard[two_mask]= 2
                        mose_ard[thr_mask]= 3'''
                        
                    
                    
                    
                    ###    Tracking 1
                    
                    csa_model = CsaOdfModel(gtab, sh_order=6)
                    
                    mask= np.logical_and(brain_mask, d_img[:,:,:,0]>20) #d_img[:,:,:,0]>0   # FA>0.2
                    
                    csa_peaks = peaks_from_model(csa_model, d_img, default_sphere,
                                                 relative_peak_threshold=.8,
                                                 min_separation_angle=25,
                                                 mask=mask)
                    
#                    ren = window.Renderer()
#                    ren.add(actor.peak_slicer(csa_peaks.peak_dirs,
#                                              csa_peaks.peak_values,
#                                              colors=None))
#                    window.record(ren, out_path= dav_dir + 'csa_direction_field.png', size=(900, 900))
#                    if True:
#                        window.show(ren, size=(800, 800))
                    white_matter= ts_img==3 #(FA>0.10) * (1-skull)
                    
                    stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25)
                    #stopping_criterion = BinaryStoppingCriterion(white_matter)
                    
#                    sli = slc#csa_peaks.gfa.shape[2] // 2
#                    plt.figure()
#                    plt.subplot(1, 2, 1).set_axis_off()
#                    plt.imshow(csa_peaks.gfa[:, :, sli].T, cmap='gray', origin='lower')
#                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
#                    save_trk(sft, dav_dir+"tractogram_deterministic2.trk")
#                    plt.subplot(1, 2, 2).set_axis_off()
#                    plt.imshow((csa_peaks.gfa[:, :, sli] > 0.25).T, cmap='gray', origin='lower')
                    
                    #plt.savefig('gfa_tracking_mask.png')
                    
                    '''CC_mask= FA>0.15
                    seed_mask = np.zeros(d_img.shape[:3])
                    seed_mask[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max]= \
                        CC_mask[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max]'''
                    
                    seed_mask= FA>0.20
                    seed_mask= seed_mask *  (1-skull)
                    
#                    affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
#                                         [0.0, 1.0, 0.0, 0.0],
#                                         [0.0, 0.0, 1.0, 0.0],
#                                         [0.0, 0.0, 0.0, 1.0],] )
                    FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
                    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
                    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
                    affine= FA_img_nii.affine
                    
                    seeds = utils.seeds_from_mask(seed_mask, affine, density=[2, 2, 2])
                    
                    streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                                          affine=affine, step_size=.5)
                    
                    streamlines = Streamlines(streamlines_generator)
                    
#                    color = colormap.line_colors(streamlines)
#                    streamlines_actor = actor.line(streamlines,
#                                                   colormap.line_colors(streamlines))
#                    r = window.Renderer()
#                    r.add(streamlines_actor)
#                    window.record(r, out_path='tractogram_EuDX.png', size=(800, 800))
#                    if True:
#                        window.show(r)
                    
                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                    
                    save_trk(sft, dav_dir+"tractogram_probabilistic_dg_sh2.trk")
                    
                    
                    
                    
                    
                    ####   Tracking 2
                    
                    FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
                    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
                    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
                    #affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
                    #                     [0.0, 1.0, 0.0, 0.0],
                    #                     [0.0, 0.0, 1.0, 0.0],
                    #                     [0.0, 0.0, 0.0, 1.0],] )
                    affine= FA_img_nii.affine
                    
                    #CC_mask= FA>0.30
                    #seed_mask = np.zeros(d_img.shape[:3])
                    #seed_mask[65:95, 100:170, 155:185 ]= CC_mask[65:95, 100:170, 155:185 ]
                    seed_mask= FA>0.20
                    seed_mask= seed_mask *  (1-skull)
                    
                    white_matter=  (FA>0.20) * (1-skull) # ts_img==3
                    
                    seeds = utils.seeds_from_mask(seed_mask, affine, density=2)
                    
                    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
                    csd_fit = csd_model.fit(d_img, mask=d_img[:,:,:,0]>20)
                    
                    csa_model = CsaOdfModel(gtab, sh_order=6)
                    gfa = csa_model.fit(d_img, mask=d_img[:,:,:,0]>20).gfa
                    stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
#                    stopping_criterion = BinaryStoppingCriterion(white_matter)
                    
                    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
                        csd_fit.shm_coeff, max_angle=30., sphere=default_sphere)
                    streamline_generator = LocalTracking(detmax_dg, stopping_criterion, seeds,
                                                         affine, step_size=.5)
                    streamlines = Streamlines(streamline_generator)
                    
                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                    save_trk(sft, dav_dir+"tractogram_deterministic_dg_sh.trk")
                    
                    
                    
                    ####   Tracking 3
                    
                    FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
                    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
                    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
                    
                    #affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
                    #                     [0.0, 1.0, 0.0, 0.0],
                    #                     [0.0, 0.0, 1.0, 0.0],
                    #                     [0.0, 0.0, 0.0, 1.0],] )
                    affine= FA_img_nii.affine
                    
                    #CC_mask= FA>0.30
                    #seed_mask = np.zeros(d_img.shape[:3])
                    #seed_mask[65:95, 100:170, 155:185 ]= CC_mask[65:95, 100:170, 155:185 ]
                    seed_mask= FA>0.15
                    seed_mask= seed_mask *  (1-skull)
                    
                    white_matter= ts_img==3 #(FA>0.20) * (1-skull)
                    
                    seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
                    
                    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
                    csd_fit = csd_model.fit(d_img, mask=white_matter)
                    
                    csa_model = CsaOdfModel(gtab, sh_order=6)
                    gfa = csa_model.fit(d_img, mask=white_matter).gfa
                    #stopping_criterion = ThresholdStoppingCriterion(gfa, .20)
                    stopping_criterion = BinaryStoppingCriterion(white_matter)
                    
                    fod = csd_fit.odf(small_sphere)
                    pmf = fod.clip(min=0)
                    
                    #######
                    option=1
                    if option==1:
                        prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                                    sphere=small_sphere)
                    elif option==2:
                        prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                                        max_angle=30.,
                                                                        sphere=default_sphere)
                    elif option==3:
                        peaks = peaks_from_model(csd_model, d_img, default_sphere, .5, 25,
                                             mask=white_matter, return_sh=True, parallel=True)
                        fod_coeff = peaks.shm_coeff
                        prob_dg = ProbabilisticDirectionGetter.from_shcoeff(fod_coeff, max_angle=30.,
                                                                        sphere=default_sphere)
                    #######
                    streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                                         affine, step_size=.5)
                    
                    streamlines = Streamlines(streamline_generator)
                    
                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                    save_trk(sft, dav_dir+"tractogram_probabilistic_dg_pmf.trk")
                    
                    
                    
                    
                    #  Tracking 4
                    
                    '''FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
                    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
                    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
                    
                    #affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
                    #                     [0.0, 1.0, 0.0, 0.0],
                    #                     [0.0, 0.0, 1.0, 0.0],
                    #                     [0.0, 0.0, 0.0, 1.0],] )
                    affine= FA_img_nii.affine
                    
                    #CC_mask= FA>0.30
                    #seed_mask = np.zeros(d_img.shape[:3])
                    #seed_mask[65:95, 100:170, 155:185 ]= CC_mask[65:95, 100:170, 155:185 ]
                    seed_mask= FA>0.20
                    seed_mask= seed_mask *  (1-skull)
                    
                    white_matter= (FA>0.20) * (1-skull)
                    
                    seeds = utils.seeds_from_mask(seed_mask, affine, density=[1,1,1])
                    
                    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
                    csd_fit = csd_model.fit(d_img, mask=white_matter)
                    
                    csa_model = CsaOdfModel(gtab, sh_order=6)
                    gfa = csa_model.fit(d_img, mask=white_matter).gfa
                    stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
                    
                    
                    
                    pmf = csd_fit.odf(small_sphere).clip(min=0)
                    peak_dg = ClosestPeakDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                                  sphere=small_sphere)
                    peak_streamline_generator = LocalTracking(peak_dg, stopping_criterion, seeds,
                                                              affine, step_size=.5)
                    
                    streamlines = Streamlines(peak_streamline_generator)
                    
                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                    save_trk(sft, dav_dir+"closest_peak_dg_CSD.trk")
                    '''
                    
                    
                    
                    #  Tracking 4
                    '''
                    FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
                    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
                    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
                    
                    #affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
                    #                     [0.0, 1.0, 0.0, 0.0],
                    #                     [0.0, 0.0, 1.0, 0.0],
                    #                     [0.0, 0.0, 0.0, 1.0],] )
                    affine= FA_img_nii.affine
                    
                    #CC_mask= FA>0.30
                    #seed_mask = np.zeros(d_img.shape[:3])
                    #seed_mask[65:95, 100:170, 155:185 ]= CC_mask[65:95, 100:170, 155:185 ]
                    seed_mask= FA>0.20
                    seed_mask= seed_mask *  (1-skull)
                    
                    white_matter= (FA>0.20) * (1-skull)
                    
                    seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
                    
                    sphere = get_sphere()
                    
                    sf_model = sfm.SparseFascicleModel(gtab, sphere=sphere,
                                                       l1_ratio=0.5, alpha=0.001,
                                                       response=response[0])
                    
                    pnm = peaks_from_model(sf_model, d_img, sphere,
                                           relative_peak_threshold=.5,
                                           min_separation_angle=25,
                                           mask=white_matter,
                                           parallel=True)
                    
                    stopping_criterion = ThresholdStoppingCriterion(pnm.gfa, .25)
                    
                    streamline_generator = LocalTracking(pnm, stopping_criterion, seeds, affine,
                                                         step_size=.5)
                    
                    streamlines = Streamlines(streamline_generator)
                    
                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                    save_trk(sft, dav_dir+"tractogram_deterministic_sfm.trk")
                    '''
                    
                    
                    
                    #   RecoBundles
                    
                    target_file= dav_dir+"tractogram_probabilistic_dg_pmf.trk"
                    
                    #atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()
                    atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
                    
                    sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=False)
                    atlas = sft_atlas.streamlines
                    
                    atlas_header = create_tractogram_header(atlas_file,
                                                            *sft_atlas.space_attribute)
                    
                    sft_target = load_trk(target_file, "same", bbox_valid_check=False)
                    target = sft_target.streamlines
                    target_header = create_tractogram_header(atlas_file,
                                                             *sft_atlas.space_attribute)
                    
                    moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
                                        atlas, target, x0='affine', verbose=True, progressive=True)
                    
                    model_af_l_file, model_cst_l_file = get_two_hcp842_bundles()
                    
                    sft_af_l = load_trk(model_af_l_file, "same", bbox_valid_check=False)
                    model_af_l = sft_af_l.streamlines
                    
                    rb = RecoBundles(moved, verbose=True)
                    
                    recognized_af_l, af_l_labels = rb.recognize(model_bundle=model_af_l,
                                            model_clust_thr=5.,
                                            reduction_thr=10,
                                            reduction_distance='mam',
                                            slr=True,
                                            slr_metric='asymmetric',
                                            pruning_distance='mam')
                    
                    
                    
                    









'''
###    CC
                    
#h5f = h5py.File(res_dir + 'cc_all.h5','w')
#h5f['CC_stats']= CC_stats
#h5f.close()

h5f = h5py.File(res_dir +  'cc_2600.h5', 'r')
CC_stats_2600 = h5f['CC_stats'][:]
h5f.close()

CC= np.concatenate( (CC_stats_all, CC_stats_400, CC_stats_1000, CC_stats_2600, CC_stats_p5, CC_stats_p75 ), axis=1)

CC= CC[:81,:]

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 16))
i_fig = 0

for i in range(12):
    
    data_to_plot= [ CC[:, i], CC[:, 12+i], CC[:, 24+i], CC[:, 36+i], CC[:, 48+i], CC[:, 60+i]]
    
    ax[i%3,i//3].violinplot(data_to_plot)
    ax[i%3,i//3].set_xticklabels([' ', 'All', 'b=400', 'b=1000', 'b=2600','remove 50 %', 'remove 75%'])

for i in range(12):
    
    data_to_plot= [ CC[:, i], CC[:, 12+i], CC[:, 24+i], CC[:, 36+i], CC[:, 48+i], CC[:, 60+i]]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.violinplot(data_to_plot)
    ax.set_xticklabels([' ', 'All', 'b=400', 'b=1000', 'b=2600','remove 50 %', 'remove 75%'])
'''




































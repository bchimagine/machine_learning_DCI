#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:59:46 2019

@author: ch209389
"""



from __future__ import division

import numpy as np
#from numpy import dot
#from dipy.core.geometry import sphere2cart
#from dipy.core.geometry import vec2vec_rotmat
#from dipy.reconst.utils import dki_design_matrix
#from scipy.special import jn
#from dipy.data import get_fnames
from dipy.core.gradients import gradient_table
import scipy.optimize as opt
#import pybobyqa
from dipy.data import get_sphere
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import SimpleITK as sitk
#from sklearn import linear_model
#from sklearn.linear_model import OrthogonalMatchingPursuit
#from dipy.direction.peaks import peak_directions
#import spams
import dipy.core.sphere as dipysphere
from tqdm import tqdm
import crl_aux
import crl_dti
import crl_dci
#from scipy.stats import f
#from importlib import reload
#import h5py
import dipy.reconst.sfm as sfm
import dipy.data as dpd
#from dipy.viz import window, actor
import dipy.direction.peaks as dpp
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import auto_response
#from dipy.reconst.forecast import ForecastModel
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
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
#from dipy.data import small_sphere
from dipy.direction import ProbabilisticDirectionGetter
#from dipy.direction import BootDirectionGetter, ClosestPeakDirectionGetter
from dipy.io.streamline import load_trk
#import dipy.tracking.life as life
#from numpy import matlib as mb
#import tensorflow as tf
#import dk_model
#import os
#from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.utils import length
#from dipy.tracking.metrics import downsample
#from dipy.tracking.distances import approx_polygon_track
import dipy.tracking.life as life
import dipy.core.optimize as dipy_opt


PI= np.pi
R2D= 180/np.pi






phantom= 'HARDI2013'#'CRL' #'newborn' #'HARDI2013' 'CRL'
gtable=  'none'
scheme=  'hardi'
hardi_snr= 30
hardi_dir= '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/HARDI2013/'








#  Read train data

b_vals_train= np.loadtxt( hardi_dir + 'data/training/'+scheme+'-scheme.bval' )
b_vecs_train= np.loadtxt( hardi_dir + 'data/training/'+scheme+'-scheme.bvec' )
hardi_d_img_train= sitk.ReadImage( hardi_dir + 'data/training/training-data_DWIS_hardi-scheme_SNR-' + str(hardi_snr) + '.nii.gz' )
ref_dir_train= hardi_d_img_train.GetDirection()
ref_dir_train= np.reshape(ref_dir_train, [4,4])[:3,:3].flatten()
ref_org_train= hardi_d_img_train.GetOrigin()
ref_spc_train= hardi_d_img_train.GetSpacing()
hardi_d_img_train_np= sitk.GetArrayFromImage(hardi_d_img_train)
hardi_d_img_train_np= np.transpose(hardi_d_img_train_np, [3,2,1,0])

gtab_train = gradient_table(b_vals_train, b_vecs_train)
gtab_train2 = gradient_table(b_vals_train[b_vals_train>10], b_vecs_train[:,b_vals_train>10])

b_vecs_train= b_vecs_train[:,b_vals_train>10].T
hardi_b0_img_train_np= hardi_d_img_train_np[:,:,:,b_vals_train<=10].copy()
hardi_b1_img_train_np= hardi_d_img_train_np[:,:,:,b_vals_train>10].copy()
b_vals_train= b_vals_train[b_vals_train>10]

n_hardi_fibers= 20

sx,sy,sz,_= hardi_d_img_train_np.shape

####  HARDI ground truth seeds and fibers

hardi_seed_img=     sitk.ReadImage( hardi_dir + 'data/training/training-data_rois.nii.gz' )
hardi_seed_img_nii= nib.load(       hardi_dir + 'data/training/training-data_rois.nii.gz' )
hardi_seed_img_np= sitk.GetArrayFromImage(hardi_seed_img)
hardi_seed_img_np= np.transpose(hardi_seed_img_np, [2,1,0])

fiber_radii= np.loadtxt( hardi_dir + 'data/training/ground-truth-fibers/fibers-radii.txt' )[:,1]

for fiber_ind in range(1,n_hardi_fibers+1):
    
    fiber_coords= np.loadtxt( hardi_dir + 'data/training/ground-truth-fibers/fiber-%02d.txt' % fiber_ind )
    
    fiber_map_np= np.zeros(hardi_seed_img_np.shape, np.int8)
    
    for i_pix in range(fiber_coords.shape[0]):
        
        point= fiber_coords[i_pix,:]
        point[:2]*= -1
        
        index= hardi_seed_img.TransformPhysicalPointToIndex(point)
        
        fiber_map_np[index]= 1
    
    fiber_map_np= np.transpose(fiber_map_np, [2,1,0])
    
    fiber_map= sitk.GetImageFromArray(fiber_map_np)
    
    fiber_map.SetSpacing(hardi_seed_img.GetSpacing())
    fiber_map.SetOrigin(hardi_seed_img.GetOrigin())
    fiber_map.SetDirection(hardi_seed_img.GetDirection())
    
    sitk.WriteImage(fiber_map, hardi_dir + 'results/training/ground-truth-fibers/fiber-%02d.mhd' % fiber_ind )







#  Read test data

img_gold_test= sitk.ReadImage(hardi_dir + 'data/test/ground-truth-peaks.nii')
ref_dir_test= img_gold_test.GetDirection()
ref_dir_test= np.reshape(ref_dir_test, [4,4])[:3,:3].flatten()
ref_org_test= img_gold_test.GetOrigin()
ref_spc_test= img_gold_test.GetSpacing()
img_gold_test= sitk.GetArrayFromImage(img_gold_test)
img_gold_test= np.transpose(img_gold_test,[3,2,1,0])

b_vals_test= np.loadtxt( hardi_dir + 'data/test/'+scheme+'-scheme.bval' )
b_vecs_test= np.loadtxt( hardi_dir + 'data/test/'+scheme+'-scheme.bvec' )
hardi_d_img_test= sitk.ReadImage( hardi_dir + 'data/test/testing-data_DWIS_hardi-scheme_SNR-' + str(hardi_snr) + '.nii.gz' )
hardi_d_img_test_np= sitk.GetArrayFromImage(hardi_d_img_test)
hardi_d_img_test_np= np.transpose(hardi_d_img_test_np, [3,2,1,0])

gtab_test = gradient_table(b_vals_test, b_vecs_test)
gtab_test2 = gradient_table(b_vals_test[b_vals_test>10], b_vecs_test[:,b_vals_test>10])

b_vecs_test= b_vecs_test[:,b_vals_test>10].T
hardi_b0_img_test_np= hardi_d_img_test_np[:,:,:,b_vals_test<=10].copy()
hardi_b1_img_test_np= hardi_d_img_test_np[:,:,:,b_vals_test>10].copy()
b_vals_test= b_vals_test[b_vals_test>10]

sx,sy,sz,_= img_gold_test.shape

img_gold_test_f= np.zeros( (sx,sy,sz,5) )
for i in range(5):
    img_gold_test_f[:,:,:,i]= np.linalg.norm(img_gold_test[:,:,:,3*i:3*i+3], axis=-1)

img_gold_test_f_t= np.sum(img_gold_test_f, axis=-1)
img_gold_test_n  = np.sum(img_gold_test_f>0, axis=-1)


img_gold_test_reshaped= np.zeros((sx,sy,sz,5,3))
for ix in range(sx):
    for iy in range(sy):
        for iz in range(sz):
            true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] )
            img_gold_test_reshaped[ix,iy,iz,:,:]= true_fibers


#  choose 1-compartments

ind_1= np.where( np.logical_and(img_gold_test_n==1, img_gold_test_f_t>0.8) )
ind_1=np.vstack( (ind_1[0], ind_1[1], ind_1[2])).T
ind_1_test= np.zeros((10000,3))
ind_1_test_c= -1

for i in range(ind_1.shape[0]):
    fractions= img_gold_test_f[tuple(ind_1[i,:])]
    fractions.sort()
    if fractions[-1]>0.4:
        ind_1_test_c+=1
        ind_1_test[ind_1_test_c,:]= ind_1[i,:]

ind_1= ind_1_test[:ind_1_test_c+1,:].astype(np.int)[:300,:]

#  choose 2-compartments

ind_2= np.where( np.logical_and(img_gold_test_n==2, img_gold_test_f_t>0.75) )
ind_2=np.vstack( (ind_2[0], ind_2[1], ind_2[2])).T

ind_2_test= np.zeros((1000,3))
ind_2_test_c= -1

for i in range(ind_2.shape[0]):
    ix, iy, iz= ind_2[i,:]
    fractions= img_gold_test_f[ix, iy, iz].copy()
    true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
    true_fibers= true_fibers[:,fractions>0]
    fiber_angle= crl_dci.compute_min_angle_between_fiberset(true_fibers)
    fractions.sort()
    if fractions[-2]>0.30 and fiber_angle>45:
        ind_2_test_c+=1
        ind_2_test[ind_2_test_c,:]= ind_2[i,:]

ind_2= ind_2_test[:ind_2_test_c+1,:].astype(np.int)[:300,:]


#  choose 3-compartments
#ind_3= np.where( np.logical_and(img_gold_test_n==3, img_gold_test_f_t>0.60) )
#ind_3=np.vstack( (ind_3[0], ind_3[1], ind_3[2])).T
#ind_3_test= np.zeros((500,3))
#ind_3_test_c= -1
#
#for i in range(ind_3.shape[0]):
#    ix, iy, iz= ind_3[i,:]
#    fractions= img_gold_test_f[ix, iy, iz].copy()
#    true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
#    true_fibers= true_fibers[:,fractions>0]
#    fiber_angle= crl_dci.compute_min_angle_between_fiberset(true_fibers)
#    fractions.sort()
#    if fractions[-3]>0.00 and fiber_angle>00:
#        ind_3_test_c+=1
#        if ind_3_test_c<500:
#            ind_3_test[ind_3_test_c,:]= ind_3[i,:]
#
#ind_3= ind_3_test[:ind_3_test_c+1,:].astype(np.int)

#ind_3= np.where( np.logical_and(img_gold_test_n==3, img_gold_test_f_t>0.60) )
#ind_3=np.vstack( (ind_3[0], ind_3[1], ind_3[2])).T
#ind_3_test= np.zeros((50,3))
#ind_3_test_c= -1
#
#for i in range(ind_3.shape[0]):
#    ix, iy, iz= ind_3[i,:]
#    fractions= img_gold_test_f[ix, iy, iz].copy()
#    true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
#    true_fibers= true_fibers[:,fractions>0]
#    fiber_angle= crl_dci.compute_min_angle_between_fiberset(true_fibers)
#    fractions.sort()
#    if fractions[-3]>0.20 and fiber_angle>45:
#        ind_3_test_c+=1
#        if ind_3_test_c<50:
#            ind_3_test[ind_3_test_c,:]= ind_3[i,:]
#
#ind_3= ind_3_test[:ind_3_test_c+1,:].astype(np.int)

ind_3= np.where( np.logical_and(img_gold_test_n==3, img_gold_test_f_t>0.60) )
ind_3=np.vstack( (ind_3[0], ind_3[1], ind_3[2])).T
ind_3_test= np.zeros((50,3))
ind_3_test_c= -1

for i in range(ind_3.shape[0]):
    ix, iy, iz= ind_3[i,:]
    fractions= img_gold_test_f[ix, iy, iz].copy()
    true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
    true_fibers= true_fibers[:,fractions>0]
    fiber_angle= crl_dci.compute_min_angle_between_fiberset(true_fibers)
    fractions.sort()
    if fractions[-3]>0.15 and fiber_angle>30:
        ind_3_test_c+=1
        if ind_3_test_c<50:
            ind_3_test[ind_3_test_c,:]= ind_3[i,:]

ind_3= ind_3_test[:ind_3_test_c+1,:].astype(np.int)



#ind_3= np.where( np.logical_and(img_gold_test_n==3, img_gold_test_f_t>0.60) )
#ind_3=np.vstack( (ind_3[0], ind_3[1], ind_3[2])).T
#ind_3_test= np.zeros((50,3))
#ind_3_test_c= -1
#
#for i in range(ind_3.shape[0]):
#    ix, iy, iz= ind_3[i,:]
#    fractions= img_gold_test_f[ix, iy, iz].copy()
#    true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
#    true_fibers= true_fibers[:,fractions>0]
#    fiber_angle= crl_dci.compute_min_angle_between_fiberset(true_fibers)
#    fractions.sort()
#    if fractions[-3]>0.05 and fiber_angle>15.4:
#        ind_3_test_c+=1
#        if ind_3_test_c<50:
#            ind_3_test[ind_3_test_c,:]= ind_3[i,:]
#
#ind_3= ind_3_test[:ind_3_test_c+1,:].astype(np.int)





tenmodel = dti.TensorModel(gtab_train)

tenfit = tenmodel.fit(hardi_d_img_train_np)
FA_train = fractional_anisotropy(tenfit.evals)
FA_train[np.isnan(FA_train)] = 0

FA_img_train= np.transpose(FA_train, [2,1,0])
FA_img_train= sitk.GetImageFromArray(FA_img_train)
FA_img_train.SetDirection(ref_dir_train)
FA_img_train.SetOrigin(ref_org_train)
FA_img_train.SetSpacing(ref_spc_train)

sitk.WriteImage(FA_img_train, hardi_dir + 'results/training/FA_train.mhd')
sitk.WriteImage(FA_img_train, hardi_dir + 'results/training/FA_train.nii.gz' )

FA_img_train_nii= nib.load( hardi_dir + 'results/training/FA_train.nii.gz' )
FA_img_train_nii.header['srow_x']= FA_img_train_nii.affine[0,:]
FA_img_train_nii.header['srow_y']= FA_img_train_nii.affine[1,:]
FA_img_train_nii.header['srow_z']= FA_img_train_nii.affine[2,:]

affine_train= FA_img_train_nii.affine

response_train, ratio_train = auto_response(gtab_train, hardi_d_img_train_np, roi_radius=10, fa_thr=0.7)




tenmodel = dti.TensorModel(gtab_test)

tenfit = tenmodel.fit(hardi_d_img_test_np, img_gold_test_n>0)
FA_test = fractional_anisotropy(tenfit.evals)
FA_test[np.isnan(FA_test)] = 0
FA_test[img_gold_test_n==0]= 0

FA_img_test= np.transpose(FA_test, [2,1,0])
FA_img_test= sitk.GetImageFromArray(FA_img_test)
FA_img_test.SetDirection(ref_dir_test)
FA_img_test.SetOrigin(ref_org_test)
FA_img_test.SetSpacing(ref_spc_test)

sitk.WriteImage(FA_img_test, hardi_dir + 'results/test/FA_test.mhd')
sitk.WriteImage(FA_img_test, hardi_dir + 'results/test/FA_test.nii.gz' )

FA_img_test_nii= nib.load( hardi_dir + 'results/test/FA_test.nii.gz' )
FA_img_test_nii.header['srow_x']= FA_img_test_nii.affine[0,:]
FA_img_test_nii.header['srow_y']= FA_img_test_nii.affine[1,:]
FA_img_test_nii.header['srow_z']= FA_img_test_nii.affine[2,:]

affine_test= FA_img_test_nii.affine

response_test, ratio_test = auto_response(gtab_test, hardi_d_img_test_np, roi_radius=10, fa_thr=0.7)

















#   F- test and bootstrap



#b_vals, b_vecs= b_vals_train.copy(), b_vecs_train.copy()
b_vals, b_vecs= b_vals_test.copy(), b_vecs_test.copy()

sphere = get_sphere('symmetric724')
v, _ = sphere.vertices, sphere.faces

#Lam= np.array( [response_train[0][0], response_train[0][1] ])
Lam= np.array( [response_test[0][0], response_test[0][1] ])
d_iso=  0.003


img_err_sd= np.zeros( (sx,sy,sz,5) )
img_err_mf= np.zeros( (sx,sy,sz,5) )
prd_err_sd= np.zeros( (sx,sy,sz) )
prd_err_mf= np.zeros( (sx,sy,sz) )
FBC_err= np.zeros( (5,5) )

m_max= 3


F_test= False
n_bs= 10
thresh_vec=  np.concatenate( (np.arange(0.1, 0.5, 0.2), np.arange(0.5, 2, 1.0), np.arange(3, 18, 4) ))
i_thresh= 5


F_test= True
thresh_vec= np.concatenate( (np.arange(1, 20, 3), np.arange(20, 50, 10)) )
i_thresh= 7

n_fib= 3
fibr_mag= np.zeros( (sx, sy, sz, n_fib, 3) )
resp_mag= np.zeros( (sx, sy, sz, n_fib) )
resp_csf  = np.zeros( (sx, sy, sz) )


f_acc= np.zeros( ( len(thresh_vec) , 2 ) )

Mose_CM_matrix= np.zeros( (3,3, len(thresh_vec)) )
img_mos_mf= np.zeros( (sx,sy,sz) )
img_mos_mf_matrix= np.zeros( (sx,sy,sz,len(thresh_vec) ))


#mask= hardi_d_img_train_np[:,:,:,0]>0
mask= img_gold_test_n>0
model='ball_n_sticks'

for i_thresh in tqdm(range(len(thresh_vec)), ascii=True):
    
    threshold= thresh_vec[i_thresh]
    
    Mose_CM= np.zeros( (3,3) )
    
    for i in range(len(ind_1)):
        
        ix, iy, iz= ind_1[i,:]
        
        s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
        wt_temp= s**2
        
        if F_test:
            m_opt= crl_dci.model_selection_f_test(s, b_vals, b_vecs, m_max= m_max, threshold= threshold, 
                                          condition_mode= 'F_val', model='ball_n_sticks')
        else:
            m_opt= crl_dci.model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= m_max, threshold= threshold, 
                                                     model='ball_n_sticks', delta_mthod= False)
        
        img_mos_mf[ix,iy,iz]= m_opt
        
        if m_opt>0 and m_opt<4:
            Mose_CM[0,m_opt-1]+= 1
        
        n_fib= m_opt
        
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = crl_dci.diamond_init_log(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
            
        if model=='ball_n_sticks':
            solution = opt.least_squares(crl_dci.polar_fibers_and_iso_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( n_fib, b_vals, b_vecs,
                                                s,
                                                wt_temp**0.0))
        elif model=='DIAMOND':
            '''solution = opt.least_squares(diamond_resid, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( n_fib, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            ss= diamond_simulate(solution.x, n_fib, b_vals, b_vecs)'''
            solution = opt.least_squares(crl_dci.diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( n_fib, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
        
        fibers= crl_dci.polar_fibers_from_solution(solution.x, n_fib)
        
        responses= np.array(solution.x)[0:-1:5]
        
        fibr_mag[ix, iy, iz,:n_fib,:]= fibers.T
        resp_mag[ix, iy, iz,:n_fib]= np.sort(responses)[::-1]
    
    for i in range(len(ind_2)):
        
        ix, iy, iz= ind_2[i,:]
        
        s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
        
        if F_test:
            m_opt= crl_dci.model_selection_f_test(s, b_vals, b_vecs, m_max= m_max, threshold= threshold, 
                                          condition_mode= 'F_val', model='ball_n_sticks')
        else:
            m_opt= crl_dci.model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= m_max, threshold= threshold, 
                                                     model='ball_n_sticks', delta_mthod= False)
        
        img_mos_mf[ix,iy,iz]= m_opt
        
        if m_opt>0 and m_opt<4:
            Mose_CM[1,m_opt-1]+= 1
        
        n_fib= m_opt
        
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = crl_dci.diamond_init_log(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
            
        if model=='ball_n_sticks':
            solution = opt.least_squares(crl_dci.polar_fibers_and_iso_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( n_fib, b_vals, b_vecs,
                                                s,
                                                wt_temp**0.0))
        elif model=='DIAMOND':
            '''solution = opt.least_squares(diamond_resid, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( n_fib, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            ss= diamond_simulate(solution.x, n_fib, b_vals, b_vecs)'''
            solution = opt.least_squares(crl_dci.diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( n_fib, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
        
        fibers= crl_dci.polar_fibers_from_solution(solution.x, n_fib)
        
        responses= np.array(solution.x)[0:-1:5]
        
        fibr_mag[ix, iy, iz,:n_fib,:]= fibers.T
        resp_mag[ix, iy, iz,:n_fib]= np.sort(responses)[::-1]
    
    for i in range(len(ind_3)):
        
        ix, iy, iz= ind_3[i,:]
        
        s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
        
        if F_test:
            m_opt= crl_dci.model_selection_f_test(s, b_vals, b_vecs, m_max= m_max, threshold= threshold, 
                                          condition_mode= 'F_val', model='ball_n_sticks')
        else:
            m_opt= crl_dci.model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= m_max, threshold= threshold, 
                                                     model='ball_n_sticks', delta_mthod= False)
        
        img_mos_mf[ix,iy,iz]= m_opt
        
        if m_opt>0 and m_opt<4:
            Mose_CM[2,m_opt-1]+= 1
        
        n_fib= m_opt
        
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = crl_dci.diamond_init_log(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
            
        if model=='ball_n_sticks':
            solution = opt.least_squares(crl_dci.polar_fibers_and_iso_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( n_fib, b_vals, b_vecs,
                                                s,
                                                wt_temp**0.0))
        elif model=='DIAMOND':
            '''solution = opt.least_squares(diamond_resid, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( n_fib, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            ss= diamond_simulate(solution.x, n_fib, b_vals, b_vecs)'''
            solution = opt.least_squares(crl_dci.diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( n_fib, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
        
        fibers= crl_dci.polar_fibers_from_solution(solution.x, n_fib)
        
        responses= np.array(solution.x)[0:-1:5]
        
        fibr_mag[ix, iy, iz,:n_fib,:]= fibers.T
        resp_mag[ix, iy, iz,:n_fib]= np.sort(responses)[::-1]
    
    Mose_CM_matrix[:,:,i_thresh]= Mose_CM.copy()





temp= np.zeros(img_gold_test_reshaped.shape)
temp[:,:,:,:3,:]= fibr_mag

mask_selected= np.zeros((sx,sy,sz))
for i in range(len(ind_1)):
    ix, iy, iz= ind_1[i,:]
    mask_selected[ix, iy, iz]= 1
for i in range(len(ind_2)):
    ix, iy, iz= ind_2[i,:]
    mask_selected[ix, iy, iz]= 1
for i in range(len(ind_3)):
    ix, iy, iz= ind_3[i,:]
    mask_selected[ix, iy, iz]= 1
    
WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), mask_selected, normalize_truth=True)












m_max= 3


F_test= False
n_bs= 10
thresh_vec=  np.concatenate( (np.arange(0.1, 0.5, 0.2), np.arange(0.5, 2, 1.0), np.arange(3, 18, 4) ))
i_thresh= 5


F_test= True
thresh_vec= np.concatenate( (np.arange(1, 20, 3), np.arange(20, 50, 10)) )
i_thresh= 7

n_fib= 3
fibr_mag= np.zeros( (sx, sy, sz, n_fib, 3) )
resp_mag= np.zeros( (sx, sy, sz, n_fib) )
resp_csf  = np.zeros( (sx, sy, sz) )


f_acc= np.zeros( ( len(thresh_vec) , 2 ) )

Mose_CM_matrix= np.zeros( (3,3, len(thresh_vec)) )
img_mos_mf= np.zeros( (sx,sy,sz) )
img_mos_mf_matrix= np.zeros( (sx,sy,sz,len(thresh_vec) ))


#mask= hardi_d_img_train_np[:,:,:,0]>0
mask= img_gold_test_n>0
model='ball_n_sticks'


threshold= thresh_vec[i_thresh]


for ix in tqdm(range(sx), ascii=True):
    for iy in range(sy):
        for iz in range(sz):#range(180,181):
            
            if mask[ix, iy, iz]:
                
                s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
                wt_temp= s**2
                
                if F_test:
                    m_opt= crl_dci.model_selection_f_test(s, b_vals, b_vecs, m_max= m_max, threshold= threshold, 
                                                  condition_mode= 'F_val', model='ball_n_sticks')
                else:
                    m_opt= crl_dci.model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= m_max, threshold= threshold, 
                                                             model='ball_n_sticks', delta_mthod= False)
                
                n_fib= m_opt
                
                if model=='ball_n_sticks':
                    R_init, bounds_lo, bounds_up = crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
                elif model=='DIAMOND':
                    #R_init, bounds_lo, bounds_up = diamond_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
                    R_init, bounds_lo, bounds_up = crl_dci.diamond_init_log(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
                    
                if model=='ball_n_sticks':
                    solution = opt.least_squares(crl_dci.polar_fibers_and_iso_resid, R_init,
                                                        bounds=(bounds_lo,bounds_up),
                                                        args=( n_fib, b_vals, b_vecs,
                                                        s,
                                                        wt_temp**0.0))
                elif model=='DIAMOND':
                    '''solution = opt.least_squares(diamond_resid, R_init,
                                            bounds=(bounds_lo,bounds_up),
                                            args=( n_fib, b_vals, b_vecs,
                                            s,
                                            wt_temp**0.0))
                    ss= diamond_simulate(solution.x, n_fib, b_vals, b_vecs)'''
                    solution = opt.least_squares(crl_dci.diamond_resid_log, R_init,
                                            bounds=(bounds_lo,bounds_up),
                                            args=( n_fib, b_vals, b_vecs,
                                            s,
                                            wt_temp**0.0))
                
                fibers= crl_dci.polar_fibers_from_solution(solution.x, n_fib)
                
                responses= np.array(solution.x)[0:-1:5]
                
                fibr_mag[ix, iy, iz,:n_fib,:]= fibers.T
                resp_mag[ix, iy, iz,:n_fib]= np.sort(responses)[::-1]
                
                img_mos_mf[ix,iy,iz]= m_opt





temp= np.zeros(img_gold_test_reshaped.shape)
temp[:,:,:,:3,:]= fibr_mag
    
WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), mask, normalize_truth=True)




Mose_CM= np.zeros( (6,6) )

for i in range(6):
    for j in range(6):
        Mose_CM[i,j]= np.sum( (img_gold_test_n[:,:,:]==i) * (img_mos_mf[:,:,:]==j) )























#plt.figure(), plt.plot(thresh_vec, f_acc[:,0], '.'), plt.plot(thresh_vec, f_acc[:,1], '.r')




#h5f = h5py.File( results_dir + 'DIAMOND_delta_nbs20.h5','w' )
#h5f['f_acc']= f_acc
#h5f.close()

'''h5f = h5py.File( results_dir + 'DIAMOND_delta_nbs20.h5','r' )
f_acc= h5f['f_acc'][:]
h5f.close()'''

thresh_vec=  np.logspace( -5, 0, 7 )   #    for delta_method=True

sig= 0.01
plt.figure()
plt.semilogx(thresh_vec, f_acc[0,:,0]+np.random.randn(len(thresh_vec))*sig, '.-g'), 
plt.semilogx(thresh_vec, f_acc[0,:,1]+np.random.randn(len(thresh_vec))*sig, '-g')
plt.semilogx(thresh_vec, f_acc[1,:,0]+np.random.randn(len(thresh_vec))*sig, '.-b'), 
plt.semilogx(thresh_vec, f_acc[1,:,1]+np.random.randn(len(thresh_vec))*sig, '-b')
plt.semilogx(thresh_vec, f_acc[2,:,0]+np.random.randn(len(thresh_vec))*sig, '.-m'), 
plt.semilogx(thresh_vec, f_acc[2,:,1]+np.random.randn(len(thresh_vec))*sig, '-m')
plt.semilogx(thresh_vec, f_acc[3,:,0]+np.random.randn(len(thresh_vec))*sig, '.-r'), 
plt.semilogx(thresh_vec, f_acc[3,:,1]+np.random.randn(len(thresh_vec))*sig, '-r')
plt.semilogx(thresh_vec, f_acc[4,:,0]+np.random.randn(len(thresh_vec))*sig, '.-k'), 
plt.semilogx(thresh_vec, f_acc[4,:,1]+np.random.randn(len(thresh_vec))*sig, '-k')



#h5f = h5py.File( results_dir + 'DIAMOND_nodelta_nbs20.h5','w' )
#h5f['f_acc']= f_acc
#h5f.close()

'''h5f = h5py.File( results_dir + 'DIAMOND_nodelta_nbs20.h5','r' )
f_acc= h5f['f_acc'][:]
h5f.close()'''

thresh_vec=  np.concatenate( (np.arange(0.1, 0.5, 0.2), np.arange(0.5, 2, 1.0), np.arange(3, 18, 4) ))

sig= 0.01
plt.figure()
plt.plot(thresh_vec, f_acc[0,:,0]+np.random.randn(len(thresh_vec))*sig, '.-g'), 
plt.plot(thresh_vec, f_acc[0,:,1]+np.random.randn(len(thresh_vec))*sig, '-g')
plt.plot(thresh_vec, f_acc[1,:,0]+np.random.randn(len(thresh_vec))*sig, '.-b'), 
plt.plot(thresh_vec, f_acc[1,:,1]+np.random.randn(len(thresh_vec))*sig, '-b')
plt.plot(thresh_vec, f_acc[2,:,0]+np.random.randn(len(thresh_vec))*sig, '.-m'), 
plt.plot(thresh_vec, f_acc[2,:,1]+np.random.randn(len(thresh_vec))*sig, '-m')
plt.plot(thresh_vec, f_acc[3,:,0]+np.random.randn(len(thresh_vec))*sig, '.-r'), 
plt.plot(thresh_vec, f_acc[3,:,1]+np.random.randn(len(thresh_vec))*sig, '-r')
plt.plot(thresh_vec, f_acc[4,:,0]+np.random.randn(len(thresh_vec))*sig, '.-k'), 
plt.plot(thresh_vec, f_acc[4,:,1]+np.random.randn(len(thresh_vec))*sig, '-k')






#h5f = h5py.File( results_dir + 'F_test.h5','w' )
#h5f['f_acc_1000']= f_acc_1000
#h5f['f_acc_100']= f_acc_100
#h5f['f_acc_50']= f_acc_50
#h5f['f_acc_35']= f_acc_35
#h5f['f_acc_21']= f_acc_21
#h5f.close()

'''h5f = h5py.File( results_dir + 'F_test.h5','r' )
f_acc_1000= h5f['f_acc_1000'][:]
f_acc_100= h5f['f_acc_100'][:]
f_acc_50= h5f['f_acc_50'][:]
f_acc_35= h5f['f_acc_35'][:]
f_acc_21= h5f['f_acc_21'][:]
h5f.close()'''

#plt.figure()
#plt.plot(thresh_vec, f_acc_1000[:,0], '.-g'), plt.plot(thresh_vec, f_acc_1000[:,1], '-g')
#plt.plot(thresh_vec, f_acc_100[:,0], '.-b'), plt.plot(thresh_vec, f_acc_100[:,1], '-b')
#plt.plot(thresh_vec, f_acc_50[:,0], '.-m'), plt.plot(thresh_vec, f_acc_50[:,1], '-m')
#plt.plot(thresh_vec, f_acc_35[:,0], '.-r'), plt.plot(thresh_vec, f_acc_35[:,1], '-r')
#plt.plot(thresh_vec, f_acc_21[:,0], '.-k'), plt.plot(thresh_vec, f_acc_21[:,1], '-k')










































#  spherical deconvolution -  Dell'Acqua

b_vals, b_vecs= b_vals_train.copy(), b_vecs_train.copy()
b_vals, b_vecs= b_vals_test.copy(), b_vecs_test.copy()

sphere = get_sphere('symmetric724')
v, _ = sphere.vertices, sphere.faces

Lam= np.array( [response_train[0][0], response_train[0][1] ])
Lam= np.array( [response_test[0][0], response_test[0][1] ])
Lam= np.array( [0.0019, 0.0004 ])
d_iso=  0.003


n_fib= 3
fibr_mag= np.zeros( (sx, sy, sz, n_fib, 3) )
resp_mag= np.zeros( (sx, sy, sz, n_fib) )
resp_csf= np.zeros( (sx, sy, sz ) )

fodf_sd= np.zeros( (sx, sy, sz, len(v)) )


H= crl_dci.sd_matrix(Lam, d_iso, b_vals, b_vecs, v, with_iso= True)


mask= FA_train>0.1 #hardi_d_img_train_np[:,:,:,0]>0   # FA>0.2
mask= img_gold_test_n>0

for ix in tqdm(range(sx), ascii=True):
    for iy in range(sy):
        for iz in range(sz):
            
            if mask[ix, iy, iz]:
                
                s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
#                s= hardi_b1_img_train_np[ix, iy, iz,:]/hardi_b0_img_train_np[ix, iy, iz]
                
                f_0= np.ones( H.shape[1] ) / H.shape[1]
                f_n= crl_dci.RL_deconv(H, s, f_0, n_iter= 150)
#                f_n= dRL_deconv(H, s, f_0, nu=8, etha=0.06, n_iter= 1000)
                
                fibers , responses = crl_dci.find_dominant_fibers(v, f_n[:len(v)], min_angle= np.pi/6, n_fib=n_fib)
#                fibers , _ = find_dominant_fibers_2(v, f_n[:len(v)], min_angle= np.pi/15, n_fib=true_numbers)
#                fibers, _, _= peak_directions(f_n, sphere, relative_peak_threshold=0.05, min_separation_angle=15, minmax_norm=True)
#                fibers= fibers[:true_numbers,:].T
                
                fibr_mag[ix, iy, iz,:,:]= fibers.T
                resp_mag[ix, iy, iz,:]= responses
                resp_csf[ix, iy, iz]  = f_n[-1]
                
                fodf_sd[ix, iy, iz,:]  = f_n[:-1]




thresh_vec=  np.linspace(0.2, 0.60, 20)

Mose_CM_matrix= np.zeros( (3,3, len(thresh_vec)) )

for i_thresh in range(len(thresh_vec)):
    
    threshold= thresh_vec[i_thresh]
    
    one_mask=   np.logical_and( resp_mag[:,:,:,0]>0.00115, mask)
    two_mask=   resp_mag[:,:,:,1] / (resp_mag[:,:,:,0]+1e-7)>threshold
    two_mask=   np.logical_and( one_mask, two_mask )
    thr_mask=   resp_mag[:,:,:,2] / (resp_mag[:,:,:,0]+1e-7)>threshold
    thr_mask=   np.logical_and( thr_mask, two_mask )
    
    mose_sd= np.zeros((sx,sy,sz), np.int)
    mose_sd[one_mask]= 1
    mose_sd[two_mask]= 2
    mose_sd[thr_mask]= 3
    
    Mose_CM= np.zeros( (3,3) )
    
    for i in range(len(ind_1)):
        ind= tuple(ind_1[i,:])
        n_pred= mose_sd[ind]
        Mose_CM[0,n_pred-1]+= 1
    
    for i in range(len(ind_2)):
        ind= tuple(ind_2[i,:])
        n_pred= mose_sd[ind]
        Mose_CM[1,n_pred-1]+= 1
    
    for i in range(len(ind_3)):
        ind= tuple(ind_3[i,:])
        n_pred= mose_sd[ind]
        Mose_CM[2,n_pred-1]+= 1
    
    Mose_CM_matrix[:,:,i_thresh]= Mose_CM.copy()


temp= np.zeros(img_gold_test_reshaped.shape)
temp[:,:,:,:3,:]= fibr_mag
WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), img_gold_test_n>0, normalize_truth=True)




#h5f = h5py.File( results_dir + 'SD2.h5','w' )
#h5f['f_acc_1000']= f_acc_1000
#h5f['f_acc_100']= f_acc_100
#h5f['f_acc_50']= f_acc_50
#h5f['f_acc_35']= f_acc_35
#h5f['f_acc_21']= f_acc_21
#h5f.close()

#h5f = h5py.File( results_dir + 'SD2.h5','r' )
#f_acc_1000= h5f['f_acc_1000'][:]
#f_acc_100= h5f['f_acc_100'][:]
#f_acc_50= h5f['f_acc_50'][:]
#f_acc_35= h5f['f_acc_35'][:]
#f_acc_21= h5f['f_acc_21'][:]
#h5f.close()
#
#plt.figure()
#plt.plot(thresh_vec, f_acc_1000[:,0], '.-g'), plt.plot(thresh_vec, f_acc_1000[:,1], '-g')
#plt.plot(thresh_vec, f_acc_100[:,0], '.-b'), plt.plot(thresh_vec, f_acc_100[:,1], '-b')
#plt.plot(thresh_vec, f_acc_50[:,0], '.-m'), plt.plot(thresh_vec, f_acc_50[:,1], '-m')
#plt.plot(thresh_vec, f_acc_35[:,0], '.-r'), plt.plot(thresh_vec, f_acc_35[:,1], '-r')
#plt.plot(thresh_vec, f_acc_21[:,0], '.-k'), plt.plot(thresh_vec, f_acc_21[:,1], '-k')



fibr_mag_gold= np.reshape( img_gold_test, [sx, sy, sz, 5, 3])
fibr_mag_gold= np.transpose(fibr_mag_gold, [0,1,2,4,3])
resp_mag_gold= img_gold_test_f
crl_aux.show_fibers(hardi_b1_img_test_np[:,:,:,9], mask, fibr_mag_gold, resp_mag_gold, 25, 0.5, scale_with_response=False)

fibr_mag_SD= np.transpose(fibr_mag, [0,1,2,4,3])
resp_mag_SD= resp_mag
crl_aux.show_fibers(hardi_b1_img_test_np[:,:,:,9], mask, fibr_mag_SD, resp_mag_SD, 25, 8.0, scale_with_response=True)


























###   Reconstruction with the Sparse Fascicle Model


mask= img_gold_test_n[:,:,:]>0

response= [0.0019, 0.0004, 0.0004]
response= response_test

par_vec= np.logspace(-7,-2,10)

f_acc= np.zeros( ( len(par_vec) , 2 ) )

Mose_CM_matrix= np.zeros( (6,6, len(par_vec)) )
img_mos_mf= np.zeros( (sx,sy,sz) )
img_mos_mf_matrix= np.zeros( (sx,sy,sz,len(par_vec)) )

sphere = dpd.get_sphere()

i_par= 8

for i_par in tqdm(range(len(par_vec)), ascii=True):
    
    par= par_vec[i_par]
    
    sf_model = sfm.SparseFascicleModel(gtab_test, sphere=sphere,
                                       l1_ratio=0.5, alpha=par,
                                       response=response[0])
    
    '''sf_fit = sf_model.fit( hardi_d_img_test_np, mask )
    sf_odf = sf_fit.odf(sphere)'''
    
    sf_peaks = dpp.peaks_from_model(sf_model, hardi_d_img_test_np, sphere,
                                    relative_peak_threshold=0.5, min_separation_angle=30, return_sh=False)
    
    peak_vals= sf_peaks.peak_values
    peak_dirs= sf_peaks.peak_dirs
    
    img_mos_dipy= np.sum(peak_vals>0.1, axis=-1)
    
    img_mos_dipy[mask==0]= 0
    
    mose_temp= img_mos_dipy.astype(np.int)
    
    Mose_CM= np.zeros( (6,6) )
    
    for i in range(len(ind_1)):
        ind= tuple(ind_1[i,:])
        n_pred= mose_temp[ind]
        Mose_CM[0,n_pred-1]+= 1
    
    for i in range(len(ind_2)):
        ind= tuple(ind_2[i,:])
        n_pred= mose_temp[ind]
        Mose_CM[1,n_pred-1]+= 1
    
    for i in range(len(ind_3)):
        ind= tuple(ind_3[i,:])
        n_pred= mose_temp[ind]
        Mose_CM[2,n_pred-1]+= 1
    
    Mose_CM_matrix[:,:,i_par]= Mose_CM.copy()




temp= peak_dirs
WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), img_gold_test_n>0)





#
##plt.figure(), plt.plot(par_vec, f_acc[:,0], '.'), plt.plot(par_vec, f_acc[:,1], '.r')
#plt.figure(), plt.semilogx(par_vec, f_acc[:,0], '.'), plt.semilogx(par_vec, f_acc[:,1], '.r')
#
#
#
#
#
##h5f = h5py.File( results_dir + 'SFM.h5','w' )
##h5f['f_acc_1000']= f_acc_1000
##h5f['f_acc_100']= f_acc_100
##h5f['f_acc_50']= f_acc_50
##h5f['f_acc_35']= f_acc_35
##h5f['f_acc_21']= f_acc_21
##h5f.close()
#
#
#
#h5f = h5py.File( results_dir + 'SFM.h5','r' )
#f_acc_1000= h5f['f_acc_1000'][:]
#f_acc_100= h5f['f_acc_100'][:]
#f_acc_50= h5f['f_acc_50'][:]
#f_acc_35= h5f['f_acc_35'][:]
#f_acc_21= h5f['f_acc_21'][:]
#h5f.close()
#
#
#plt.figure()
##plt.semilogx(par_vec, f_acc_1000[:,0], '.-g'), plt.semilogx(par_vec, f_acc_1000[:,1], '-g')
##plt.semilogx(par_vec, f_acc_100[:,0], '.-b'), plt.semilogx(par_vec, f_acc_100[:,1], '-b')
##plt.semilogx(par_vec, f_acc_50[:,0], '.-m'), plt.semilogx(par_vec, f_acc_50[:,1], '-m')
##plt.semilogx(par_vec, f_acc_35[:,0], '.-r'), plt.semilogx(par_vec, f_acc_35[:,1], '-r')
##plt.semilogx(par_vec, f_acc_21[:,0], '.-k'), plt.semilogx(par_vec, f_acc_21[:,1], '-k')
#sig= 0.003
#plt.semilogx(par_vec, f_acc_1000[:,0]+np.random.randn(len(par_vec))*sig, '.-g'), 
#plt.semilogx(par_vec, f_acc_1000[:,1]+np.random.randn(len(par_vec))*sig, '-g')
#plt.semilogx(par_vec, f_acc_100[:,0]+np.random.randn(len(par_vec))*sig, '.-b'), 
#plt.semilogx(par_vec, f_acc_100[:,1]+np.random.randn(len(par_vec))*sig, '-b')
#plt.semilogx(par_vec, f_acc_50[:,0]+np.random.randn(len(par_vec))*sig, '.-m'), 
#plt.semilogx(par_vec, f_acc_50[:,1]+np.random.randn(len(par_vec))*sig, '-m')
#plt.semilogx(par_vec, f_acc_35[:,0]+np.random.randn(len(par_vec))*sig, '.-r'), 
#plt.semilogx(par_vec, f_acc_35[:,1]+np.random.randn(len(par_vec))*sig, '-r')
#plt.semilogx(par_vec, f_acc_21[:,0]+np.random.randn(len(par_vec))*sig, '.-k'), 
#plt.semilogx(par_vec, f_acc_21[:,1]+np.random.randn(len(par_vec))*sig, '-k')



























###   Reconstruction with Constrained Spherical Deconvolution


mask= img_gold_test_n[:,:,:]>0

response= [0.0019, 0.0004, 0.0004]
response= response_test

#par_vec= np.logspace(-5,-1,10)
par_vec= np.linspace(0.01,0.99,10)

sphere = dpd.get_sphere()


f_acc= np.zeros( ( len(par_vec) , 2 ) )

Mose_CM_matrix= np.zeros( (6,6, len(par_vec)) )
img_mos_mf= np.zeros( (sx,sy,sz) )
img_mos_mf_matrix= np.zeros( (sx,sy,sz,len(par_vec)) )

for i_par in tqdm(range(len(par_vec)), ascii=True):
    
    par= par_vec[i_par]
    
    csd_model = ConstrainedSphericalDeconvModel(gtab_test, response)
    
    '''csd_fit = csd_model.fit(hardi_d_img_test_np, mask)
    
    csd_odf = csd_fit.odf(sphere)'''
    
    csd_peaks = peaks_from_model(model=csd_model, data=hardi_d_img_test_np, mask= mask, sphere=default_sphere,
                             relative_peak_threshold=0.5, min_separation_angle=30, parallel=False)
    
    peak_vals= csd_peaks.peak_values
    peak_dirs= csd_peaks.peak_dirs
    
    img_mos_dipy= np.sum(peak_vals>par, axis=-1)
    
    img_mos_dipy[mask==0]= 0
    
    mose_temp= img_mos_dipy.astype(np.int)
    
    Mose_CM= np.zeros( (6,6) )
    
    for i in range(len(ind_1)):
        ind= tuple(ind_1[i,:])
        n_pred= mose_temp[ind]
        Mose_CM[0,n_pred-1]+= 1
        
    for i in range(len(ind_2)):
        ind= tuple(ind_2[i,:])
        n_pred= mose_temp[ind]
        Mose_CM[1,n_pred-1]+= 1
        
    for i in range(len(ind_3)):
        ind= tuple(ind_3[i,:])
        n_pred= mose_temp[ind]
        Mose_CM[2,n_pred-1]+= 1
        
    Mose_CM_matrix[:,:,i_par]= Mose_CM.copy()






temp= peak_dirs
WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), img_gold_test_n>0)





#plt.figure(), plt.plot(par_vec, f_acc[:,0], '.'), plt.plot(par_vec, f_acc[:,1], '.r')
#
#
#
##h5f = h5py.File( results_dir + 'CSD.h5','w' )
##h5f['f_acc_1000']= f_acc_1000
##h5f['f_acc_100']= f_acc_100
##h5f['f_acc_50']= f_acc_50
##h5f['f_acc_35']= f_acc_35
##h5f['f_acc_21']= f_acc_21
##h5f.close()
#
#
#
#
#h5f = h5py.File( results_dir + 'CSD.h5','r' )
#f_acc_1000= h5f['f_acc_1000'][:]
#f_acc_100= h5f['f_acc_100'][:]
#f_acc_50= h5f['f_acc_50'][:]
#f_acc_35= h5f['f_acc_35'][:]
#f_acc_21= h5f['f_acc_21'][:]
#h5f.close()
#
#plt.figure()
##plt.plot(par_vec, f_acc_1000[:,0], '.-g'), plt.plot(par_vec, f_acc_1000[:,1], '-g')
##plt.plot(par_vec, f_acc_100[:,0], '.-b'), plt.plot(par_vec, f_acc_100[:,1], '-b')
##plt.plot(par_vec, f_acc_50[:,0], '.-m'), plt.plot(par_vec, f_acc_50[:,1], '-m')
##plt.plot(par_vec, f_acc_35[:,0], '.-r'), plt.plot(par_vec, f_acc_35[:,1], '-r')
##plt.plot(par_vec, f_acc_21[:,0], '.-k'), plt.plot(par_vec, f_acc_21[:,1], '-k')
#sig= 0.007
#plt.plot(par_vec, f_acc_1000[:,0]+np.random.randn(len(par_vec))*sig, '.-g'), 
#plt.plot(par_vec, f_acc_1000[:,1]+np.random.randn(len(par_vec))*sig, '-g')
#plt.plot(par_vec, f_acc_100[:,0]+np.random.randn(len(par_vec))*sig, '.-b'), 
#plt.plot(par_vec, f_acc_100[:,1]+np.random.randn(len(par_vec))*sig, '-b')
#plt.plot(par_vec, f_acc_50[:,0]+np.random.randn(len(par_vec))*sig, '.-m'), 
#plt.plot(par_vec, f_acc_50[:,1]+np.random.randn(len(par_vec))*sig, '-m')
#plt.plot(par_vec, f_acc_35[:,0]+np.random.randn(len(par_vec))*sig, '.-r'), 
#plt.plot(par_vec, f_acc_35[:,1]+np.random.randn(len(par_vec))*sig, '-r')
#plt.plot(par_vec, f_acc_21[:,0]+np.random.randn(len(par_vec))*sig, '.-k'), 
#plt.plot(par_vec, f_acc_21[:,1]+np.random.randn(len(par_vec))*sig, '-k')
#





















###############   Bayesian ARD

#b_vals, b_vecs= b_vals_train.copy(), b_vecs_train.copy()
b_vals, b_vecs= b_vals_test.copy(), b_vecs_test.copy()

sphere = get_sphere('symmetric724')
v, _ = sphere.vertices, sphere.faces

#Lam= np.array( [response_train[0][0], response_train[0][1] ])
Lam= np.array( [response_test[0][0], response_test[0][1] ])
d_iso=  0.003

m_max= 3

n_fib= 3
fibr_mag= np.zeros( (sx, sy, sz, n_fib, 3) )
resp_mag= np.zeros( (sx, sy, sz, n_fib) )
resp_csf  = np.zeros( (sx, sy, sz) )


#f_acc= np.zeros( ( len(thresh_vec) , 2 ) )
#
#Mose_CM_matrix= np.zeros( (3,3, len(thresh_vec)) )
#img_mos_mf= np.zeros( (sx,sy,sz) )
#img_mos_mf_matrix= np.zeros( (sx,sy,sz,len(thresh_vec) ))


#mask= hardi_d_img_train_np[:,:,:,0]>0
mask= img_gold_test_n>0

N_mcmc= 5000

Mose_CM= np.zeros( (3,3) )

for i in range(len(ind_1)):
    
    ix, iy, iz= ind_1[i,:]
    
    s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
    
    n_fib_init= 1
    R_init, bounds_lo, bounds_up = crl_dci.polar_fibers_and_iso_init(n_fib_init, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    
    solution = opt.least_squares(crl_dci.polar_fibers_and_iso_resid, R_init,
                                                    bounds=(bounds_lo,bounds_up),
                                                    args=(n_fib_init, b_vals, b_vecs,
                                                    s,  s**0.0))
    
    s_pred= crl_dci.polar_fibers_and_iso_simulate(solution.x, n_fib_init, b_vals, b_vecs)
    
    fibers= crl_dci.polar_fibers_from_solution(solution.x, n_fib_init)
    
    responses= np.array(solution.x)[0:-1:5]
    
    R_inter, _, _ = crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    R_inter[1:5]= solution.x[1:5]
    R_inter[-1]= solution.x[-1]
    R_inter[5:-1:5]= 0.00
    
    sigma= np.std(s_pred-s)*0.5
    
    R_MCMC, prob_track= crl_dci.MCMC(R_inter, b_vals, b_vecs, s, n_fib, sigma, step_f= 0.03, step_lam= 0.00001, step_ang= 0.2, step_sigma= 0.0001*sigma, \
                         aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=N_mcmc, N_aneal=N_mcmc, sample_stride=1)
    
    fibers= crl_dci.polar_fibers_from_solution(R_MCMC, n_fib)
    
    responses= np.array(R_MCMC)[0:-1:5]
    
    fibr_mag[ix, iy, iz,:,:]= fibers.T
    resp_mag[ix, iy, iz,:]= np.sort(responses)[::-1]
    
    resp_max= responses.max()
    m_opt= np.sum(responses>0.2)
    
    img_mos_mf[ix,iy,iz]= m_opt
    
    if m_opt>0 and m_opt<4:
        Mose_CM[0,m_opt-1]+= 1

for i in range(len(ind_2)):
    
    ix, iy, iz= ind_2[i,:]
    
    s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
    
    n_fib_init= 1
    R_init, bounds_lo, bounds_up = crl_dci.polar_fibers_and_iso_init(n_fib_init, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    
    solution = opt.least_squares(crl_dci.polar_fibers_and_iso_resid, R_init,
                                                    bounds=(bounds_lo,bounds_up),
                                                    args=(n_fib_init, b_vals, b_vecs,
                                                    s,  s**0.0))
    
    s_pred= crl_dci.polar_fibers_and_iso_simulate(solution.x, n_fib_init, b_vals, b_vecs)
    
    fibers= crl_dci.polar_fibers_from_solution(solution.x, n_fib_init)
    
    responses= np.array(solution.x)[0:-1:5]
    
    R_inter, _, _ = crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    R_inter[1:5]= solution.x[1:5]
    R_inter[-1]= solution.x[-1]
    R_inter[5:-1:5]= 0.00
    
    sigma= np.std(s_pred-s)*0.5
    
    R_MCMC, prob_track= crl_dci.MCMC(R_inter, b_vals, b_vecs, s, n_fib, sigma, step_f= 0.03, step_lam= 0.00001, step_ang= 0.2, step_sigma= 0.0001*sigma, \
                         aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=N_mcmc, N_aneal=1000, sample_stride=1)
    
    fibers= crl_dci.polar_fibers_from_solution(R_MCMC, n_fib)
    
    responses= np.array(R_MCMC)[0:-1:5]
    
    fibr_mag[ix, iy, iz,:,:]= fibers.T
    resp_mag[ix, iy, iz,:]= np.sort(responses)[::-1]
    
    resp_max= responses.max()
    m_opt= np.sum(responses>0.2)
    
    img_mos_mf[ix,iy,iz]= m_opt
    
    if m_opt>0 and m_opt<4:
        Mose_CM[1,m_opt-1]+= 1

for i in range(len(ind_3)):
    
    ix, iy, iz= ind_3[i,:]
    
    s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
    
    n_fib_init= 1
    R_init, bounds_lo, bounds_up = crl_dci.polar_fibers_and_iso_init(n_fib_init, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    
    solution = opt.least_squares(crl_dci.polar_fibers_and_iso_resid, R_init,
                                                    bounds=(bounds_lo,bounds_up),
                                                    args=(n_fib_init, b_vals, b_vecs,
                                                    s,  s**0.0))
    
    s_pred= crl_dci.polar_fibers_and_iso_simulate(solution.x, n_fib_init, b_vals, b_vecs)
    
    fibers= crl_dci.polar_fibers_from_solution(solution.x, n_fib_init)
    
    responses= np.array(solution.x)[0:-1:5]
    
    R_inter, _, _ = crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    R_inter[1:5]= solution.x[1:5]
    R_inter[-1]= solution.x[-1]
    R_inter[5:-1:5]= 0.00
    
    sigma= np.std(s_pred-s)*0.5
    
    R_MCMC, prob_track= crl_dci.MCMC(R_inter, b_vals, b_vecs, s, n_fib, sigma, step_f= 0.03, step_lam= 0.00001, step_ang= 0.2, step_sigma= 0.0001*sigma, \
                         aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=N_mcmc, N_aneal=1000, sample_stride=1)
    
    fibers= crl_dci.polar_fibers_from_solution(R_MCMC, n_fib)
    
    responses= np.array(R_MCMC)[0:-1:5]
    
    fibr_mag[ix, iy, iz,:,:]= fibers.T
    resp_mag[ix, iy, iz,:]= np.sort(responses)[::-1]
    
    resp_max= responses.max()
    m_opt= np.sum(responses>0.2)
    
    img_mos_mf[ix,iy,iz]= m_opt
    
    if m_opt>0 and m_opt<4:
        Mose_CM[2,m_opt-1]+= 1






n_fib= 3
fibr_mag= np.zeros( (sx, sy, sz, n_fib, 3) )
resp_mag= np.zeros( (sx, sy, sz, n_fib) )
resp_csf  = np.zeros( (sx, sy, sz) )

mask= img_gold_test_n>0

N_mcmc= 5000

Mose_CM= np.zeros( (3,3) )
    

for ix in tqdm(range(sx), ascii=True):
    for iy in range(sy):
        for iz in range(sz):
            
            if mask[ix, iy, iz]:
                
                s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
                
                n_fib_init= 1
                R_init, bounds_lo, bounds_up = crl_dci.polar_fibers_and_iso_init(n_fib_init, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
                
                solution = opt.least_squares(crl_dci.polar_fibers_and_iso_resid, R_init,
                                                                bounds=(bounds_lo,bounds_up),
                                                                args=(n_fib_init, b_vals, b_vecs,
                                                                s,  s**0.0))
                
                s_pred= crl_dci.polar_fibers_and_iso_simulate(solution.x, n_fib_init, b_vals, b_vecs)
                
                fibers= crl_dci.polar_fibers_from_solution(solution.x, n_fib_init)
                
                responses= np.array(solution.x)[0:-1:5]
                
                R_inter, _, _ = crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
                R_inter[1:5]= solution.x[1:5]
                R_inter[-1]= solution.x[-1]
                R_inter[5:-1:5]= 0.00
                
                sigma= np.std(s_pred-s)*0.5
                
                R_MCMC, prob_track= crl_dci.MCMC(R_inter, b_vals, b_vecs, s, n_fib, sigma, step_f= 0.03, step_lam= 0.00001, step_ang= 0.2, step_sigma= 0.0001*sigma, \
                                     aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=N_mcmc, N_aneal=1000, sample_stride=1)
                
                fibers= crl_dci.polar_fibers_from_solution(R_MCMC, n_fib)
                
                responses= np.array(R_MCMC)[0:-1:5]
                
                fibr_mag[ix, iy, iz,:,:]= fibers.T
                resp_mag[ix, iy, iz,:]= np.sort(responses)[::-1]
                
                resp_max= responses.max()
                m_opt= np.sum(responses>0.2)
                
                img_mos_mf[ix,iy,iz]= m_opt
                
                if m_opt>0 and m_opt<4:
                    Mose_CM[2,m_opt-1]+= 1
            
            
            



temp= np.zeros(img_gold_test_reshaped.shape)
temp[:,:,:,:3,:]= fibr_mag
WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), img_gold_test_n>0)



















































############    Proposed ML - train

sphere = get_sphere('symmetric724')
v, _ = sphere.vertices, sphere.faces

N= 15
M= 15

fodf_ml= np.zeros( (sx, sy, sz, len(v)) )
fodf_ml2= np.zeros( (sx, sy, sz, len(v)) )

mask= hardi_d_img_train_np[:,:,:,0]>0 
mask= FA_train>0.2  

for ix in tqdm(range(sx), ascii=True):
    for iy in range(sy):
        for iz in range(sz):#range(sz):
            
            if mask[ix, iy, iz]:
                
                s= hardi_b1_img_train_np[ix, iy, iz,:]/hardi_b0_img_train_np[ix, iy, iz,0]
                
                Ang= np.zeros( (v.shape[0], 2) )
                
                batch_x= crl_dci.compute_angular_feature_vector_cont_full_sphere(s, b_vecs_train.T, v, N= N, M= M, full_circ= False)
                
                y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})
                
                Ang[:,0:1]= y_pred
                
                ang_s= crl_aux.smooth_spherical_fast(v.T, Ang[:,0], n_neighbor=5, n_outlier=0, power=20,  method='1', div=60, s=1.0)
                
                ang_t, ang_p= Ang[:,1], Ang[:,0]
                
                fodf_ml[ix,iy,iz,:]= 1/ang_p
                fodf_ml2[ix,iy,iz,:]= 1/ang_s
                



############    Proposed ML - test

sphere= 'dipy'
sphere_size= 5000

if sphere=='mine':
    Xp, Yp, Zp= crl_aux.distribute_on_sphere(sphere_size)
    sphere = dipysphere.Sphere(Xp, Yp, Zp)
else:
    sphere = get_sphere('symmetric724')
    
sphere = get_sphere('symmetric724')
v, _ = sphere.vertices, sphere.faces

N= 15
M= 15
theta_thr= 30
ang_res= 12.5

n_fib= 5

fibr_mag_1= np.zeros( (sx, sy, sz, n_fib, 3) )
resp_mag_1= np.zeros( (sx, sy, sz, n_fib) )
fibr_mag_2= np.zeros( (sx, sy, sz, n_fib, 3) )
resp_mag_2= np.zeros( (sx, sy, sz, n_fib) )

fodf_ml_1= np.zeros( (sx, sy, sz, len(v)) )
fodf_ml_2= np.zeros( (sx, sy, sz, len(v)) )

mose_ml_1= np.zeros( (sx, sy, sz) )
mose_ml_2= np.zeros( (sx, sy, sz) )

mask= img_gold_test_n>0

for ix in tqdm(range(sx), ascii=True):
    for iy in range(sy):
        for iz in range(sz):#range(sz):
            
            if mask[ix, iy, iz]:
                
                s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz,0]
                
                Ang= np.zeros( (v.shape[0], 2) )
                
                batch_x= crl_dci.compute_angular_feature_vector_cont_full_sphere(s, b_vecs_test.T, v, N= N, M= M, full_circ= False)
                
                y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})
                
                Ang[:,0:1]= y_pred
                
                ang_s= crl_aux.smooth_spherical_fast(v.T, Ang[:,0], n_neighbor=7, n_outlier=0, power=20,  method='1', div=60, s=1.0)
#                try:
#                    ang_s= crl_aux.smooth_spherical(v.T, Ang[:,0], n_neighbor=5, n_outlier=0, power=20,  method='2', div=60, s=1.0)
#                except:
#                    ang_s= crl_aux.smooth_spherical_fast(v.T, Ang[:,0], n_neighbor=5, n_outlier=0, power=20,  method='1', div=60, s=1.0)
                
                v_sel, labels, n_pred, pred_fibers= crl_aux.spherical_clustering(v, ang_s, theta_thr=theta_thr, ang_res=ang_res, max_n_cluster=3, symmertric_sphere=False)
                
                mose_ml_1[ix,iy,iz]= n_pred
                fibr_mag_1[ix,iy,iz,:n_pred,:]= pred_fibers.T
                
                ang_t, ang_p= Ang[:,1], Ang[:,0]
                pred_fibers, pred_resps= crl_dci.find_dominant_fibers_dipy_way(sphere, 1/ang_s, 20, n_fib, peak_thr=.01, optimize=False, Psi= None, opt_tol=1e-7)
                to_keep= np.where(1/pred_resps<theta_thr)[0]
                pred_fibers= pred_fibers[:,to_keep]
                pred_resps= pred_resps[to_keep]
                
                n_pred= pred_fibers.shape[1]
                
                #crl_aux.plot_odf_and_fibers( v_sel , labels+1, true_fibers, true_resp=None, pred_fib=pred_fibers, pred_resp=None, N= 1000)
                
                #ang_temp= Ang[:,0]
                #crl_aux.plot_odf_and_fibers( v_sel , 1/ang_temp[ang_temp<theta_thr], true_fibers, true_resp=None, pred_fib=pred_fibers, pred_resp=None, N= 1000)
                
                mose_ml_2[ix,iy,iz]= n_pred
                
                #print(n_pred)
                
                if n_pred>n_fib:
                    pred_fibers= pred_fibers[:,:n_fib]
                    pred_resps=  pred_resps[:n_fib]
                    n_pred= n_fib
                
                fibr_mag_2[ix, iy, iz,:n_pred,:]= pred_fibers.T
                resp_mag_2[ix, iy, iz,:n_pred]= pred_resps
                #resp_csf[ix, iy, iz]  = f_n[-1]
                
                fodf_ml_1[ix,iy,iz,:]= 1/ang_p
                fodf_ml_2[ix,iy,iz,:]= 1/ang_s
                
                #min_angle_diff= crl_dci.find_min_angle_diff( fibers, true_fibers )
                
                #min_angle_diff=min_angle_diff[min_angle_diff<np.inf]
                #true_numbers_c= len( min_angle_diff )
                
                #img_err_sd[ix, iy, iz,:true_numbers_c]=  min_angle_diff*R2D






fibr_mag_ML= fibr_mag_2
fibr_mag_ML= np.transpose(fibr_mag_ML, [0,1,2,4,3])
#resp_mag_ML= resp_mag






mose_temp= mose_ml_2.astype(np.int)

Mose_CM= np.zeros( (6,6) )

for i in range(len(ind_1)):
    ind= tuple(ind_1[i,:])
    n_pred= mose_temp[ind]
    Mose_CM[0,n_pred-1]+= 1

for i in range(len(ind_2)):
    ind= tuple(ind_2[i,:])
    n_pred= mose_temp[ind]
    Mose_CM[1,n_pred-1]+= 1

for i in range(len(ind_3)):
    ind= tuple(ind_3[i,:])
    n_pred= mose_temp[ind]
    Mose_CM[2,n_pred-1]+= 1
    



temp=  fibr_mag_2
WAAE_tot, WAAE_count, WAAE_mean= crl_aux.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), img_gold_test_n>0, normalize_truth=True)







fibr_mag_ML= fibr_mag_2.copy()
fibr_mag_ML= np.transpose(fibr_mag_ML, [0,1,2,4,3])
resp_mag_ML= resp_mag_1

crl_aux.show_fibers(hardi_b1_img_test_np[:,:,:,9], mask, fibr_mag_ML, resp_mag_ML, 25, 0.3, scale_with_response=False)



''''
crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_gold, resp_mag_gold, 25, 0.5, scale_with_response=False)
crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_SD, resp_mag_SD, 25, 8.0, scale_with_response=True)
crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_ML, resp_mag_ML, 25, 0.3, scale_with_response=False)

crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_gold, resp_mag_gold, 20, 0.5, scale_with_response=False, direction='x')
crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_SD, resp_mag_SD, 20, 0.2, scale_with_response=False, direction='x')
#crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_SD, resp_mag_SD, 20, 10, scale_with_response=True, direction='x')
crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_ML, resp_mag_ML, 20, 0.3, scale_with_response=False, direction='x')

crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_gold, resp_mag_gold, 25, 0.5, scale_with_response=False, direction='y')
crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_SD, resp_mag_SD, 25, 0.2, scale_with_response=False, direction='y')
crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_SD, resp_mag_SD, 25, 10, scale_with_response=True, direction='y')
crl_aux.show_fibers(hardi_b1_img[:,:,:,9], mask, fibr_mag_ML, resp_mag_ML, 25, 1.0, scale_with_response=False, direction='y')
'''







####  HARDI test ground truth FODF

fodf_gt= np.zeros( (sx, sy, sz, len(v)) )

for ix in range(sx):
    for iy in range(sy):
        for iz in range(sz):
            
            if img_gold_test_n[ix, iy, iz]>0:
                
                true_fractions= img_gold_test_f[ix, iy, iz, :]
                true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
                true_numbers= img_gold_test_n[ix, iy, iz]
                
                true_fibers= true_fibers[:,true_fractions>0]
                
                for i_fiber in range(true_numbers):
                    
                    true_fiber= true_fibers[:,i_fiber]
                    
                    vv= np.dot(v, true_fiber)
                    
                    temp_ind= np.argmax(vv)
                    
                    fodf_gt[ix,iy,iz,temp_ind]= vv[temp_ind]











































'''
# Settings that worked
results_dir= '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/dHCP/results/sub-CC00320XX07/ses-102300/'

#affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
#                     [0.0, 1.0, 0.0, 0.0],
#                     [0.0, 0.0, 1.0, 0.0],
#                     [0.0, 0.0, 0.0, 1.0],] )
affine= FA_img_nii.affine

#CC_mask= FA>0.30
#seed_mask = np.zeros(d_img.shape[:3])
#seed_mask[65:95, 100:170, 155:185 ]= CC_mask[65:95, 100:170, 155:185 ]
seed_mask= FA>0.30
seed_mask= seed_mask *  (1-skull)

white_matter= ts_img==3 #(FA>0.20) * (1-skull)

seeds = utils.seeds_from_mask(seed_mask, affine, density=2)

#csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
#csd_fit = csd_model.fit(d_img, mask=white_matter)
#
#csa_model = CsaOdfModel(gtab, sh_order=6)
#gfa = csa_model.fit(d_img, mask=white_matter).gfa
#stopping_criterion = ThresholdStoppingCriterion(gfa, .20)
stopping_criterion = BinaryStoppingCriterion(white_matter)

#fod = csd_fit.odf(small_sphere)
pmf = fodf_ml2 # fodf_sd #fod.clip(min=0)
pmf[pmf>0.5]= 0.5
pmf= pmf**10
#######
prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=15.,
                                                sphere=sphere)

streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                     affine, step_size=.25)

streamlines = Streamlines(streamline_generator)

sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
save_trk(sft, results_dir+"tractogram_dhcp_ml_5.trk")
'''









































###############################################################################
###############################################################################
###############################################################################

#    Statistics of the HARDI2013 phantom

img_gold_test= sitk.ReadImage(hardi_dir + 'test/ground-truth-peaks.nii')
img_gold_test= sitk.GetArrayFromImage(img_gold_test)
img_gold_test= np.transpose(img_gold_test,[3,2,1,0])

sx,sy,sz,_= img_gold_test.shape

img_gold_test_f= np.zeros( (sx,sy,sz,5) )
for i in range(5):
    img_gold_test_f[:,:,:,i]= np.linalg.norm(img_gold_test[:,:,:,3*i:3*i+3], axis=-1)

img_gold_test_f_t= np.sum(img_gold_test_f, axis=-1)
img_gold_test_n  = np.sum(img_gold_test_f>0, axis=-1)


# 1
fracts_1= np.zeros( (len(np.where(img_gold_test_n==1)[0]), 1) )
fracts_1_img= np.zeros( (sx,sy,sz) )

X,Y,Z= np.where(img_gold_test_n==1)[0:3]

for i in range(len(X)):
    
    ix, iy, iz= X[i], Y[i], Z[i]
    
    true_fractions= img_gold_test_f[ix, iy, iz, :]
    true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
    
    true_fibers= true_fibers[:,true_fractions>0]
    
    fracts_1[i]= img_gold_test_f_t[ix, iy, iz]
    fracts_1_img[ix,iy,iz]= img_gold_test_f_t[ix, iy, iz]


# 2
angles_2= np.zeros( (len(np.where(img_gold_test_n==2)[0]), 1) )
fracts_2= np.zeros( (len(np.where(img_gold_test_n==2)[0]), 1) )
angles_2_img= np.zeros( (sx,sy,sz) )
fracts_2_t_img= np.zeros( (sx,sy,sz) )
fracts_2_img= np.zeros( (sx,sy,sz, 2) )

X,Y,Z= np.where(img_gold_test_n==2)[0:3]

for i in range(len(X)):
    
    ix, iy, iz= X[i], Y[i], Z[i]
    
    true_fractions= img_gold_test_f[ix, iy, iz, :]
    true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
    
    true_fibers= true_fibers[:,true_fractions>0]
    
    angles_2[i]= crl_dti.angle_between_vectors_0_90( true_fibers[:,0] , true_fibers[:,1] ) * R2D
    fracts_2[i]= img_gold_test_f_t[ix, iy, iz]
    
    true_fractions= true_fractions[true_fractions>0]
    true_fractions= np.sort(true_fractions)
    
    fracts_2_img[ix,iy,iz,: ]= true_fractions
    fracts_2_t_img[ix,iy,iz ]= img_gold_test_f_t[ix, iy, iz]
    angles_2_img[ix,iy,iz]= crl_dti.angle_between_vectors_0_90( true_fibers[:,0] , true_fibers[:,1] ) * R2D


# 3
angles_3= np.zeros( (len(np.where(img_gold_test_n==3)[0]), 3) )
fracts_3= np.zeros( (len(np.where(img_gold_test_n==3)[0]), 1) )

X,Y,Z= np.where(img_gold_test_n==3)[0:3]

for i in range(len(X)):
    
    ix, iy, iz= X[i], Y[i], Z[i]
    
    true_fractions= img_gold_test_f[ix, iy, iz, :]
    true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
    
    true_fibers= true_fibers[:,true_fractions>0]
    
    angles_3[i,0]= crl_dti.angle_between_vectors_0_90( true_fibers[:,0] , true_fibers[:,1] ) * R2D
    angles_3[i,1]= crl_dti.angle_between_vectors_0_90( true_fibers[:,0] , true_fibers[:,2] ) * R2D
    angles_3[i,2]= crl_dti.angle_between_vectors_0_90( true_fibers[:,1] , true_fibers[:,2] ) * R2D
    fracts_3[i]= img_gold_test_f_t[ix, iy, iz]












img_n_t= img_gold_test_n
img_n_p= img_mos_mf

prd_1_1= np.logical_and(img_n_t==1, img_n_p==1)
prd_1_2= np.logical_and(img_n_t==1, img_n_p==2)
prd_2_1= np.logical_and(img_n_t==2, img_n_p==1)
prd_2_2= np.logical_and(img_n_t==2, img_n_p==2)

prd_2_1_frc= fracts_2_t_img[prd_2_1]
prd_2_2_frc= fracts_2_t_img[prd_2_2]
prd_2_1_frc_min= fracts_2_img[prd_2_1,0]
prd_2_2_frc_min= fracts_2_img[prd_2_2,0]
prd_2_1_frc_max= fracts_2_img[prd_2_1,1]
prd_2_2_frc_max= fracts_2_img[prd_2_2,1]
prd_2_1_ang= angles_2_img[prd_2_1]
prd_2_2_ang= angles_2_img[prd_2_2]

plt.figure(), plt.hist(prd_2_1_frc, bins= 30);
plt.figure(), plt.hist(prd_2_2_frc, bins= 30);
plt.figure(), plt.hist(prd_2_1_frc_min, bins= 30);
plt.figure(), plt.hist(prd_2_2_frc_min, bins= 30);
plt.figure(), plt.hist(prd_2_1_ang, bins= 30);
plt.figure(), plt.hist(prd_2_2_ang, bins= 30);

plt.figure(), plt.plot( prd_2_2_frc_max, prd_2_2_frc_min, '*b', markersize=15 ), plt.plot( prd_2_1_frc_max, prd_2_1_frc_min, '.r' )
plt.xlabel('Fraction occupancy of the larger fascicle')
plt.ylabel('Fraction occupancy of the smaller fascicle')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(prd_2_2_frc_max, prd_2_2_frc_min, prd_2_2_ang, '*b')
ax.scatter(prd_2_1_frc_max, prd_2_1_frc_min, prd_2_1_ang, '.r')

















































































###    Tracking baseline 1

method_name= 'baseline'
save_dir=    hardi_dir + 'results/training/track/baseline/'

csa_model = CsaOdfModel(gtab_train, sh_order=6)

mask= hardi_d_img_train_np[:,:,:,0]>0   # FA>0.2

csa_peaks = peaks_from_model(csa_model, hardi_d_img_train_np, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=mask)

stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, 0.25)
min_bundle_length= 5
min_dist_seed= 1.0
min_dist_seed= 2.0

connectivity_stats= np.zeros( (n_hardi_fibers,2) )

for i_fiber in range(n_hardi_fibers):
    
    start_label= 2*i_fiber+1
    end_label=   2*i_fiber+2
    
    seed_mask = hardi_seed_img_np==start_label
    seeds = utils.seeds_from_mask(seed_mask, affine_train, density=2)
    
    target_mask = hardi_seed_img_np==end_label
    targets = utils.seeds_from_mask(target_mask, affine_train, density=2)
    
    streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                          affine=affine_train, step_size=.5)
    
    streamlines = Streamlines(streamlines_generator)
    
    sft = StatefulTractogram(streamlines, FA_img_train_nii, Space.RASMM)
    
    save_trk(sft, save_dir + method_name + '_' + str(i_fiber) + '.trk')
    
    bundles = [bundle for bundle in streamlines]
    
    bundles_lengths = list(length(bundles))
    
    n_hit= 0
    n_mis= 0
    
    for i_bundle in range(len(bundles)):
        
        if bundles_lengths[i_bundle]>min_bundle_length:
            
            bundle= bundles[i_bundle]
            
            dist_seed_0= np.min( np.linalg.norm( seeds- bundle[0,:], axis=1) )
            dist_seed_1= np.min( np.linalg.norm( seeds- bundle[-1,:], axis=1) )
            
            if dist_seed_0>min_dist_seed and dist_seed_1>min_dist_seed:
                
                print('Neither start nor end close to seed!', i_bundle)
            
            else:
                
                if dist_seed_0<dist_seed_1:
                    target_pr= bundle[-1,:]
                else:
                    target_pr= bundle[0,:]
                
                dist_target= np.min( np.linalg.norm( targets- target_pr, axis=1) )
                
                if dist_target<min_dist_seed:
                    n_hit+= 1
                else:
                    n_mis+= 1
            
    connectivity_stats[i_fiber,:]= [n_hit, n_mis]


np.savetxt(save_dir + method_name + '.txt', connectivity_stats, fmt='%4.0f', delimiter=',')


####  LiFE

track_address= hardi_dir + 'results/training/track/baseline/baseline_8.trk'

candidate_sl_sft = load_trk( track_address , FA_img_train_nii)
#candidate_sl_sft.to_vox()
candidate_sl = candidate_sl_sft.streamlines

streamlines= Streamlines(candidate_sl)

lengths = list(length(streamlines))
long_streamlines = Streamlines()
n_long= 0
for i, sl in enumerate(streamlines):
    if lengths[i] > 30:
        long_streamlines.append(sl)
        n_long+= sl.shape[0]

inv_affine = np.linalg.inv(FA_img_train_nii.affine)

#long_data= long_streamlines.data
#long_data= long_data[:n_long,:]
#long_data_streamlines= Streamlines(long_data)

sft = StatefulTractogram(long_streamlines, FA_img_train_nii, Space.RASMM)

save_trk(sft, hardi_dir + 'results/training/track/baseline/baseline_8_long.trk')

life_affine= inv_affine
#life_affine= np.eye(4)

fiber_model = life.FiberModel(gtab_train)

fiber_fit = fiber_model.fit(hardi_d_img_train_np, long_streamlines, affine=life_affine)

model_predict = fiber_fit.predict()

model_error = model_predict - fiber_fit.data
model_rmse = np.sqrt(np.mean(model_error[:, 1:] ** 2, -1))

beta_baseline = np.zeros(fiber_fit.beta.shape[0])
pred_weighted = np.reshape(dipy_opt.spdot(fiber_fit.life_matrix, beta_baseline),
                           (fiber_fit.vox_coords.shape[0],
                            np.sum(~gtab_train.b0s_mask)))
mean_pred = np.empty((fiber_fit.vox_coords.shape[0], gtab_train.bvals.shape[0]))
S0 = fiber_fit.b0_signal

mean_pred[..., gtab_train.b0s_mask] = S0[:, None]
mean_pred[..., ~gtab_train.b0s_mask] =\
    (pred_weighted + fiber_fit.mean_signal[:, None]) * S0[:, None]

mean_error = mean_pred - fiber_fit.data
mean_rmse = np.sqrt(np.mean(mean_error ** 2, -1))

fig, ax = plt.subplots(1)
ax.hist(mean_rmse - model_rmse, bins=100, histtype='step')
ax.text(0.2, 0.9, 'Median RMSE, mean model: %.2f' % np.median(mean_rmse),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes)
ax.text(0.2, 0.8, 'Median RMSE, LiFE: %.2f' % np.median(model_rmse),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes)
ax.set_xlabel('RMS Error')
ax.set_ylabel('# voxels')
fig.savefig('error_histograms.png')


####  end LiFE


        
        








###    Tracking baseline 2

method_name= 'baseline2'
save_dir=    hardi_dir + 'results/training/track/baseline2/'

csa_model = CsaOdfModel(gtab_train, sh_order=6)

mask= hardi_d_img_train_np[:,:,:,0]>0   # FA>0.2

csa_peaks = peaks_from_model(csa_model, hardi_d_img_train_np, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=mask)

csd_model = ConstrainedSphericalDeconvModel(gtab_train, response_train, sh_order=6)
csd_fit = csd_model.fit(hardi_d_img_train_np, mask=mask)

gfa = csa_model.fit(hardi_d_img_train_np, mask=mask).gfa

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
    csd_fit.shm_coeff, max_angle=47., sphere=default_sphere)

stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, 0.25)
min_bundle_length= 5
min_dist_seed= 1.0
min_dist_seed= 2.0

connectivity_stats= np.zeros( (n_hardi_fibers,2) )

for i_fiber in range(n_hardi_fibers):
    
    start_label= 2*i_fiber+1
    end_label=   2*i_fiber+2
    
    seed_mask = hardi_seed_img_np==start_label
    seeds = utils.seeds_from_mask(seed_mask, affine_train, density=2)
    
    target_mask = hardi_seed_img_np==end_label
    targets = utils.seeds_from_mask(target_mask, affine_train, density=2)
    
    streamline_generator = LocalTracking(detmax_dg, stopping_criterion, seeds,
                                         affine_train, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    
    sft = StatefulTractogram(streamlines, FA_img_train_nii, Space.RASMM)
    save_trk(sft, save_dir + method_name + '_' + str(i_fiber) + '.trk')
    
    bundles = [bundle for bundle in streamlines]
    
    bundles_lengths = list(length(bundles))
    
    n_hit= 0
    n_mis= 0
    
    for i_bundle in range(len(bundles)):
        
        if bundles_lengths[i_bundle]>min_bundle_length:
            
            bundle= bundles[i_bundle]
            
            dist_seed_0= np.min( np.linalg.norm( seeds- bundle[0,:], axis=1) )
            dist_seed_1= np.min( np.linalg.norm( seeds- bundle[-1,:], axis=1) )
            
            if dist_seed_0>min_dist_seed and dist_seed_1>min_dist_seed:
                
                print('Neither start nor end close to seed!', i_bundle)
            
            else:
                
                if dist_seed_0<dist_seed_1:
                    target_pr= bundle[-1,:]
                else:
                    target_pr= bundle[0,:]
                
                dist_target= np.min( np.linalg.norm( targets- target_pr, axis=1) )
                
                if dist_target<min_dist_seed:
                    n_hit+= 1
                else:
                    n_mis+= 1
            
    connectivity_stats[i_fiber,:]= [n_hit, n_mis]


np.savetxt(save_dir + method_name + '.txt', connectivity_stats, fmt='%4.0f', delimiter=',')










###    Tracking SD

method_name= 'SD'
save_dir=    hardi_dir + 'results/training/track/SD/'

pmf = fodf_sd

prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                sphere=sphere)

csa_model = CsaOdfModel(gtab_train, sh_order=6)

mask= hardi_d_img_train_np[:,:,:,0]>0

csa_model = CsaOdfModel(gtab_train, sh_order=6)
gfa = csa_model.fit(hardi_d_img_train_np, mask=mask).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, .0)

min_bundle_length= 5
min_dist_seed= 1.0
min_dist_seed= 2.0

connectivity_stats= np.zeros( (n_hardi_fibers,2) )

for i_fiber in range(n_hardi_fibers):
    
    start_label= 2*i_fiber+1
    end_label=   2*i_fiber+2
    
    seed_mask = hardi_seed_img_np==start_label
    seeds = utils.seeds_from_mask(seed_mask, affine_train, density=2)
    
    target_mask = hardi_seed_img_np==end_label
    targets = utils.seeds_from_mask(target_mask, affine_train, density=2)
    
    streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                         affine=affine_train, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    
    sft = StatefulTractogram(streamlines, FA_img_train_nii, Space.RASMM)
    
    save_trk(sft, save_dir + method_name + '_' + str(i_fiber) + '.trk')
    
    bundles = [bundle for bundle in streamlines]
    
    bundles_lengths = list(length(bundles))
    
    n_hit= 0
    n_mis= 0
    
    for i_bundle in range(len(bundles)):
        
        if bundles_lengths[i_bundle]>min_bundle_length:
            
            bundle= bundles[i_bundle]
            
            dist_seed_0= np.min( np.linalg.norm( seeds- bundle[0,:], axis=1) )
            dist_seed_1= np.min( np.linalg.norm( seeds- bundle[-1,:], axis=1) )
            
            if dist_seed_0>min_dist_seed and dist_seed_1>min_dist_seed:
                
                print('Neither start nor end close to seed!', i_fiber, i_bundle)
            
            else:
                
                if dist_seed_0<dist_seed_1:
                    target_pr= bundle[-1,:]
                else:
                    target_pr= bundle[0,:]
                
                dist_target= np.min( np.linalg.norm( targets- target_pr, axis=1) )
                
                if dist_target<min_dist_seed:
                    n_hit+= 1
                else:
                    n_mis+= 1
            
    connectivity_stats[i_fiber,:]= [n_hit, n_mis]


np.savetxt(save_dir + method_name + '.txt', connectivity_stats, fmt='%4.0f', delimiter=',')















###    Tracking ML

method_name= 'ML'
save_dir=    hardi_dir + 'results/training/track/ML/'

pmf = fodf_ml2.copy()
pmf[pmf>1.0]= 1.0
pmf= pmf**5

prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                sphere=sphere)

csa_model = CsaOdfModel(gtab_train, sh_order=6)

mask= hardi_d_img_train_np[:,:,:,0]>0
#mask= FA_train[:,:,:]>0.2

csa_model = CsaOdfModel(gtab_train, sh_order=6)
gfa = csa_model.fit(hardi_d_img_train_np, mask=mask).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, .25)

min_bundle_length= 5
min_dist_seed= 1.0
min_dist_seed= 2.0

connectivity_stats= np.zeros( (n_hardi_fibers,2) )

for i_fiber in range(n_hardi_fibers):
    
    start_label= 2*i_fiber+1
    end_label=   2*i_fiber+2
    
    seed_mask = hardi_seed_img_np==start_label
    seeds = utils.seeds_from_mask(seed_mask, affine_train, density=2)
    
    target_mask = hardi_seed_img_np==end_label
    targets = utils.seeds_from_mask(target_mask, affine_train, density=2)
    
    streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                         affine=affine_train, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    
    sft = StatefulTractogram(streamlines, FA_img_train_nii, Space.RASMM)
    
    save_trk(sft, save_dir + method_name + '_' + str(i_fiber) + '.trk')
    
    bundles = [bundle for bundle in streamlines]
    
    bundles_lengths = list(length(bundles))
    
    n_hit= 0
    n_mis= 0
    
    for i_bundle in range(len(bundles)):
        
        if bundles_lengths[i_bundle]>min_bundle_length:
            
            bundle= bundles[i_bundle]
            
            dist_seed_0= np.min( np.linalg.norm( seeds- bundle[0,:], axis=1) )
            dist_seed_1= np.min( np.linalg.norm( seeds- bundle[-1,:], axis=1) )
            
            if dist_seed_0>min_dist_seed and dist_seed_1>min_dist_seed:
                
                print('Neither start nor end close to seed!', i_fiber, i_bundle)
            
            else:
                
                if dist_seed_0<dist_seed_1:
                    target_pr= bundle[-1,:]
                else:
                    target_pr= bundle[0,:]
                
                dist_target= np.min( np.linalg.norm( targets- target_pr, axis=1) )
                
                if dist_target<min_dist_seed:
                    n_hit+= 1
                else:
                    n_mis+= 1
            
    connectivity_stats[i_fiber,:]= [n_hit, n_mis]


np.savetxt(save_dir + method_name + '.txt', connectivity_stats, fmt='%4.0f', delimiter=',')




























####   Tracking 3
#
#
#FA_img_nii= nib.load( dav_folder + 'FA.nii.gz' )
#FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
#FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
#FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
#
##affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
##                     [0.0, 1.0, 0.0, 0.0],
##                     [0.0, 0.0, 1.0, 0.0],
##                     [0.0, 0.0, 0.0, 1.0],] )
#affine= FA_img_nii.affine
#
##CC_mask= FA>0.30
##seed_mask = np.zeros(d_img.shape[:3])
##seed_mask[65:95, 100:170, 155:185 ]= CC_mask[65:95, 100:170, 155:185 ]
#seed_mask= FA>0.20
#seed_mask= seed_mask *  (1-skull)
#
#white_matter= (FA>0.20) * (1-skull)
#
#seeds = utils.seeds_from_mask(seed_mask, affine, density=[1,1,1])
#
#csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
#csd_fit = csd_model.fit(d_img, mask=white_matter)
#
#csa_model = CsaOdfModel(gtab, sh_order=6)
#gfa = csa_model.fit(d_img, mask=white_matter).gfa
#stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
#
#fod = csd_fit.odf(small_sphere)
#pmf = fod.clip(min=0)
#
########
#option=3
#if option==1:
#    prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
#                                                sphere=small_sphere)
#elif option==2:
#    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
#                                                    max_angle=30.,
#                                                    sphere=default_sphere)
#elif option==3:
#    peaks = peaks_from_model(csd_model, d_img, default_sphere, .5, 25,
#                         mask=white_matter, return_sh=True, parallel=True)
#    fod_coeff = peaks.shm_coeff
#    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(fod_coeff, max_angle=30.,
#                                                    sphere=default_sphere)
########
#streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
#                                     affine, step_size=.5)
#
#streamlines = Streamlines(streamline_generator)
#
#sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
#save_trk(sft, dav_folder+"tractogram_probabilistic_dg_pmf3.trk")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#####   Tracking 4
#
#
#FA_img_nii= nib.load( dav_folder + 'FA.nii.gz' )
#FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
#FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
#FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
#
##affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
##                     [0.0, 1.0, 0.0, 0.0],
##                     [0.0, 0.0, 1.0, 0.0],
##                     [0.0, 0.0, 0.0, 1.0],] )
#affine= FA_img_nii.affine
#
##CC_mask= FA>0.30
##seed_mask = np.zeros(d_img.shape[:3])
##seed_mask[65:95, 100:170, 155:185 ]= CC_mask[65:95, 100:170, 155:185 ]
#seed_mask= FA>0.30
#seed_mask= seed_mask *  (1-skull)
#
#white_matter= (FA>0.20) * (1-skull)
#
#seeds = utils.seeds_from_mask(seed_mask, affine, density=[1,1,1])
#
#csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
#csd_fit = csd_model.fit(d_img, mask=white_matter)
#
#csa_model = CsaOdfModel(gtab, sh_order=6)
#gfa = csa_model.fit(d_img, mask=white_matter).gfa
#stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
#
#
#
#pmf = csd_fit.odf(small_sphere).clip(min=0)
#peak_dg = ClosestPeakDirectionGetter.from_pmf(pmf, max_angle=30.,
#                                              sphere=small_sphere)
#peak_streamline_generator = LocalTracking(peak_dg, stopping_criterion, seeds,
#                                          affine, step_size=.5)
#
#streamlines = Streamlines(peak_streamline_generator)
#
#sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
#save_trk(sft, dav_folder+"closest_peak_dg_CSD.trk")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#####   Tracking sfm
#
#
#FA_img_nii= nib.load( dav_folder + 'FA.nii.gz' )
#FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
#FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
#FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
#
##affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
##                     [0.0, 1.0, 0.0, 0.0],
##                     [0.0, 0.0, 1.0, 0.0],
##                     [0.0, 0.0, 0.0, 1.0],] )
#affine= FA_img_nii.affine
#
##CC_mask= FA>0.30
##seed_mask = np.zeros(d_img.shape[:3])
##seed_mask[65:95, 100:170, 155:185 ]= CC_mask[65:95, 100:170, 155:185 ]
#seed_mask= FA>0.20
#seed_mask= seed_mask *  (1-skull)
#
#white_matter= (FA>0.20) * (1-skull)
#
#seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
#
#sphere = get_sphere()
#
#sf_model = sfm.SparseFascicleModel(gtab, sphere=sphere,
#                                   l1_ratio=0.5, alpha=0.001,
#                                   response=response[0])
#
#pnm = peaks_from_model(sf_model, d_img, sphere,
#                       relative_peak_threshold=.5,
#                       min_separation_angle=25,
#                       mask=white_matter,
#                       parallel=True)
#
#stopping_criterion = ThresholdStoppingCriterion(pnm.gfa, .25)
#
#streamline_generator = LocalTracking(pnm, stopping_criterion, seeds, affine,
#                                     step_size=.5)
#
#streamlines = Streamlines(streamline_generator)
#
#sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
#save_trk(sft, dav_folder+"tractogram_deterministic_sfm_20_20_25.trk")
#
#
#
#
#
#
#
#
#
#
#
#
#
################################################################################
#
####  LiFE
#
#
#sel_trak= dav_folder+"tractogram_deterministic_sfm_30_20_25.trk"
#
#candidate_sl_sft = load_trk(sel_trak, 'same')
#candidate_sl_sft.to_vox()
#candidate_sl = candidate_sl_sft.streamlines
#
#
#fiber_model = life.FiberModel(gtab)
#
##inv_affine = np.linalg.inv(hardi_img.affine)
#
#fiber_fit = fiber_model.fit(d_img, candidate_sl, affine=np.eye(4))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#csa_model = CsaOdfModel(gtab, sh_order=6)
#
#mask= np.logical_and(brain_mask, d_img[:,:,:,0]>20) #d_img[:,:,:,0]>0   # FA>0.2
#
#csa_peaks = peaks_from_model(csa_model, d_img, default_sphere,
#                             relative_peak_threshold=.8,
#                             min_separation_angle=45,
#                             mask=mask)
#
##                    ren = window.Renderer()
##                    ren.add(actor.peak_slicer(csa_peaks.peak_dirs,
##                                              csa_peaks.peak_values,
##                                              colors=None))
##                    
##                    window.record(ren, out_path= dav_dir + 'csa_direction_field.png', size=(900, 900))
##                    
##                    if True:
##                        window.show(ren, size=(800, 800))
#
#stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25)
#
##                    sli = slc#csa_peaks.gfa.shape[2] // 2
##                    plt.figure()
##                    plt.subplot(1, 2, 1).set_axis_off()
##                    plt.imshow(csa_peaks.gfa[:, :, sli].T, cmap='gray', origin='lower')
##                    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
##                    save_trk(sft, dav_dir+"tractogram_deterministic2.trk")
##                    plt.subplot(1, 2, 2).set_axis_off()
##                    plt.imshow((csa_peaks.gfa[:, :, sli] > 0.25).T, cmap='gray', origin='lower')
#
##plt.savefig('gfa_tracking_mask.png')
#
#CC_mask= FA>0.25
#seed_mask = np.zeros(d_img.shape[:3])
#seed_mask[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max]= \
#    CC_mask[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max]
#
##                    affine= np.array( [ [1.0, 0.0, 0.0, 0.0],
##                                         [0.0, 1.0, 0.0, 0.0],
##                                         [0.0, 0.0, 1.0, 0.0],
##                                         [0.0, 0.0, 0.0, 1.0],] )
#
#FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
#FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
#FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
#FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
#
#affine= FA_img_nii.affine
#
#seeds = utils.seeds_from_mask(seed_mask, affine, density=[2, 2, 2])
#
#streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
#                                      affine=affine, step_size=.5)
#
#streamlines = Streamlines(streamlines_generator)
#
##                    color = colormap.line_colors(streamlines)
##                    
##                    streamlines_actor = actor.line(streamlines,
##                                                   colormap.line_colors(streamlines))
##                    
##                    r = window.Renderer()
##                    r.add(streamlines_actor)
##                    
##                    window.record(r, out_path='tractogram_EuDX.png', size=(800, 800))
##                    if True:
##                        window.show(r)
#
#sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
#
#save_trk(sft, dav_dir+"tractogram_probabilistic_dg_sh.trk")
#
#                    

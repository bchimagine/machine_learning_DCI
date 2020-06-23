#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""



from __future__ import division

import numpy as np
from dipy.core.gradients import gradient_table
import scipy.optimize as opt
from dipy.data import get_sphere
import SimpleITK as sitk
#import dipy.core.sphere as dipysphere
from tqdm import tqdm
import dci
import dipy.reconst.sfm as sfm
import dipy.data as dpd
import dipy.direction.peaks as dpp
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import auto_response
import tensorflow as tf
import os











# Read test data; can be downloaded from:
# http://hardi.epfl.ch/static/events/2013_ISBI/testing_data.html
# files needed are ground-truth-peaks.nii, hardi-scheme.bval, hardi-scheme.bvec


phantom= 'HARDI2013'
gtable=  'none'
scheme=  'hardi'
hardi_snr= 30
test_dir= '... /HARDI2013/'

img_gold_test= sitk.ReadImage(test_dir + 'data/test/ground-truth-peaks.nii')
ref_dir_test= img_gold_test.GetDirection()
ref_dir_test= np.reshape(ref_dir_test, [4,4])[:3,:3].flatten()
ref_org_test= img_gold_test.GetOrigin()
ref_spc_test= img_gold_test.GetSpacing()
img_gold_test= sitk.GetArrayFromImage(img_gold_test)
img_gold_test= np.transpose(img_gold_test,[3,2,1,0])

b_vals_test= np.loadtxt( test_dir + 'data/test/'+scheme+'-scheme.bval' )
b_vecs_test= np.loadtxt( test_dir + 'data/test/'+scheme+'-scheme.bvec' )
hardi_d_img_test= sitk.ReadImage( test_dir + 'data/test/testing-data_DWIS_hardi-scheme_SNR-' + str(hardi_snr) + '.nii.gz' )
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



#  choose a subset of voxels for test
            
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
    fiber_angle= dci.compute_min_angle_between_fiberset(true_fibers)
    fractions.sort()
    if fractions[-2]>0.30 and fiber_angle>45:
        ind_2_test_c+=1
        ind_2_test[ind_2_test_c,:]= ind_2[i,:]

ind_2= ind_2_test[:ind_2_test_c+1,:].astype(np.int)[:300,:]

#  choose 3-compartments

ind_3= np.where( np.logical_and(img_gold_test_n==3, img_gold_test_f_t>0.60) )
ind_3=np.vstack( (ind_3[0], ind_3[1], ind_3[2])).T
ind_3_test= np.zeros((50,3))
ind_3_test_c= -1

for i in range(ind_3.shape[0]):
    ix, iy, iz= ind_3[i,:]
    fractions= img_gold_test_f[ix, iy, iz].copy()
    true_fibers= np.reshape( img_gold_test[ix, iy, iz, :], [5,3] ).T
    true_fibers= true_fibers[:,fractions>0]
    fiber_angle= dci.compute_min_angle_between_fiberset(true_fibers)
    fractions.sort()
    if fractions[-3]>0.15 and fiber_angle>30:
        ind_3_test_c+=1
        if ind_3_test_c<50:
            ind_3_test[ind_3_test_c,:]= ind_3[i,:]

ind_3= ind_3_test[:ind_3_test_c+1,:].astype(np.int)






'''tenmodel = dti.TensorModel(gtab_test)

tenfit = tenmodel.fit(hardi_d_img_test_np, img_gold_test_n>0)
FA_test = fractional_anisotropy(tenfit.evals)
FA_test[np.isnan(FA_test)] = 0
FA_test[img_gold_test_n==0]= 0

FA_img_test= np.transpose(FA_test, [2,1,0])
FA_img_test= sitk.GetImageFromArray(FA_img_test)
FA_img_test.SetDirection(ref_dir_test)
FA_img_test.SetOrigin(ref_org_test)
FA_img_test.SetSpacing(ref_spc_test)

sitk.WriteImage(FA_img_test, test_dir + 'results/test/FA_test.mhd')
sitk.WriteImage(FA_img_test, test_dir + 'results/test/FA_test.nii.gz' )

FA_img_test_nii= nib.load( test_dir + 'results/test/FA_test.nii.gz' )
FA_img_test_nii.header['srow_x']= FA_img_test_nii.affine[0,:]
FA_img_test_nii.header['srow_y']= FA_img_test_nii.affine[1,:]
FA_img_test_nii.header['srow_z']= FA_img_test_nii.affine[2,:]

affine_test= FA_img_test_nii.affine'''

response_test, ratio_test = auto_response(gtab_test, hardi_d_img_test_np, roi_radius=10, fa_thr=0.7)
















# Tensor fitting methods

#   F- test and bootstrap

b_vals, b_vecs= b_vals_test.copy(), b_vecs_test.copy()

Lam= np.array( [response_test[0][0], response_test[0][1] ])
d_iso=  0.003

img_err_sd= np.zeros( (sx,sy,sz,5) )
img_err_mf= np.zeros( (sx,sy,sz,5) )
prd_err_sd= np.zeros( (sx,sy,sz) )
prd_err_mf= np.zeros( (sx,sy,sz) )
FBC_err= np.zeros( (5,5) )

m_max= 3

# if you set this to False, it will use generalization error computed with boosting 
# to determine the number of fascicles
F_test= False
n_bs= 10
# this is the vector of threshold values we tried
thresh_vec=  np.concatenate( (np.arange(0.1, 0.5, 0.2), np.arange(0.5, 2, 1.0), np.arange(3, 18, 4) ))
# we found a threshold of around 7 to work best, so can use this
i_thresh= 5

# if you set this to False, it will use F-test
# to determine the number of fascicles
F_test= True
# this is the vector of threshold values we tried
thresh_vec= np.concatenate( (np.arange(1, 20, 3), np.arange(20, 50, 10)) )
# we found a threshold of around 20 to work best, so can use this
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
            m_opt= dci.model_selection_f_test(s, b_vals, b_vecs, m_max= m_max, threshold= threshold, 
                                          condition_mode= 'F_val', model='ball_n_sticks')
        else:
            m_opt= dci.model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= m_max, threshold= threshold, 
                                                     model='ball_n_sticks', delta_mthod= False)
        
        img_mos_mf[ix,iy,iz]= m_opt
        
        if m_opt>0 and m_opt<4:
            Mose_CM[0,m_opt-1]+= 1
        
        n_fib= m_opt
        
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = dci.diamond_init_log(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
            
        if model=='ball_n_sticks':
            solution = opt.least_squares(dci.polar_fibers_and_iso_resid, R_init,
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
            solution = opt.least_squares(dci.diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( n_fib, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
        
        fibers= dci.polar_fibers_from_solution(solution.x, n_fib)
        
        responses= np.array(solution.x)[0:-1:5]
        
        fibr_mag[ix, iy, iz,:n_fib,:]= fibers.T
        resp_mag[ix, iy, iz,:n_fib]= np.sort(responses)[::-1]
    
    for i in range(len(ind_2)):
        
        ix, iy, iz= ind_2[i,:]
        
        s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
        
        if F_test:
            m_opt= dci.model_selection_f_test(s, b_vals, b_vecs, m_max= m_max, threshold= threshold, 
                                          condition_mode= 'F_val', model='ball_n_sticks')
        else:
            m_opt= dci.model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= m_max, threshold= threshold, 
                                                     model='ball_n_sticks', delta_mthod= False)
        
        img_mos_mf[ix,iy,iz]= m_opt
        
        if m_opt>0 and m_opt<4:
            Mose_CM[1,m_opt-1]+= 1
        
        n_fib= m_opt
        
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = dci.diamond_init_log(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
            
        if model=='ball_n_sticks':
            solution = opt.least_squares(dci.polar_fibers_and_iso_resid, R_init,
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
            solution = opt.least_squares(dci.diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( n_fib, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
        
        fibers= dci.polar_fibers_from_solution(solution.x, n_fib)
        
        responses= np.array(solution.x)[0:-1:5]
        
        fibr_mag[ix, iy, iz,:n_fib,:]= fibers.T
        resp_mag[ix, iy, iz,:n_fib]= np.sort(responses)[::-1]
    
    for i in range(len(ind_3)):
        
        ix, iy, iz= ind_3[i,:]
        
        s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
        
        if F_test:
            m_opt= dci.model_selection_f_test(s, b_vals, b_vecs, m_max= m_max, threshold= threshold, 
                                          condition_mode= 'F_val', model='ball_n_sticks')
        else:
            m_opt= dci.model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= m_max, threshold= threshold, 
                                                     model='ball_n_sticks', delta_mthod= False)
        
        img_mos_mf[ix,iy,iz]= m_opt
        
        if m_opt>0 and m_opt<4:
            Mose_CM[2,m_opt-1]+= 1
        
        n_fib= m_opt
        
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = dci.diamond_init_log(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
            
        if model=='ball_n_sticks':
            solution = opt.least_squares(dci.polar_fibers_and_iso_resid, R_init,
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
            solution = opt.least_squares(dci.diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( n_fib, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
        
        fibers= dci.polar_fibers_from_solution(solution.x, n_fib)
        
        responses= np.array(solution.x)[0:-1:5]
        
        fibr_mag[ix, iy, iz,:n_fib,:]= fibers.T
        resp_mag[ix, iy, iz,:n_fib]= np.sort(responses)[::-1]
    
    Mose_CM_matrix[:,:,i_thresh]= Mose_CM.copy()




# the same as above but we run on all voxels to compute WAAE

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
                    m_opt= dci.model_selection_f_test(s, b_vals, b_vecs, m_max= m_max, threshold= threshold, 
                                                  condition_mode= 'F_val', model='ball_n_sticks')
                else:
                    m_opt= dci.model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= m_max, threshold= threshold, 
                                                             model='ball_n_sticks', delta_mthod= False)
                
                n_fib= m_opt
                
                if model=='ball_n_sticks':
                    R_init, bounds_lo, bounds_up = dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
                elif model=='DIAMOND':
                    #R_init, bounds_lo, bounds_up = diamond_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
                    R_init, bounds_lo, bounds_up = dci.diamond_init_log(n_fib, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
                    
                if model=='ball_n_sticks':
                    solution = opt.least_squares(dci.polar_fibers_and_iso_resid, R_init,
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
                    solution = opt.least_squares(dci.diamond_resid_log, R_init,
                                            bounds=(bounds_lo,bounds_up),
                                            args=( n_fib, b_vals, b_vecs,
                                            s,
                                            wt_temp**0.0))
                
                fibers= dci.polar_fibers_from_solution(solution.x, n_fib)
                
                responses= np.array(solution.x)[0:-1:5]
                
                fibr_mag[ix, iy, iz,:n_fib,:]= fibers.T
                resp_mag[ix, iy, iz,:n_fib]= np.sort(responses)[::-1]
                
                img_mos_mf[ix,iy,iz]= m_opt



temp= np.zeros(img_gold_test_reshaped.shape)
temp[:,:,:,:3,:]= fibr_mag
    
WAAE_tot, WAAE_count, WAAE_mean= dci.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), mask, normalize_truth=True)





























###   Sparse Fascicle Model - using DIPY

mask= img_gold_test_n[:,:,:]>0

#response= [0.0019, 0.0004, 0.0004]
response= response_test

# range of thresholds to try
par_vec= np.logspace(-7,-2,10)
# we dound a threshold of around 0.001-0.003 to work best
i_par= 8

f_acc= np.zeros( ( len(par_vec) , 2 ) )

Mose_CM_matrix= np.zeros( (6,6, len(par_vec)) )
img_mos_mf= np.zeros( (sx,sy,sz) )
img_mos_mf_matrix= np.zeros( (sx,sy,sz,len(par_vec)) )

sphere = dpd.get_sphere()


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
WAAE_tot, WAAE_count, WAAE_mean= dci.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), img_gold_test_n>0)







###   Constrained Spherical Deconvolution - using DIPY implementation


mask= img_gold_test_n[:,:,:]>0

#response= [0.0019, 0.0004, 0.0004]
response= response_test

# thresholds vector to try
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
WAAE_tot, WAAE_count, WAAE_mean= dci.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), img_gold_test_n>0)











###############   Bayesian method  - Automatic Relevance Determination

#b_vals, b_vecs= b_vals_train.copy(), b_vecs_train.copy()
b_vals, b_vecs= b_vals_test.copy(), b_vecs_test.copy()

sphere = get_sphere('symmetric724')
v, _ = sphere.vertices, sphere.faces

Lam= np.array( [response_test[0][0], response_test[0][1] ])
d_iso=  0.003

m_max= 3



#   Run on selected voxels only

n_fib= 3
fibr_mag= np.zeros( (sx, sy, sz, n_fib, 3) )
resp_mag= np.zeros( (sx, sy, sz, n_fib) )
resp_csf  = np.zeros( (sx, sy, sz) )


mask= img_gold_test_n>0

N_mcmc= 500

Mose_CM= np.zeros( (3,3) )

for i in range(len(ind_1)):
    
    ix, iy, iz= ind_1[i,:]
    
    s= hardi_b1_img_test_np[ix, iy, iz,:]/hardi_b0_img_test_np[ix, iy, iz]
    
    n_fib_init= 1
    R_init, bounds_lo, bounds_up = dci.polar_fibers_and_iso_init(n_fib_init, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    
    solution = opt.least_squares(dci.polar_fibers_and_iso_resid, R_init,
                                                    bounds=(bounds_lo,bounds_up),
                                                    args=(n_fib_init, b_vals, b_vecs,
                                                    s,  s**0.0))
    
    s_pred= dci.polar_fibers_and_iso_simulate(solution.x, n_fib_init, b_vals, b_vecs)
    
    fibers= dci.polar_fibers_from_solution(solution.x, n_fib_init)
    
    responses= np.array(solution.x)[0:-1:5]
    
    R_inter, _, _ = dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    R_inter[1:5]= solution.x[1:5]
    R_inter[-1]= solution.x[-1]
    R_inter[5:-1:5]= 0.00
    
    sigma= np.std(s_pred-s)*0.5
    
    R_MCMC, prob_track= dci.MCMC(R_inter, b_vals, b_vecs, s, n_fib, sigma, step_f= 0.03, step_lam= 0.00001, step_ang= 0.2, step_sigma= 0.0001*sigma, \
                         aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=N_mcmc, N_aneal=N_mcmc, sample_stride=1)
    
    fibers= dci.polar_fibers_from_solution(R_MCMC, n_fib)
    
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
    R_init, bounds_lo, bounds_up = dci.polar_fibers_and_iso_init(n_fib_init, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    
    solution = opt.least_squares(dci.polar_fibers_and_iso_resid, R_init,
                                                    bounds=(bounds_lo,bounds_up),
                                                    args=(n_fib_init, b_vals, b_vecs,
                                                    s,  s**0.0))
    
    s_pred= dci.polar_fibers_and_iso_simulate(solution.x, n_fib_init, b_vals, b_vecs)
    
    fibers= dci.polar_fibers_from_solution(solution.x, n_fib_init)
    
    responses= np.array(solution.x)[0:-1:5]
    
    R_inter, _, _ = dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    R_inter[1:5]= solution.x[1:5]
    R_inter[-1]= solution.x[-1]
    R_inter[5:-1:5]= 0.00
    
    sigma= np.std(s_pred-s)*0.5
    
    R_MCMC, prob_track= dci.MCMC(R_inter, b_vals, b_vecs, s, n_fib, sigma, step_f= 0.03, step_lam= 0.00001, step_ang= 0.2, step_sigma= 0.0001*sigma, \
                         aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=N_mcmc, N_aneal=1000, sample_stride=1)
    
    fibers= dci.polar_fibers_from_solution(R_MCMC, n_fib)
    
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
    R_init, bounds_lo, bounds_up = dci.polar_fibers_and_iso_init(n_fib_init, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    
    solution = opt.least_squares(dci.polar_fibers_and_iso_resid, R_init,
                                                    bounds=(bounds_lo,bounds_up),
                                                    args=(n_fib_init, b_vals, b_vecs,
                                                    s,  s**0.0))
    
    s_pred= dci.polar_fibers_and_iso_simulate(solution.x, n_fib_init, b_vals, b_vecs)
    
    fibers= dci.polar_fibers_from_solution(solution.x, n_fib_init)
    
    responses= np.array(solution.x)[0:-1:5]
    
    R_inter, _, _ = dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
    R_inter[1:5]= solution.x[1:5]
    R_inter[-1]= solution.x[-1]
    R_inter[5:-1:5]= 0.00
    
    sigma= np.std(s_pred-s)*0.5
    
    R_MCMC, prob_track= dci.MCMC(R_inter, b_vals, b_vecs, s, n_fib, sigma, step_f= 0.03, step_lam= 0.00001, step_ang= 0.2, step_sigma= 0.0001*sigma, \
                         aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=N_mcmc, N_aneal=1000, sample_stride=1)
    
    fibers= dci.polar_fibers_from_solution(R_MCMC, n_fib)
    
    responses= np.array(R_MCMC)[0:-1:5]
    
    fibr_mag[ix, iy, iz,:,:]= fibers.T
    resp_mag[ix, iy, iz,:]= np.sort(responses)[::-1]
    
    resp_max= responses.max()
    m_opt= np.sum(responses>0.2)
    
    img_mos_mf[ix,iy,iz]= m_opt
    
    if m_opt>0 and m_opt<4:
        Mose_CM[2,m_opt-1]+= 1





# same as above, but run on all voxels to compute WAAE
        
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
                R_init, bounds_lo, bounds_up = dci.polar_fibers_and_iso_init(n_fib_init, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
                
                solution = opt.least_squares(dci.polar_fibers_and_iso_resid, R_init,
                                                                bounds=(bounds_lo,bounds_up),
                                                                args=(n_fib_init, b_vals, b_vecs,
                                                                s,  s**0.0))
                
                s_pred= dci.polar_fibers_and_iso_simulate(solution.x, n_fib_init, b_vals, b_vecs)
                
                fibers= dci.polar_fibers_from_solution(solution.x, n_fib_init)
                
                responses= np.array(solution.x)[0:-1:5]
                
                R_inter, _, _ = dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
                R_inter[1:5]= solution.x[1:5]
                R_inter[-1]= solution.x[-1]
                R_inter[5:-1:5]= 0.00
                
                sigma= np.std(s_pred-s)*0.5
                
                R_MCMC, prob_track= dci.MCMC(R_inter, b_vals, b_vecs, s, n_fib, sigma, step_f= 0.03, step_lam= 0.00001, step_ang= 0.2, step_sigma= 0.0001*sigma, \
                                     aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=N_mcmc, N_aneal=1000, sample_stride=1)
                
                fibers= dci.polar_fibers_from_solution(R_MCMC, n_fib)
                
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
WAAE_tot, WAAE_count, WAAE_mean= dci.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), img_gold_test_n>0)



















































############    Proposed Machine learning Method

N= 15
n_feat_vec= np.array([2*N+2, 30, 60, 80, 80, 60, 30, 1])


def reg_net(X, n_feat_vec, p_keep_hidden, bias_init=0.001):
    
    # defines MLP architecture
    
    inp = X
    
    for i in range(len(n_feat_vec)-1):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc_'+str(i))
        inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
        
        inp = tf.nn.dropout(inp, p_keep_hidden)
    
    return inp

X = tf.placeholder("float32", [None, n_feat_vec[0]])
Y = tf.placeholder("float32", [None, n_feat_vec[-1]])

learning_rate = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

Y_p = reg_net(X, n_feat_vec, p_keep_hidden, bias_init=0.001)

gpu_ind= 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())


saved_model_path =  '... /model_v1_correct_mean_M15_N15_snr20_ang30_err_7_69.ckpt'
saver.restore(sess, saved_model_path)




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
                
                batch_x= dci.compute_angular_feature_vector_cont_full_sphere(s, b_vecs_test.T, v, N= N, M= M, full_circ= False)
                
                y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})
                
                Ang[:,0:1]= y_pred
                
                ang_s= dci.smooth_spherical_fast(v.T, Ang[:,0], n_neighbor=7, n_outlier=0, power=20,  method='1', div=60, s=1.0)

                v_sel, labels, n_pred, pred_fibers= dci.spherical_clustering(v, ang_s, theta_thr=theta_thr, ang_res=ang_res, max_n_cluster=3, symmertric_sphere=False)
                
                mose_ml_1[ix,iy,iz]= n_pred
                fibr_mag_1[ix,iy,iz,:n_pred,:]= pred_fibers.T
                
                ang_t, ang_p= Ang[:,1], Ang[:,0]
                pred_fibers, pred_resps= dci.find_dominant_fibers_dipy_way(sphere, 1/ang_s, 20, n_fib, peak_thr=.01, optimize=False, Psi= None, opt_tol=1e-7)
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
                
                fodf_ml_1[ix,iy,iz,:]= 1/ang_p
                fodf_ml_2[ix,iy,iz,:]= 1/ang_s
                




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
Error_matrix, WAAE_tot, WAAE_count, WAAE_mean= dci.compute_WAAE(img_gold_test_reshaped.copy(), temp.copy(), img_gold_test_n>0, normalize_truth=True)



















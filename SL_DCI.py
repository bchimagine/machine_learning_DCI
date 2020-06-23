#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:24:47 2020

@author: davood
"""




import numpy as np
import crl_dci
import crl_aux
import matplotlib.pyplot as plt
from dipy.data import get_sphere
#from sklearn import svm
import tensorflow as tf
import dk_model
import os
from tqdm import tqdm
import h5py
from numpy import matlib as mb
from os import listdir
from os.path import isfile, join, isdir
import dipy.core.sphere as dipysphere


base_dir= '/media/nerossd2/ML-DCI/'


gtab_scheme=  'irontract'

if gtab_scheme=='dHCP':
    
    b_vals= np.loadtxt( base_dir + 'sub-CC00060XX03_ses-12501_desc-preproc_dwi.bval' )
    b_vecs= np.loadtxt( base_dir + 'sub-CC00060XX03_ses-12501_desc-preproc_dwi.bvec' ).T
    b_vecs= b_vecs[b_vals==1000]
    b_vals= b_vals[b_vals==1000]
    
    b_vals_test= np.loadtxt( base_dir + 'sub-CC00124XX09_ses-42302_desc-preproc_dwi.bval' )
    b_vecs_test= np.loadtxt( base_dir + 'sub-CC00124XX09_ses-42302_desc-preproc_dwi.bvec' ).T
    b_vecs_test= b_vecs_test[b_vals_test==1000]
    b_vals_test= b_vals_test[b_vals_test==1000]
    
    train_bvecs = [f for f in listdir(base_dir+'train_gtabs/') if isfile(join(base_dir+'train_gtabs/', f)) and 'vec' in f]
    train_bvals = [f for f in listdir(base_dir+'train_gtabs/') if isfile(join(base_dir+'train_gtabs/', f)) and 'val' in f]
    train_bvecs.sort()
    train_bvals.sort()
    n_train_gtab= len(train_bvals)
    
elif gtab_scheme=='HARDI2013':
    
    b_vals= np.loadtxt( '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/HARDI2013/data/test/hardi-scheme.bval' )
    b_vecs= np.loadtxt( '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/HARDI2013/data/test/hardi-scheme.bvec' ).T
    b_vecs= b_vecs[b_vals>10,:]
    b_vals= b_vals[b_vals>10]
    
    b_vals_test= b_vals
    b_vecs_test= b_vecs
    
    train_bvecs= [b_vecs]
    train_bvals= [b_vals]
    n_train_gtab= len(train_bvals)
    
elif gtab_scheme=='irontract':
    
    b_vals= np.loadtxt( '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/irontract/bvalues.hcpl.txt' )
    b_vecs= np.loadtxt( '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/irontract/gradients.hcpl.txt' )
    b_norms= np.zeros(b_vecs.shape[0])
    
    b_val_ind= np.where( b_vals <10000 )[0][1:]
    
    for i in range(b_vecs.shape[0]):
        temp= np.linalg.norm(b_vecs[i,:])
        b_norms[i]= temp
        if temp>0:
            b_vecs[i,:]/= temp
    
    b_vals= b_vals[b_val_ind]
    b_vecs= b_vecs[b_val_ind,:]
    
    b_vecs= b_vecs[b_vals>10,:]
    b_vals= b_vals[b_vals>10]
    
    b_vals_test= b_vals
    b_vecs_test= b_vecs
    
    train_bvecs= [b_vecs]
    train_bvals= [b_vals]
    n_train_gtab= len(train_bvals)



true_fibers= np.array( [[0.3 , 0.0, 0.3/np.sqrt(2)], [0.0 , 0.3, 0.3/np.sqrt(2)] , [0.0 , 0.0, 0.0]] , np.float64)
true_fibers= np.array( [[0.40 , 0.0], [0.0 , 0.40] , [0.0 , 0.0]] , np.float64)
true_fibers= np.array( [ [ np.sqrt(2)/2 , np.sqrt(2)/2 , 0 ] ] , np.float64).T

s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, 0.0019, 0.0004, 0.003)
s= crl_aux.add_rician_noise(s, snr=15)

N= 15
M= 15

n_sim= 100

plt.figure()

for i_sim in range(n_sim):
    
    test_dir= np.random.rand(3)
        
    f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
    
    ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
    
    color= [ang/90, 0.0, (90-ang)/90]
    
    plt.plot(f_cont[:,0], color=color)
    





####    Generate training and test data

N= 15
M= 15

min_fiber_separation= 30

D_par_range= [0.0019, 0.00235]
D_per_range= [0.00045, 0.00060]

if gtab_scheme=='irontract':
    D_par_range= [0.0015, 0.0020]
    D_per_range= [0.00035, 0.00045]

###########

n_fib= 1
CSF_range= [0.10, 0.80]
snr= 15

n_fib_sim= 100000
n_sig_sim= 10

X_train_1= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_train_1= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    if gtab_scheme=='dHCP':
        i_gtab= np.random.randint(0,n_train_gtab)
        b_vals= np.loadtxt( base_dir+'train_gtabs/' + train_bvals[i_gtab] )
        b_vecs= np.loadtxt( base_dir+'train_gtabs/' + train_bvecs[i_gtab] ).T
        b_vecs= b_vecs[b_vals==1000]
        b_vals= b_vals[b_vals==1000]
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_train+= 1
        
        X_train_1[i_train,:]= f_cont.flatten('F')
        
        Y_train_1[i_train,:]= ang

n_fib_sim= 10000
n_sig_sim= 10

X_test_1= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_test_1= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_test= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_test+= 1
        
        X_test_1[i_test,:]= f_cont.flatten('F')
        
        Y_test_1[i_test,:]= ang
            
############

n_fib= 2
CSF_range= [0.0, 0.50]
Fr1_range= [0.15, 0.30]

n_fib_sim= 500000
n_sig_sim= 10

X_train_2= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_train_2= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    if gtab_scheme=='dHCP':
        i_gtab= np.random.randint(0,n_train_gtab)
        b_vals= np.loadtxt( base_dir+'train_gtabs/' + train_bvals[i_gtab] )
        b_vecs= np.loadtxt( base_dir+'train_gtabs/' + train_bvecs[i_gtab] ).T
        b_vecs= b_vecs[b_vals==1000]
        b_vals= b_vals[b_vals==1000]
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_train+= 1
        
        X_train_2[i_train,:]= f_cont.flatten('F')
        
        Y_train_2[i_train,:]= ang

n_fib_sim= 10000
n_sig_sim= 10

X_test_2= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_test_2= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_test= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_test+= 1
        
        X_test_2[i_test,:]= f_cont.flatten('F')
        
        Y_test_2[i_test,:]= ang

############

n_fib= 3
CSF_range= [0.0, 0.3]
Fr1_range= [0.2, 0.3]
Fr2_range= [0.2, 0.3]

n_fib_sim= 500000
n_sig_sim= 10

X_train_3= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_train_3= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    if gtab_scheme=='dHCP':
        i_gtab= np.random.randint(0,n_train_gtab)
        b_vals= np.loadtxt( base_dir+'train_gtabs/' + train_bvals[i_gtab] )
        b_vecs= np.loadtxt( base_dir+'train_gtabs/' + train_bvecs[i_gtab] ).T
        b_vecs= b_vecs[b_vals==1000]
        b_vals= b_vals[b_vals==1000]
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_train+= 1
        
        X_train_3[i_train,:]= f_cont.flatten('F')
        
        Y_train_3[i_train,:]= ang

n_fib_sim= 10000
n_sig_sim= 10

X_test_3= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_test_3= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_test= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_test+= 1
        
        X_test_3[i_test,:]= f_cont.flatten('F')
        
        Y_test_3[i_test,:]= ang









####    Generate training and test data - spherical
'''
N= 15
M= 15

min_fiber_separation= 30

D_par_range= [0.0019, 0.00235]
D_per_range= [0.00045, 0.00060]

sphere= 'dipy'
sphere_size= 500

if sphere=='mine':
    Xp, Yp, Zp= crl_aux.distribute_on_sphere(sphere_size)
    sphere = dipysphere.Sphere(Xp, Yp, Zp)
else:
    sphere = get_sphere('symmetric724')

v, _ = sphere.vertices, sphere.faces

sphere_size= v.shape[0]

##########

n_fib= 1
CSF_range= [0.10, 0.80]
snr= 35

n_fib_sim= 100000
n_sig_sim= 10

X_train_1= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_train_1= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    if gtab_scheme=='dHCP':
        i_gtab= np.random.randint(0,n_train_gtab)
        b_vals= np.loadtxt( base_dir+'train_gtabs/' + train_bvals[i_gtab] )
        b_vecs= np.loadtxt( base_dir+'train_gtabs/' + train_bvecs[i_gtab] ).T
        b_vecs= b_vecs[b_vals==1000]
        b_vals= b_vals[b_vals==1000]
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    
    
    x1 = crl_dci.compute_angular_feature_vector_cont_full_sphere(s, b_vecs.T, v, N= N, M= M, full_circ= False)
    x2= crl_dci.compute_min_angle_between_vector_sets_full_sphere(v.T, true_fibers)
    

    
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_train+= 1
        
        X_train_1[i_train,:]= f_cont.flatten('F')
        
        Y_train_1[i_train,:]= ang

n_fib_sim= 10000
n_sig_sim= 10

X_test_1= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_test_1= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_test= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_test+= 1
        
        X_test_1[i_test,:]= f_cont.flatten('F')
        
        Y_test_1[i_test,:]= ang
            
###########

n_fib= 2
CSF_range= [0.0, 0.50]
Fr1_range= [0.15, 0.30]

n_fib_sim= 500000
n_sig_sim= 10

X_train_2= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_train_2= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    if gtab_scheme=='dHCP':
        i_gtab= np.random.randint(0,n_train_gtab)
        b_vals= np.loadtxt( base_dir+'train_gtabs/' + train_bvals[i_gtab] )
        b_vecs= np.loadtxt( base_dir+'train_gtabs/' + train_bvecs[i_gtab] ).T
        b_vecs= b_vecs[b_vals==1000]
        b_vals= b_vals[b_vals==1000]
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_train+= 1
        
        X_train_2[i_train,:]= f_cont.flatten('F')
        
        Y_train_2[i_train,:]= ang

n_fib_sim= 10000
n_sig_sim= 10

X_test_2= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_test_2= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_test= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_test+= 1
        
        X_test_2[i_test,:]= f_cont.flatten('F')
        
        Y_test_2[i_test,:]= ang
        
###########

n_fib= 3
CSF_range= [0.0, 0.3]
Fr1_range= [0.2, 0.3]
Fr2_range= [0.2, 0.3]

n_fib_sim= 500000
n_sig_sim= 10

X_train_3= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_train_3= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    if gtab_scheme=='dHCP':
        i_gtab= np.random.randint(0,n_train_gtab)
        b_vals= np.loadtxt( base_dir+'train_gtabs/' + train_bvals[i_gtab] )
        b_vecs= np.loadtxt( base_dir+'train_gtabs/' + train_bvecs[i_gtab] ).T
        b_vecs= b_vecs[b_vals==1000]
        b_vals= b_vals[b_vals==1000]
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_train+= 1
        
        X_train_3[i_train,:]= f_cont.flatten('F')
        
        Y_train_3[i_train,:]= ang

n_fib_sim= 10000
n_sig_sim= 10

X_test_3= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_test_3= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_test= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_test+= 1
        
        X_test_3[i_test,:]= f_cont.flatten('F')
        
        Y_test_3[i_test,:]= ang
'''









#h5f = h5py.File(base_dir + 'data_snr_15_irontract.h5','w')
#h5f['X_train_1']= X_train_1
#h5f['Y_train_1']= Y_train_1
#h5f['X_test_1']= X_test_1
#h5f['Y_test_1']= Y_test_1
#h5f['X_train_2']= X_train_2
#h5f['Y_train_2']= Y_train_2
#h5f['X_test_2']= X_test_2
#h5f['Y_test_2']= Y_test_2
#h5f['X_train_3']= X_train_3
#h5f['Y_train_3']= Y_train_3
#h5f['X_test_3']= X_test_3
#h5f['Y_test_3']= Y_test_3
#h5f.close()



#h5f = h5py.File(base_dir + 'data_snr_20_diff_M15_N15_ang30_hardi.h5', 'r')
#h5f = h5py.File(base_dir + 'data_snr_20_diff_gtable2_M15_N15_ang45.h5', 'r')
#h5f = h5py.File(base_dir + 'data_snr_20_diff_gtable2_M15_N15_ang25.h5', 'r')
h5f = h5py.File(base_dir + 'data_snr_20_irontract.h5', 'r')
X_train_1 = h5f['X_train_1'][:]
Y_train_1 = h5f['Y_train_1'][:]
X_test_1 = h5f['X_test_1'][:]
Y_test_1 = h5f['Y_test_1'][:]
X_train_2 = h5f['X_train_2'][:]
Y_train_2 = h5f['Y_train_2'][:]
X_test_2 = h5f['X_test_2'][:]
Y_test_2 = h5f['Y_test_2'][:]
X_train_3 = h5f['X_train_3'][:]
Y_train_3 = h5f['Y_train_3'][:]
X_test_3 = h5f['X_test_3'][:]
Y_test_3 = h5f['Y_test_3'][:]
h5f.close()

#plt.figure(), plt.hist(Y_train_1, bins=100);
#plt.figure(), plt.hist(Y_test_1, bins=100);
#plt.figure(), plt.hist(Y_train_2, bins=100);
#plt.figure(), plt.hist(Y_test_2, bins=100);
#plt.figure(), plt.hist(Y_train_3, bins=100);
#plt.figure(), plt.hist(Y_test_3, bins=100);

#n_plot= 100
#plt.figure()
#for i_sim in range(n_plot):
#    f_cont= X_train_1[i_sim*10000,:N+1]
#    ang= Y_train_1[i_sim*10000,0]
#    color= [ang/90, 0.0, (90-ang)/90]
#    plt.plot(f_cont, color=color)
#  
#n_plot= 100
#plt.figure()
#for i_sim in range(n_plot):
#    f_cont= X_train_2[i_sim*10000,:N+1]
#    ang= Y_train_2[i_sim*10000,0]
#    color= [ang/90, 0.0, (90-ang)/90]
#    plt.plot(f_cont, color=color)
#
#n_plot= 100
#plt.figure()
#for i_sim in range(n_plot):
#    f_cont= X_train_3[i_sim*10000,:N+1]
#    ang= Y_train_3[i_sim*10000,0]
#    color= [ang/90, 0.0, (90-ang)/90]
#    plt.plot(f_cont, color=color)


#######################################################################

X_train= np.concatenate( (X_train_1, X_train_2, X_train_3), axis=0 )
Y_train= np.concatenate( (Y_train_1, Y_train_2, Y_train_3), axis=0 )
#X_train= X_train_2
#Y_train= Y_train_2

X_test= np.concatenate( (X_test_1, X_test_2, X_test_3), axis=0 )
Y_test= np.concatenate( (Y_test_1, Y_test_2, Y_test_3), axis=0 )
#X_test= X_test_2
#Y_test= Y_test_2









###############################################################################
'''
###   SVR

clf = svm.SVR()

clf.fit(X_train, Y_train)

Y_test_prd= clf.predict(X_test)

plt.figure(), plt.plot(Y_test, Y_test_prd, '.')

print(np.mean(  (  Y_test- Y_test_prd )**2  ))
'''
###############################################################################










###############################################################################

###  MLP



#X_train= X_train[:,:N+1]
#X_test = X_test[:,:N+1]
#X_train_1= X_train_1[:,:N+1]
#X_test_1 = X_test_1[:,:N+1]
#X_train_2= X_train_2[:,:N+1]
#X_test_2 = X_test_2[:,:N+1]
#X_train_3= X_train_3[:,:N+1]
#X_test_3 = X_test_3[:,:N+1]



gpu_ind= 1

L_Rate = 1.0e-6


#n_feat_vec= np.array([11, 40, 40, 30, 20, 1])
#n_feat_vec= np.array([22, 20, 40, 60, 40, 20, 1])
#n_feat_vec= np.array([22, 30, 60, 80, 80, 60, 30, 1])
#n_feat_vec= np.array([N+1, 30, 60, 80, 80, 60, 30, 1])
n_feat_vec= np.array([2*N+2, 30, 60, 80, 80, 60, 30, 1])


X = tf.placeholder("float32", [None, n_feat_vec[0]])
Y = tf.placeholder("float32", [None, n_feat_vec[-1]])

learning_rate = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")


Y_p = dk_model.davood_reg_net(X, n_feat_vec, p_keep_hidden, bias_init=0.001)

cost= tf.reduce_mean( tf.pow(( Y - Y_p ), 2) )
cost2= tf.reduce_mean( tf.div( tf.pow(( Y - Y_p ), 2) , tf.pow(( Y + 5.0 ), 3) ) )

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)


saver = tf.train.Saver(max_to_keep=50)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



i_global = 0
best_test = 0

i_eval = -1



batch_size = 1000
n_epochs = 5000


n_train= X_train.shape[0]
n_test = X_test.shape[0]

test_interval = n_train//batch_size * 10


for epoch_i in range(n_epochs):
    
    for i_train in range(n_train//batch_size):
        
        q= np.random.randint(0,n_train,(batch_size))
        batch_x = X_train[q, :].copy()
        batch_y = Y_train[q, :].copy()
        
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, learning_rate: L_Rate, p_keep_hidden: 1.0})
        
        i_global += 1
        
        if i_global % test_interval == 0:
            
            i_eval += 1
            
            print('\n', epoch_i, i_train, i_global)
            
            cost_v = np.zeros(n_train//batch_size)
            
            for i_v in tqdm(range(n_train//batch_size), ascii=True):
                
                batch_x = X_train[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
                batch_y = Y_train[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
                
                cost_v[i_v]= sess.run(cost, feed_dict={X: batch_x, Y: batch_y, learning_rate: L_Rate, p_keep_hidden: 1.0})
            
            print('Train cost  %.3f' % cost_v.mean())
            
            cost_v = np.zeros(n_test//batch_size)
            
            for i_v in tqdm(range(n_test//batch_size), ascii=True):
                
                batch_x = X_test[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
                batch_y = Y_test[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
                
                cost_v[i_v]= sess.run(cost, feed_dict={X: batch_x, Y: batch_y, learning_rate: L_Rate, p_keep_hidden: 1.0})
            
            print('Test cost   %.3f' % cost_v.mean())



#temp_path = base_dir + 'MLP-model/model_saved_diff_gtable_9.ckpt'
temp_path = base_dir + 'MLP-model/model_saved_hardi_5.ckpt'
#temp_path = base_dir + 'MLP-model/model_saved_hardi_6_csf.ckpt'
#temp_path = base_dir + 'MLP-model/model_saved_irontract__snr_20.ckpt'


#saver.save(sess, temp_path)

saver.restore(sess, temp_path)


n_rows, n_cols = 2, 4
fig, ax = plt.subplots(figsize=(16, 10), nrows=n_rows, ncols=n_cols)


Y_test_prd= np.zeros(Y_test.shape)

for i_v in tqdm(range(n_test//batch_size), ascii=True):
                
    batch_x = X_test[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
    
    Y_test_prd[i_v * batch_size:(i_v + 1) * batch_size]= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})

plt.subplot(n_rows, n_cols, 1), plt.plot(Y_test[::10], Y_test_prd[::10], '.'), plt.plot(np.arange(100), np.arange(100), '.r')
plt.subplot(n_rows, n_cols, 5), plt.hist(Y_test - Y_test_prd, bins=50);

print( np.sqrt( np.mean(  (  Y_test- Y_test_prd )**2  )) )



###############################################################################

Y_test_prd_1= np.zeros(Y_test_1.shape)

for i_v in tqdm(range(X_test_1.shape[0]//batch_size), ascii=True):
    batch_x = X_test_1[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
    Y_test_prd_1[i_v * batch_size:(i_v + 1) * batch_size]= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})

plt.subplot(n_rows, n_cols, 2), plt.plot(Y_test_1[::100], Y_test_prd_1[::100], '.'), plt.plot(np.arange(100), np.arange(100), '.r')       
plt.subplot(n_rows, n_cols, 6), plt.hist(Y_test_1 - Y_test_prd_1, bins=50);

print( np.sqrt( np.mean(  (  Y_test_1- Y_test_prd_1 )**2  )) )

Y_test_prd_2= np.zeros(Y_test_2.shape)

for i_v in tqdm(range(X_test_2.shape[0]//batch_size), ascii=True):
    batch_x = X_test_2[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
    Y_test_prd_2[i_v * batch_size:(i_v + 1) * batch_size]= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})

plt.subplot(n_rows, n_cols, 3), plt.plot(Y_test_2[::100], Y_test_prd_2[::100], '.'), plt.plot(np.arange(100), np.arange(100), '.r')       
plt.subplot(n_rows, n_cols, 7), plt.hist(Y_test_2 - Y_test_prd_2, bins=50);

print( np.sqrt( np.mean(  (  Y_test_2- Y_test_prd_2 )**2  )) )

Y_test_prd_3= np.zeros(Y_test_3.shape)

for i_v in tqdm(range(X_test_3.shape[0]//batch_size), ascii=True):
    batch_x = X_test_3[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
    Y_test_prd_3[i_v * batch_size:(i_v + 1) * batch_size]= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})

plt.subplot(n_rows, n_cols, 4), plt.plot(Y_test_3[::100], Y_test_prd_3[::100], '.'), plt.plot(np.arange(100), np.arange(100), '.r')
plt.subplot(n_rows, n_cols, 8), plt.hist(Y_test_3 - Y_test_prd_3, bins=50);

print( np.sqrt( np.mean(  (  Y_test_3- Y_test_prd_3 )**2  )) )








##############################################################################



snr= 20
N= 15
M= 15

sphere= 'dipy'
sphere_size= 5000

if sphere=='mine':
    Xp, Yp, Zp= crl_aux.distribute_on_sphere(sphere_size)
    sphere = dipysphere.Sphere(Xp, Yp, Zp)
else:
    sphere = get_sphere('symmetric724')

v, _ = sphere.vertices, sphere.faces



n_fib= 2



if n_fib==1:
    CSF_range= [0.0, 0.50]
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    true_fibers= (1-CSF)/n_fib* true_fibers
elif n_fib==2:
    CSF_range= [0.0, 0.30]
    Fr1_range= [0.3, 0.5]
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
elif n_fib==3:
    CSF_range= [0.0, 0.20]
    Fr1_range= [0.3, 0.4]
    Fr2_range= [0.2, 0.3]
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])




s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, 0.0021, 0.00050, 0.003)
s= crl_aux.add_rician_noise(s, snr=snr)


###   One at a time

Ang= np.zeros( (v.shape[0], 2) )

for i_sim in range(v.shape[0]):
    
    test_dir= v[i_sim,:]
    
    f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
    
    batch_x= mb.repmat(f_cont.flatten('F')[:,np.newaxis], 1,1000).T
#    batch_x= mb.repmat(f_cont, 1,1000).T
    
    y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})
    
    ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
    
    Ang[i_sim,0]= y_pred[0]
    Ang[i_sim,1]= ang
    


### All sphere

batch_size = v.shape

batch_x= crl_dci.compute_angular_feature_vector_cont_full_sphere(s, b_vecs_test.T, v, N= N, M= M, full_circ= False)

y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})

Ang[:,0:1]= y_pred







ang_t, ang_p= Ang[:,1], Ang[:,0]

plt.figure(), plt.plot(ang_t, ang_p, '.')

ang_s= crl_aux.smooth_spherical(v.T, ang_p, n_neighbor=5, n_outlier=0, power=20, method='1', div=60, s=1.0)

ang_s2= crl_aux.smooth_spherical_fast(v.T, ang_p, n_neighbor=5, n_outlier=0, power= 20, method='1', div=60, s=1.0)
ang_s3= crl_aux.smooth_spherical(v.T, ang_p, n_neighbor=5, n_outlier=0, power=20,  method='2', div=50, s=1.0)

plt.plot(ang_t, ang_s, '.r')
plt.plot(ang_t, ang_s2, '.k')
plt.plot(ang_t, ang_s3, '.g')

print(np.mean(  (  ang_p- ang_t )**2  ))
print(np.mean(  (  ang_s- ang_t )**2  ))
print(np.mean(  (  ang_s2- ang_t )**2  ))
print(np.mean(  (  ang_s3- ang_t )**2  ))



theta_thr= 15
ang_res= 10

peaks_ind= np.where(ang_s3<theta_thr )[0]

v_sel, labels, n_pred, pred_fibers= crl_aux.spherical_clustering(v, ang_s3, theta_thr=theta_thr, ang_res=ang_res, max_n_cluster=3, symmertric_sphere=False)

crl_aux.plot_odf_and_fibers( v_sel , np.ones(len(peaks_ind)), true_fibers, true_resp=None, pred_fib=pred_fibers, pred_resp=None, N= 1000)

pred_fibers, pred_resps= crl_dci.find_dominant_fibers_dipy_way(sphere, 1/ang_s, 30, 2, peak_thr=.01, optimize=False, Psi= None, opt_tol=1e-7)
to_keep= np.where(1/pred_resps<25)[0]
pred_fibers= pred_fibers[:,to_keep]

crl_aux.plot_odf_and_fibers( v_sel , np.ones(len(peaks_ind)), true_fibers, true_resp=None, pred_fib=pred_fibers, pred_resp=None, N= 1000)





#plt.figure(), plt.plot(Ang[:,1], Ang[:,0], '.')
#
##pred_fibers, pred_resp= crl_dci.find_dominant_fibers(v, 1/(Ang[:,0]+0), np.pi/6, n_fib)
##crl_aux.plot_odf_and_fibers(v.T, 1/(Ang[:,0]+0), true_fibers, true_resp=None, pred_fib=pred_fibers, pred_resp=None, N= 1000)
#
#
#theta_thr= 20
#peaks_ind= np.where( Ang[:,0]<theta_thr )[0]
#
##crl_aux.plot_odf_and_fibers(v[peaks_ind].T, np.ones(len(peaks_ind)), true_fibers, true_resp=None, pred_fib=pred_fibers, pred_resp=None, N= 1000)
#
#
#v_sel, labels, n_pred, pred_fibers= crl_aux.spherical_clustering(v, Ang[:,0], theta_thr=theta_thr, ang_res= 12, max_n_cluster=3)
#
#
##crl_aux.plot_odf_and_fibers( v_sel , labels+1, true_fibers, true_resp=None, pred_fib=pred_fibers, pred_resp=None, N= 1000)
#crl_aux.plot_odf_and_fibers( v_sel , np.ones(len(peaks_ind)), true_fibers, true_resp=None, pred_fib=pred_fibers, pred_resp=None, N= 1000)
#

# smoothing
#
#ang_s= crl_aux.smooth_spherical(v.T, Ang[:,0], n_neighbor=7)
#
#plt.figure(), plt.plot(Ang[:,1], Ang[:,0], '.')
#plt.plot(Ang[:,1], ang_s, '.r')
#
#pred_fibers, pred_resp= crl_dci.find_dominant_fibers(v, 1/(ang_s+0), np.pi/6, n_fib)
#
#crl_aux.plot_odf_and_fibers(v.T, 1/(Ang[:,0]+0), true_fibers, true_resp=None, pred_fib=pred_fibers, pred_resp=None, N= 1000)










'''theta = np.linspace(0., np.pi, 7)
phi = np.linspace(0., 2*np.pi, 9)
data = np.empty((theta.shape[0], phi.shape[0]))
data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
data[1:-1,1], data[1:-1,-1] = 1., 1.
data[1,1:-1], data[-2,1:-1] = 1., 1.
data[2:-2,2], data[2:-2,-2] = 2., 2.
data[2,2:-2], data[-3,2:-2] = 2., 2.
data[3,3:-2] = 3.
data = np.roll(data, 4, 1)


lats, lons = np.meshgrid(theta, phi)
from scipy.interpolate import SmoothSphereBivariateSpline
lut = SmoothSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                  data.T.ravel(), s=3.5)

temp1, temp2, temp3= lats.ravel(), lons.ravel(),   data.T.ravel()

lut = SmoothSphereBivariateSpline(temp1, temp2, temp3, s=3.5)

data_orig = lut(theta, phi)'''









































##  test on dhcp







































from __future__ import division

import numpy as np
import os
from numpy import dot
from dipy.core.geometry import sphere2cart
from dipy.core.geometry import vec2vec_rotmat
from dipy.reconst.utils import dki_design_matrix
from scipy.special import jn
from dipy.data import get_fnames
from dipy.core.gradients import gradient_table
import scipy.optimize as opt
import pybobyqa
from dipy.data import get_sphere
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn import linear_model
from sklearn.linear_model import OrthogonalMatchingPursuit
from dipy.direction.peaks import peak_directions
import spams
import dipy.core.sphere as dipysphere
from tqdm import tqdm
import crl_aux
import crl_dti
import crl_dci
from scipy.stats import f
from importlib import reload
import h5py
import dipy.reconst.sfm as sfm
import dipy.data as dpd
from dipy.viz import window, actor
import dipy.direction.peaks as dpp
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.forecast import ForecastModel
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.denoise.localpca import localpca
from dipy.denoise.localpca import mppca
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.viz import has_fury
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.viz import colormap
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
import nibabel as nib
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.data import small_sphere
from dipy.direction import ProbabilisticDirectionGetter
from dipy.direction import ClosestPeakDirectionGetter
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
import numpy as np
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from fury import actor, window
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_trk, save_trk
from dipy.io.utils import create_tractogram_header
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion

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
#
###   Create JHU atlas
#
#jhu_dir= '/media/nerossd2/atlases/JHU_neonate_SS/'
#
#bundles_2_read= ['00_cc']
#pr_lb= 50
#
#t2_jhu= sitk.ReadImage( jhu_dir + 'JHU_neonate_nonlinear_t2ss.img')
#t2_jhu.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
#t2_jhu_np= sitk.GetArrayFromImage(t2_jhu)
#t2_jhu_np= np.transpose(t2_jhu_np, [2,1,0])
#
#lb_jhu_np= np.zeros(t2_jhu_np.shape)
#bundle_count= -1
#for bundle_2_read in bundles_2_read:
#    bundle_count+= 1
#    temp_img= sitk.ReadImage( jhu_dir + '/bundles/' + bundles_2_read[bundle_count] + '.img')
#    temp_img_np= sitk.GetArrayFromImage(temp_img)
#    temp_img_np= np.transpose(temp_img_np, [2,1,0])
#    lb_jhu_np[temp_img_np>pr_lb]= bundle_count+1
#
#lb_jhu_np= np.transpose(lb_jhu_np, [2,1,0])
#lb_jhu= sitk.GetImageFromArray(lb_jhu_np)
#lb_jhu.SetDirection(t2_jhu.GetDirection())
#lb_jhu.SetOrigin(t2_jhu.GetOrigin())
#lb_jhu.SetSpacing(t2_jhu.GetSpacing())
#
#sitk.WriteImage(lb_jhu, jhu_dir + '/bundles/'  + 'lb_jhu.mhd')
#
#tn_jhu= sitk.ReadImage( jhu_dir + '/tensors/Dall2.img')
#tn_jhu_np= sitk.GetArrayFromImage(tn_jhu)
#tn_jhu_np= np.transpose(tn_jhu_np, [3,2,1,0])
#
#
###############################################################################




save_data_thumbs= True


CC_stats= np.zeros((500, 12))
CC_i= -1

min_age= 35.001
max_age= 36

subsample_g_table= True
subsample_g_table_mode= 'keep_bs'
b_keep= [0,1000] 
fraction_del= 0.75

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
                    
                    print('\n'*1, 'Processing subject: ', subj, ',   session:', sess_info.loc[j, 'session_id'], ',    age:', age_c)
                    
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
                    brain_mask= sitk.GetArrayFromImage(brain_mask_img)
                    brain_mask= np.transpose( brain_mask, [2,1,0] )
                    
                    ref_dir= brain_mask_img.GetDirection()
                    ref_org= brain_mask_img.GetOrigin()
                    ref_spc= brain_mask_img.GetSpacing()
                    
                    file_name= subject + '_' + session + '_desc-restore_T2w.nii.gz'
                    t2_img= sitk.ReadImage( ant_dir + file_name )
                    t2_img= dk_aux.resample_imtar_to_imref(t2_img, brain_mask_img, sitk.sitkBSpline, False)
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
                    pr_img= sitk.GetArrayFromImage(pr_img)
                    pr_img= np.transpose( pr_img, [2,1,0] )
                    
                    file_name= subject + '_' + session + '_desc-drawem9_space-T2w_dseg.nii.gz'
                    ts_img= sitk.ReadImage( ant_dir + file_name )
                    ts_img= dk_aux.resample_imtar_to_imref(ts_img, brain_mask_img, sitk.sitkNearestNeighbor, False)
                    ts_img= sitk.GetArrayFromImage(ts_img)
                    ts_img= np.transpose( ts_img, [2,1,0] )
                    
                    b_vals_full, b_vecs_full, d_img_full= b_vals.copy(), b_vecs.copy(), d_img.copy()
                    gtab_full = gradient_table(b_vals_full, b_vecs_full)
                    
                    if subsample_g_table:
                        
                        b_vals, b_vecs, keep_ind= crl_aux.subsample_g_table(b_vals, b_vecs, mode=subsample_g_table_mode, b_keep=b_keep, fraction_del=fraction_del)
                        
                        d_img= d_img[:,:,:,keep_ind]
                    
                    gtab = gradient_table(b_vals, b_vecs)
                    
                    
                    sx, sy, sz, _= d_img.shape
                    
                    
                    
                    
                    #skull= crl_aux.create_rough_skull_mask(d_img[:,:,:,0]>0, closing_radius= 4, radius= 6.0)
                    skull= crl_aux.skull_from_brain_mask(brain_mask, radius= 1.0)
                    skull_img= np.transpose(skull, [2,1,0])
                    skull_img= sitk.GetImageFromArray(skull_img)
                    skull_img.SetDirection(ref_dir)
                    skull_img.SetOrigin(ref_org)
                    skull_img.SetSpacing(ref_spc)
                    sitk.WriteImage(skull_img, dav_dir + 'skull1.mhd')
                    
                    
                    
                    
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
                    #elif method=='lpca':
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
                    
                    tenmodel = dti.TensorModel(gtab_full)
                    
                    tenfit = tenmodel.fit(d_img_full, brain_mask)
                    
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
                    FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
                    
                    RGB = color_fa(FA, tenfit.evecs)
                    
                    GFA_img= np.transpose(RGB, [2,1,0,3])
                    GFA_img= sitk.GetImageFromArray(GFA_img)
                    GFA_img.SetDirection(ref_dir)
                    GFA_img.SetOrigin(ref_org)
                    GFA_img.SetSpacing(ref_spc)
                    sitk.WriteImage(GFA_img, dav_dir + 'GFA.mhd')
                    
                    
                    
                    ### CC
                    
                    cc_label= 48
                    cc_pixels= np.where(pr_img==cc_label)
                    cc_x_min, cc_x_max, cc_y_min, cc_y_max, cc_z_min, cc_z_max= \
                            cc_pixels[0].min(), cc_pixels[0].max(), cc_pixels[1].min(), cc_pixels[1].max(), cc_pixels[2].min(), cc_pixels[2].max()
                    
                    slc= (cc_z_min+cc_z_max)//2
                    
                    ###   response estimation
                    
                    sphere = get_sphere('repulsion724')
                    
                    response, ratio = auto_response(gtab_full, d_img_full[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max,:], roi_radius=300, fa_thr=0.7)
                    
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
                    CC_stats[CC_i,11]= np.percentile(fa_cc, 95)'''
                    
                    
                    
                    #################    dk_sD
                    
                    n_fib= 3
                    
                    fibr_img_full= np.zeros( (sx, sy, sz, 3, n_fib) )
                    resp_img_full= np.zeros( (sx, sy, sz, n_fib) )
                    resp_csf_full= np.zeros( (sx, sy, sz) )
                    
                    fibr_img= np.zeros( (sx, sy, sz, 3, n_fib) )
                    resp_img= np.zeros( (sx, sy, sz, n_fib) )
                    resp_csf= np.zeros( (sx, sy, sz) )
                    
                    fibr_img_ML= np.zeros( (sx, sy, sz, 3, n_fib) )
                    resp_img_ML= np.zeros( (sx, sy, sz, n_fib) )
                    
                    #sphere = get_sphere('symmetric724')
                    #sphere = get_sphere('repulsion724')
                    v, _ = sphere.vertices, sphere.faces
                    
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
                        H_full= crl_dci.sd_matrix(Lam, d_iso, b_vals_full, b_vecs_full.T, v, with_iso= with_iso)
                    
                    #mask= b0_img[:,:,:,0]>0
                    mask= d_img[:,:,:,0]>10
                    
                    b1_ind= np.where(b_vals>0)[0]
                    b0_ind= np.where(b_vals==0)[0]
                    
                    b1_img= d_img[:,:,:,b1_ind]
                    b0_img= np.mean( d_img[:,:,:,b0_ind] , axis=-1)
                    
                    for ix in tqdm(range(sx), ascii=True):
                        for iy in range(sy):
                            for iz in range(slc,slc+1): #iz in range(sz):#(slc,slc+1):
                                
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
                                    
                                    
                                    #s= b1_img[ix, iy, iz,:]/b0_img[ix, iy, iz,0]
                                    s= d_img_full[ix, iy, iz,:]
                                    
                                    f_0= np.ones( H_full.shape[1] ) / H_full.shape[1]
                                    f_n= crl_dci.RL_deconv(H_full, s, f_0, n_iter= 150)
                                    #f_n= crl_dci.dRL_deconv(H, s, f_0, n_iter= 1000)
                                    
                                    fibers , responses = crl_dci.find_dominant_fibers(v, f_n[:len(v)], min_angle= np.pi/6, n_fib=n_fib)
                                    
                                    fibr_img_full[ix, iy, iz,:,:]= fibers
                                    resp_img_full[ix, iy, iz,:]= responses
                                    resp_csf_full[ix, iy, iz]  = f_n[-1]
                                    
                                    
                                    
                                    
                                    s_ml= b1_img[ix,iy,iz,:] / b0_img[ix,iy,iz]
                                    
                                    Ang= np.zeros( (v.shape[0], 2) )
                                    
                                    for i_sim in range(v.shape[0]):
                                        
                                        test_dir= v[i_sim,:]
                                        
                                        f_cont= crl_dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
                                        
                                        #batch_x= mb.repmat(f_cont.flatten('F')[:,np.newaxis], 1,1000).T
                                        batch_x= mb.repmat(f_cont, 1,1000).T
                                        
                                        y_pred= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})
                                        
                                        ang= crl_dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
                                        
                                        Ang[i_sim,0]= y_pred[0]
                                        Ang[i_sim,1]= ang
                                        
                                    pred_fibers, pred_resp= crl_dci.find_dominant_fibers(v, 1/(Ang[:,0]+0), np.pi/6, n_fib)
                                    
                                    fibr_img_ML[ix, iy, iz,:,:]= fibers
                                    resp_img_ML[ix, iy, iz,:]= pred_resp
                                    
                                    
                    one_mask=   np.logical_and( resp_csf[:,:,:]<35, mask)
                    two_mask=   resp_img[:,:,:,1] / (resp_img[:,:,:,0]+1e-7)>0.75
                    two_mask=   np.logical_and( one_mask, two_mask )
                    thr_mask=   resp_img[:,:,:,2] / (resp_img[:,:,:,0]+1e-7)>0.85
                    thr_mask=   np.logical_and( thr_mask, two_mask )









































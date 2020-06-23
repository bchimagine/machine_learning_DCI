#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:25:24 2019

@author: ch209389
"""


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import numpy.polynomial.polynomial as poly
#import sys
from tqdm import tqdm
import SimpleITK as sitk
from scipy.interpolate import SmoothSphereBivariateSpline
from scipy.spatial import ConvexHull
import crl_dci

np.warnings.filterwarnings('ignore')




#def add_rician_noise(sig, snr=30):
#    
#    sig_shape= sig.shape
#    sigma = 1 / snr
#    
#    noise1 = np.random.normal(0, sigma, size=sig_shape)
#    noise2 = np.random.normal(0, sigma, size=sig_shape)
#    
#    sig_noisy= np.sqrt((sig + noise1) ** 2 + noise2 ** 2)
#    
#    return sig_noisy
    

def add_rician_noise(sig, snr=20, n_sim=20):
    
    s0= sig.copy()
    
    sig_shape= sig.shape
    sigma = 1 / snr
    
    sigma_v= np.logspace(np.log10(sigma/100), np.log10(sigma*100), n_sim)
    
    snr_v= np.zeros(n_sim)
    
    for i_sim in range(n_sim):
        
        sigma_c= sigma_v[i_sim]
        noise1 = np.random.normal(0, sigma_c, size=sig_shape)
        noise2 = np.random.normal(0, sigma_c, size=sig_shape)
        
        sig_noisy= np.sqrt((sig + noise1) ** 2 + noise2 ** 2)
        
        snr_v[i_sim]= 10* np.log10( np.mean(s0**2) / np.mean( (s0-sig_noisy)**2) )
        
    sigma= sigma_v[ np.argmin( np.abs(snr_v-snr)) ]
    
    noise1 = np.random.normal(0, sigma, size=sig_shape)
    noise2 = np.random.normal(0, sigma, size=sig_shape)
    
    sig_noisy= np.sqrt((sig + noise1) ** 2 + noise2 ** 2)
    
    return sig_noisy


def add_gaussian_noise(sig, sigma=0.1):
    
    sig_shape= sig.shape
        
    sig_noisy= sig + np.random.normal(0, sigma, size=sig_shape)
    
    return sig_noisy
    

def plot_3d(V, w):
    
    '''Plot vector field with unit vector directions in V and vector magnitudes in w.'''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(V[0,:]*w,V[1,:]*w, V[2,:]*w,  marker='o')


def plot_odf_and_fibers(V, w, true_fib_orig, true_resp=None, pred_fib=None, pred_resp=None, N= 1000):
    
    true_fib= true_fib_orig.copy()
    #print(true_fib)
    '''   .'''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(V[0,:]*w,V[1,:]*w, V[2,:]*w,  marker='o')
    
    for i in range(true_fib.shape[1]):
        true_fib[:,i]/= np.linalg.norm(true_fib[:,i])
        if not true_resp is None:
            true_fib[:,i]*= true_resp[i]
    
    if not pred_fib is None:
        for i in range(pred_fib.shape[1]):
            pred_fib[:,i]/= np.linalg.norm(pred_fib[:,i])
            if not pred_resp is None:
                pred_fib[:,i]*= pred_resp[i]
    
    w_max= w.max()
    
    for i in range(true_fib.shape[1]):
        p1= true_fib[:,i]
        p2= -true_fib[:,i]
        W= np.zeros( (3, N) )
        for i in range(N):
            W[:,i]= p1 + (p2-p1)/N*i
        ax.scatter(W[0,:]*w_max,W[1,:]*w_max, W[2,:]*w_max,  marker='.', c='r')
    
    if not pred_resp is None:
        w_max/= pred_resp.max()
    
    if not pred_fib is None:
        for i in range(pred_fib.shape[1]):
            p1= pred_fib[:,i]
            p2= -pred_fib[:,i]
            W= np.zeros( (3, N) )
            for i in range(N):
                W[:,i]= p1 + (p2-p1)/N*i
            ax.scatter(W[0,:]*w_max,W[1,:]*w_max, W[2,:]*w_max,  marker='.', c='k')
    
    w_max= 1.0* np.abs(np.array([ax.get_zlim(), ax.get_xlim(), ax.get_ylim()])).max()
    borders= np.array([ [w_max, w_max, w_max],
                        [w_max, w_max, -w_max],
                        [w_max, -w_max, w_max],
                        [w_max, -w_max, -w_max],
                        [-w_max, w_max, w_max],
                        [-w_max, w_max, -w_max],
                        [-w_max, -w_max, w_max],
                        [-w_max, -w_max, -w_max],
                       ]).T
    
    ax.scatter(borders[0,:], borders[1,:], borders[2,:],  marker='.', c='k')
    






def plot_odf_and_cone(V, w, true_fib_orig, true_resp=None, N= 1000, 
                      n_cone= 12, nu= np.pi/6, r= 1, N_cone= 20, r_u= 1, N_u= 200, cone_marker_size=2):
    
    true_fib= true_fib_orig.copy()
    #print(true_fib)
    '''   .'''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(V[0,:]*w,V[1,:]*w, V[2,:]*w,  marker='o')
    
    for i in range(true_fib.shape[1]):
        true_fib[:,i]/= np.linalg.norm(true_fib[:,i])
        if not true_resp is None:
            true_fib[:,i]*= true_resp[i]
    
    w_max= w.max()
    
    pred_fib= np.zeros((3,n_cone))
    
    for i_cone in range(n_cone):
        phi = 2*np.pi/n_cone*i_cone
        pred_fib[0,i_cone] = r*np.sin(nu)*np.cos(phi)
        pred_fib[1,i_cone] = r*np.sin(nu)*np.sin(phi)
        pred_fib[2,i_cone] = r*np.cos(nu)
        
    for i in range(true_fib.shape[1]):
        p1= true_fib[:,i]
        p2= -true_fib[:,i]
        W= np.zeros( (3, N) )
        for i in range(N):
            W[:,i]= p1 + (p2-p1)/N*i
        ax.scatter(W[0,:]*w_max,W[1,:]*w_max, W[2,:]*w_max,  marker='.', c='k')
    
    for i in range(pred_fib.shape[1]):
        p1= pred_fib[:,i]
        p2= -pred_fib[:,i]
        W= np.zeros( (3, N_cone) )
        for i in range(N_cone):
            W[:,i]= p1 + (p2-p1)/N_cone*i
        ax.scatter(W[0,:]*w_max,W[1,:]*w_max, W[2,:]*w_max,  marker='.', c='r', s=cone_marker_size)
    
    u_vector= np.zeros((N_u, 3))
    u_vector[:,2]= np.linspace(-r_u, r_u, N_u)
    ax.scatter(u_vector[:,0], u_vector[:,1], u_vector[:,2], marker='.', c='r', s=20)
    
    w_max= 1.0* np.abs(np.array([ax.get_zlim(), ax.get_xlim(), ax.get_ylim()])).max()
    borders= np.array([ [w_max, w_max, w_max],
                        [w_max, w_max, -w_max],
                        [w_max, -w_max, w_max],
                        [w_max, -w_max, -w_max],
                        [-w_max, w_max, w_max],
                        [-w_max, w_max, -w_max],
                        [-w_max, -w_max, w_max],
                        [-w_max, -w_max, -w_max],
                       ]).T
    
    ax.scatter(borders[0,:], borders[1,:], borders[2,:],  marker='.', c='k')
    








def plot_fibers(true_fib, true_resp=None, pred_fib=None, pred_resp=None, N= 1000):
    
    '''   .'''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(true_fib.shape[1]):
        true_fib[:,i]/= np.linalg.norm(true_fib[:,i])
        if not true_resp is None:
            true_fib[:,i]*= true_resp[i]
    
    if not pred_fib is None:
        for i in range(pred_fib.shape[1]):
            pred_fib[:,i]/= np.linalg.norm(pred_fib[:,i])
            if not pred_resp is None:
                pred_fib[:,i]*= pred_resp[i]
    
    w_max= 1.0
    
    for i in range(true_fib.shape[1]):
        p1= true_fib[:,i]
        p2= -true_fib[:,i]
        W= np.zeros( (3, N) )
        for i in range(N):
            W[:,i]= p1 + (p2-p1)/N*i
        ax.scatter(W[0,:]*w_max,W[1,:]*w_max, W[2,:]*w_max,  marker='.', c='r')
    
    if not pred_resp is None:
        w_max/= pred_resp.max()
    
    if not pred_fib is None:
        for i in range(pred_fib.shape[1]):
            p1= pred_fib[:,i]
            p2= -pred_fib[:,i]
            W= np.zeros( (3, N) )
            for i in range(N):
                W[:,i]= p1 + (p2-p1)/N*i
            ax.scatter(W[0,:]*w_max,W[1,:]*w_max, W[2,:]*w_max,  marker='.', c='k')
    
    w_max= 1.0* np.abs(np.array([ax.get_zlim(), ax.get_xlim(), ax.get_ylim()])).max()
    borders= np.array([ [w_max, w_max, w_max],
                        [w_max, w_max, -w_max],
                        [w_max, -w_max, w_max],
                        [w_max, -w_max, -w_max],
                        [-w_max, w_max, w_max],
                        [-w_max, w_max, -w_max],
                        [-w_max, -w_max, w_max],
                        [-w_max, -w_max, -w_max],
                       ]).T
    
    ax.scatter(borders[0,:], borders[1,:], borders[2,:],  marker='.', c='k')
    
    
    
def random_spherical( N=1 ):
    
    '''Generate random points uniformly distributed on the unit sphere.'''
    
    theta= np.random.rand(N)
    
    theta= np.arccos(2*theta-1)
        
    phi= np.random.rand(N)*2*np.pi
    
    return np.concatenate( (theta[:,np.newaxis] , phi[:,np.newaxis]), axis=1 )



def distribute_on_sphere(n, r=1):
    
    '''Generate a set of n random points with approximately equally distance between neighboring points 
        distributed on the unit sphere.'''
    
    xp= np.zeros(n)
    yp= np.zeros(n)
    zp= np.zeros(n)
    
    alpha = 4.0*np.pi*r*r/n
    d = np.sqrt(alpha)
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    count = 0
    for m in range (0,m_nu):
        nu = np.pi*(m+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for n in range (0,m_phi):
            phi = 2*np.pi*n/m_phi
            xp[count] = r*np.sin(nu)*np.cos(phi)
            yp[count] = r*np.sin(nu)*np.sin(phi)
            zp[count] = r*np.cos(nu)
            count = count +1
    
    xp= xp[:count]
    yp= yp[:count]
    zp= zp[:count]
    
    return xp, yp, zp



def show_fibers(bk_img, mk_img, fbr_mag, rsp_mag, slc, mag_scale, scale_with_response=True, direction='z', colored= False):

    background_img, mask_img, fibr_mag, resp_mag= bk_img.copy(), mk_img.copy(), fbr_mag.copy(), rsp_mag.copy()
    color= ['b','r', 'k', 'g', 'y']

    if direction=='x' or direction=='X':
        background_img= np.transpose(background_img, [2,1,0])
        mask_img=       np.transpose(mask_img, [2,1,0])
        fibr_mag=       np.transpose(fibr_mag, [2,1,0,3,4])
        temp= fibr_mag
        fibr_mag[:,:,:,0,:]= temp[:,:,:,2,:]
        fibr_mag[:,:,:,2,:]= temp[:,:,:,0,:]
        resp_mag=        np.transpose(resp_mag, [2,1,0,3])
    elif direction=='y' or direction=='Y':
        background_img= np.transpose(background_img, [0,2,1])
        mask_img=       np.transpose(mask_img, [0,2,1])
        fibr_mag=           np.transpose(fibr_mag, [0,2,1,3,4])
        temp= fibr_mag
        fibr_mag[:,:,:,1,:]= temp[:,:,:,2,:]
        fibr_mag[:,:,:,2,:]= temp[:,:,:,1,:]
        resp_mag=        np.transpose(resp_mag, [0,2,1,3])
    
    plt.figure(), plt.imshow( background_img[:,:,slc], cmap='gray')
    
    n_fib= fibr_mag.shape[-1]
    
    for i in tqdm(range(background_img.shape[0]), ascii=True):
        for j in range(background_img.shape[1]):
            
            if mask_img[i,j,slc]:
                
                f_c= fibr_mag[i,j,slc,:,:]
                if scale_with_response:
                    r_c= resp_mag[i,j,slc,:]*mag_scale
                else:
                    r_c= mag_scale*np.ones(n_fib)
                
                for i_fiber in range(n_fib):
                    if colored:
                        plt.plot( np.array([j-r_c[i_fiber]*f_c[1,i_fiber], j+r_c[i_fiber]*f_c[1,i_fiber] ]) ,
                              np.array([i-r_c[i_fiber]*f_c[0,i_fiber], i+r_c[i_fiber]*f_c[0,i_fiber] ]) , color[i_fiber]  )
                    else:
                        plt.plot( np.array([j-r_c[i_fiber]*f_c[1,i_fiber], j+r_c[i_fiber]*f_c[1,i_fiber] ]) ,
                              np.array([i-r_c[i_fiber]*f_c[0,i_fiber], i+r_c[i_fiber]*f_c[0,i_fiber] ]) , 'b'  )
    



def create_crl_phantom(N= 15, csf_fraction= 0.1):
    
    X= np.zeros( (N,N,15) )
    
    for i in range(int(4*N/15)):
        for j in range(N):
            X[i,j,3*0+1]= 1/3
    
    for i in range(N):
        for j in range(int(5*N/15), int(9*N/15)):
            X[i,j,3*1+0]= 1/3
    
    for i in range(N):
        X[N-1-i,i,3*2+0]= 1/(3*np.sqrt(2))
        X[N-1-i,i,3*2+1]= -1/(3*np.sqrt(2))
    for i in range(1,N):
        X[N-1-i,i-1,3*2+0]= 1/(3*np.sqrt(2))
        X[N-1-i,i-1,3*2+1]= -1/(3*np.sqrt(2))
    for i in range(N-1):
        X[N-i-1,i+1,3*2+0]= 1/(3*np.sqrt(2))
        X[N-i-1,i+1,3*2+1]= -1/(3*np.sqrt(2))
    
    for i in range(int(6*N/15), int(10*N/15)):
        for j in range(int(6*N/15), int(9*N/15)):
            X[i,j,3*3+2]= 1/3
    
    X_f= np.zeros( (N,N,5) )
    for i in range(5):
        X_f[:,:,i]= np.linalg.norm(X[:,:,3*i:3*i+3], axis=-1)
    
    X_n  = np.sum(X_f>0, axis=-1)
    
    for i in range(N):
        for j in range(N):
            if X_n[i,j]==1:
                for k in range(5):
                    if X_f[i,j,k]>0:
                        X[i,j,3*k:3*k+3]*= (1-csf_fraction)/(1/3)
            elif X_n[i,j]==2:
                for k in range(5):
                    if X_f[i,j,k]>0:
                        X[i,j,3*k:3*k+3]*= (1-csf_fraction)/(2/3)
            elif X_n[i,j]==3:
                for k in range(5):
                    if X_f[i,j,k]>0:
                        X[i,j,3*k:3*k+3]*= (1-csf_fraction)
    
    return X







def change_csf_fraction(f, csf_fraction= 0.1):
    
    f_n= np.zeros( f.shape )
    
    current_fiber_fraction= 0
    
    for i_f in range(f.shape[1]):
        
        current_fiber_fraction += np.linalg.norm( f[:,i_f] )
        
    for i_f in range(f.shape[1]):
        
        f_n[:,i_f]= f[:,i_f] * (1 - csf_fraction) / current_fiber_fraction
    
    return f_n








def smooth(x,window_len=5,window='hanning'):
    
    if window == 'flat': #moving average
        w= np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y1=np.convolve(w/w.sum(),x,mode='valid')
    
    y= np.concatenate( ( x[:window_len//2] , y1, x[-window_len//2+1:]  ) )
    
    return y





def seg_2_boundary_3d(x):
    
    a, b, c= x.shape
    
    y= np.zeros(x.shape)
    z= np.nonzero(x)
    
    if len(z[0])>1:
        x_sum= np.zeros(x.shape)
        for shift_x in range(-1, 2):
            for shift_y in range(-1, 2):
                for shift_z in range(-1, 2):
                    x_sum[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
        y= np.logical_and( x==1 , np.logical_and( x_sum>0, x_sum<27 ) )

    return y


def create_rough_skull_mask(brain_mask, closing_radius= 2, radius= 6.0):
    
    mask= brain_mask.copy()
    
    mask= sitk.GetImageFromArray(mask.astype(np.int))
    
    mask_closed= sitk.BinaryMorphologicalClosing(mask, closing_radius)
    
    mask_closed= sitk.GetArrayFromImage( mask_closed )
    
    mask_boundary= seg_2_boundary_3d(mask_closed)
    
    mask_boundary= sitk.GetImageFromArray(mask_boundary.astype(np.int))
    
    dist_image = sitk.SignedMaurerDistanceMap(mask_boundary, insideIsPositive=True, useImageSpacing=True, 
                                           squaredDistance=False)
    
    dist_image= - sitk.GetArrayFromImage(dist_image)
    dist_image= dist_image<radius
    
    skull= dist_image * mask_closed
    
    return skull

def skull_from_brain_mask(brain_mask, radius= 2.0):
    
    mask_copy= brain_mask.copy()
    
    size_x, size_y, size_z= brain_mask.shape
    mask= np.zeros((size_x+20, size_y+20, size_z+20))
    mask[10:10+size_x, 10:10+size_y, 10:10+size_z]= mask_copy
    
    mask_boundary= seg_2_boundary_3d(mask)
    
    mask_boundary= sitk.GetImageFromArray(mask_boundary.astype(np.int))
    
    dist_image = sitk.SignedMaurerDistanceMap(mask_boundary, insideIsPositive=True, useImageSpacing=True,
                                           squaredDistance=False)
    
    dist_image= - sitk.GetArrayFromImage(dist_image)
    dist_image= dist_image<radius
    
    skull= dist_image * mask
    
    '''mask= brain_mask.copy()
    
    mask_boundary= seg_2_boundary_3d(mask)
    
    mask_boundary= sitk.GetImageFromArray(mask_boundary.astype(np.int))
    
    dist_image = sitk.SignedMaurerDistanceMap(mask_boundary, insideIsPositive=True, useImageSpacing=True,
                                           squaredDistance=False)
    
    dist_image= - sitk.GetArrayFromImage(dist_image)
    dist_image= dist_image<radius
    
    skull= dist_image * mask'''
    
    skull= skull[10:10+size_x, 10:10+size_y, 10:10+size_z]
    
    return skull




def remove_skull(seed_mask, radius= 6.0):
    
    mask= seed_mask.copy()
    
    mask= mask.astype(np.int)
    
    points_temp = np.where( mask>0 )
    points= np.zeros( (len( points_temp[0] ), 3) )
    points[:,0]= points_temp[0]
    points[:,1]= points_temp[1]
    points[:,2]= points_temp[2]
    
    hull = ConvexHull(points)
    
    hull_img= np.zeros( mask.shape )
    
    hull_img[ points[hull.vertices,0].astype(np.int), points[hull.vertices,1].astype(np.int), 
              points[hull.vertices,2].astype(np.int) ]= 1    
    
    hull_img= sitk.GetImageFromArray(hull_img.astype(np.uint8))
    
    dist_image = sitk.SignedMaurerDistanceMap(hull_img, insideIsPositive=True, useImageSpacing=True, squaredDistance=False)
    dist_image= - sitk.GetArrayFromImage(dist_image)
    dist_image= dist_image>radius
    
    mask_new= mask * dist_image
    
    return mask_new






def subsample_g_table(b_vals, b_vecs, mode='keep_bs', b_keep=[0], fraction_del=0.5):
    
    b_vals_n, b_vecs_n= b_vals.copy(), b_vecs.copy()
    n_b= len(b_vals)
             
    if mode=='keep_bs':
        
        keep_ind= list()
        
        for i in range(n_b):
            
            if b_vals[i] in b_keep:
                
                keep_ind.append(i)
                
        b_vals_n, b_vecs_n= b_vals_n[keep_ind], b_vecs_n[:,keep_ind]
        keep_ind= np.array(keep_ind)
    
    elif mode=='random':
        
        np.random.seed(0)
        
        keep_ind= np.zeros((1,1), np.int16)
        b_unique_v= np.unique(b_vals[b_vals>-1])
        
        for b_unique in b_unique_v:
            
            temp= np.where(b_vals==b_unique)[0]
            np.random.shuffle(temp)
            temp= temp[int(fraction_del*len(temp)):][:,np.newaxis]
            
            keep_ind= np.concatenate((keep_ind, temp))
        
        keep_ind= keep_ind[1:]
        
        b_vals_n, b_vecs_n= np.squeeze( b_vals_n[keep_ind] ), np.squeeze( b_vecs_n[:,keep_ind] )
        
        keep_ind= np.squeeze(keep_ind)
    
    elif mode=='random_keep_bs':
        
        np.random.seed(0)
        
        keep_ind= np.zeros((1,1), np.int16)
        b_unique_v= np.unique(b_vals[b_vals>-1])
        
        for b_unique in b_unique_v:
            
            if b_unique in b_keep:
                
                temp= np.where(b_vals==b_unique)[0]
                np.random.shuffle(temp)
                temp= temp[int(fraction_del*len(temp)):][:,np.newaxis]
                
                keep_ind= np.concatenate((keep_ind, temp))
        
        keep_ind= keep_ind[1:]
        
        b_vals_n, b_vecs_n= np.squeeze( b_vals_n[keep_ind] ), np.squeeze( b_vecs_n[:,keep_ind] )
        
        keep_ind= np.squeeze(keep_ind)
    
    else:
        
        print('Subsampling not recognized!')
        
    return b_vals_n, b_vecs_n, keep_ind











def register_JHU_tract_2_dHCP(my_t2, my_mk, jh_t2, jh_mk, jh_lb):
    
    my_t2_np= sitk.GetArrayFromImage( my_t2)
    my_mk_np= sitk.GetArrayFromImage( my_mk)
    
    my_t2_mk_np= my_t2_np * my_mk_np
    my_t2_mk= sitk.GetImageFromArray(my_t2_mk_np)
    
    my_t2_mk.SetDirection(my_mk.GetDirection())
    my_t2_mk.SetOrigin(my_mk.GetOrigin())
    my_t2_mk.SetSpacing(my_mk.GetSpacing())
    
    fixed_image= my_t2_mk
    
    jh_t2_np= sitk.GetArrayFromImage( jh_t2)
    jh_mk_np= sitk.GetArrayFromImage( jh_mk)
    
    jh_t2_mk_np= jh_t2_np * (jh_mk_np>200)
    jh_t2_mk= sitk.GetImageFromArray(jh_t2_mk_np)
    
    jh_t2_mk.SetDirection(jh_mk.GetDirection())
    jh_t2_mk.SetOrigin(jh_mk.GetOrigin())
    jh_t2_mk.SetSpacing(jh_mk.GetSpacing())
    
    moving_image= jh_t2_mk
    
    moving_image.SetDirection( fixed_image.GetDirection() )
    
    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
                                                          moving_image, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(initial_transform)
    
    moving_image_2= resample.Execute(moving_image)
    
    registration_method = sitk.ImageRegistrationMethod()
    
    grid_physical_spacing = [50.0, 50.0, 50.0]
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
    
    final_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    registration_method.SetInitialTransform(final_transform)
    
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    
    registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                sitk.Cast(moving_image_2, sitk.sitkFloat32))
    
    final_transform_v = sitk.Transform(final_transform)
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(final_transform_v)
    
    #moving_image_3= resample.Execute(moving_image_2)
    
    tx= initial_transform
    tx.AddTransform(final_transform)
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  
    resample.SetTransform(tx)
    
    jh_lb_warped= resample.Execute(jh_lb)
    
    return jh_lb_warped



def Cart_2_Spherical(xyz):
    
    xy = xyz[:,0]**2 + xyz[:,1]**2
    r = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2])
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    phi[phi<0]= 2*np.pi+phi[phi<0]
    
    return r, theta, phi




def smooth_spherical(V_orig, S_orig, n_neighbor=5, n_outlier=2, power= 20, antipodal=False, method='1', div=50, s=1.0):
    
    if method=='1':
        
        V= V_orig.copy()
        S= S_orig.copy()
        
        #r, theta, phi= Cart_2_Spherical(V)
        
        S_smooth= np.zeros(S.shape)
        
        for i in range(len(S)):
            
            if antipodal:
                theta= np.arccos( np.clip( np.abs( np.dot( V[:,i], V ) ), 0, 1) )  
            else:
                theta= np.arccos( np.clip( np.dot( V[:,i], V ), -1, 1) )
            
            arg= np.argsort(theta)[:n_neighbor]
            
            sig= S[arg]
            wgt= np.dot( V[:,i], V[:,arg] )**power
            
            if n_outlier>0:
                sig_m= sig[0] #sig.mean()
                inliers= np.argsort( np.abs( sig- sig_m ) )[:n_neighbor-n_outlier]
                sig= sig[inliers]
                wgt= wgt[inliers]
            
            S_smooth[i]= np.sum( sig*wgt ) / np.sum( wgt )
            
    elif method=='2':
        
        V= V_orig.copy()
        S= S_orig.copy()
        
        r, theta, phi= Cart_2_Spherical(V.T)
        
        lats, lons = np.meshgrid(theta, phi)
        
        lut = SmoothSphereBivariateSpline(theta, phi, S/div, s=s)
        
        S_smooth= np.zeros(S.shape)
        
        for i in range(724):
            S_smooth[i]= div*lut(theta[i], phi[i])
        
    
    return S_smooth






def smooth_spherical_fast(V, S, n_neighbor=5, n_outlier=2, power= 20, antipodal=False, method='1', div=50,s=1.0):
    
    if method=='1':
        
        #r, theta, phi= Cart_2_Spherical(V)
        
        #S_smooth= np.zeros(S.shape)
        
        if antipodal:
            WT= np.dot( V.T, V )
            theta= np.arccos( np.clip( np.abs( WT ), 0, 1) )
            WT= WT**power
        else:
            WT= np.clip( np.dot( V.T, V ) , -1, 1)
            theta= np.arccos(  WT )
#            WT= np.power(WT, power)
            WT= WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT*WT
        
        arg= np.argsort(theta)[:,:n_neighbor]
        arg2= [ np.arange(len(S))[:,np.newaxis],arg ]
         
        #sig= S[arg]
        #wgt= WT[arg2]
        
        '''if n_outlier>0:
            sig_m= sig[0] #sig.mean()
            inliers= np.argsort( np.abs( sig- sig_m ) )[:n_neighbor-n_outlier]
            sig= sig[inliers]
            wgt= wgt[inliers]'''
        
        S_smooth= np.sum( S[arg]*WT[arg2], axis=1 ) / np.sum( WT[arg2], axis=1 )
        
    elif method=='2':
        
        r, theta, phi= Cart_2_Spherical(V.T)
        
        lats, lons = np.meshgrid(theta, phi)
        
        lut = SmoothSphereBivariateSpline(theta, phi, S/div, s=s)
        
        S_smooth= np.zeros(S.shape)
        
        for i in range(724):
            S_smooth[i]= div*lut(theta[i], phi[i])
        
    
    return S_smooth





def spherical_mean(V_doub, n_iter= 5, symmertric_sphere=False):
    
    if symmertric_sphere:
        
        n_vec= V_doub.shape[1]//2
        
        V_sing= np.zeros( (3,n_vec) )
        V_doub_ind= np.ones(n_vec*2)
        i_sing= -1
        
        for i in range(n_vec*2):
            
            if V_doub_ind[i]:
                
                i_sing+= 1
                V_sing[:,i_sing]= V_doub[:,i]
                
                cosines= np.abs( np.dot( V_doub.T , V_doub[:,i] ) )
                V_doub_ind[cosines>0.9999]= 0
        
        assert(i_sing==n_vec-1)
    
    else:
        
        cosines= np.dot( V_doub.T , V_doub[:,0]  )
        V_sing= V_doub[:,cosines>0.5]
        n_vec= V_sing.shape[1]
    
    q= V_sing[:,0]
    for i in range(1,n_vec):
        if np.dot( V_sing[:,i] , q )<0:
            V_sing[:,i] *= -1
    
    p= np.mean( V_sing, axis=1 )
    p/= np.linalg.norm(p)
    
    x_hat_mat= np.zeros(V_sing.shape)
    
    for i_iter in range(n_iter):
        
        cosines= np.dot( V_sing.T , p )
        theta=   np.arccos(cosines)
        sines = np.sqrt(1-cosines**2)
        
        for i in range(n_vec):
            x_hat_mat[:,i]= (V_sing[:,i] - p * cosines[i]) * ( theta[i] )/ ( sines[i])
        
        x_hat= np.mean( x_hat_mat, axis=1 )
        
        x_hat_norm= np.linalg.norm(x_hat)
        
        p= p * np.cos(x_hat_norm) + x_hat/x_hat_norm * np.sin(x_hat_norm)
    
    return p




def spherical_clustering(v, ang_est, theta_thr=20, ang_res= 10, max_n_cluster=3, symmertric_sphere=False):
    
    
    V= v[ang_est<theta_thr,:].T
    
    if V.shape[1]>0:
        
        #cos_thr=     np.cos(theta_thr*np.pi/180)
        cos_thr_res= np.cos(ang_res*np.pi/180)
        
        labels=  max_n_cluster * np.ones((V.shape[1]))
        
        labels[0]= 0
        label_count=0
        
        unassigned= np.where(labels==max_n_cluster)[0]
        
        while label_count<max_n_cluster and len(unassigned)>0:
            
            found_new= True
            
            while found_new:
                
                cosines= np.max( np.abs( np.dot( V[:,unassigned].T, V[:,labels==label_count] ) ) , axis=1 )
                close_ind= np.where(cosines>cos_thr_res)[0]
                new_assign= unassigned[close_ind]
                
                if len( new_assign)>0:
                    
                    labels[new_assign]= label_count
                    unassigned= np.where(labels==max_n_cluster)[0]
                    
                    if len(unassigned)==0:
                        
                        found_new= False
                    
                else:
                    
                    found_new= False
            
            if len(unassigned)>0:
                label_count+=1
                labels[ np.where(labels==max_n_cluster)[0][0] ] = label_count
                unassigned= np.where(labels==max_n_cluster)[0]
        
        
        n_cluster= label_count+1
        
        V_cent=  np.zeros((3,n_cluster))
        
        for i_cluster in range(n_cluster):
            
            V_to_cluster= V[:, labels==i_cluster ]
            
            if V_to_cluster.shape[1]>2:
                V_cent[:,i_cluster]= spherical_mean( V_to_cluster, symmertric_sphere=symmertric_sphere )
            else:
                V_cent[:,i_cluster]= V_to_cluster[:,0]
    
    else:
        
        ind_min= np.argmin(ang_est)
        V= V_cent= v[ind_min,:]
        labels= 0
        n_cluster= 1
    
    if V_cent.shape==(3,):
        V_cent= V_cent[:,np.newaxis]
    
    return V, labels, n_cluster, V_cent
        
    
    



def compute_WAAE(x_gt, x_pr, mask, normalize_truth=False, penalize_miss=False):
    
    assert(x_gt.shape==x_pr.shape)
    
    sx, sy, sz, n_fib, _ = x_gt.shape
    
    WAAE= np.zeros(n_fib)
    WAAE_count= np.zeros(n_fib)
    
    Error_matrix= np.zeros((sx, sy, sz, n_fib))
    
    x_gt_norm= np.zeros((sx, sy, sz, n_fib))
    x_pr_norm= np.zeros((sx, sy, sz, n_fib))
    
    for ix in range(sx):
        for iy in range(sy):
            for iz in range(sz):
                
                if mask[ix,iy,iz]:
                    
                    for i_fib in range(n_fib):
                        
                        gt_norm= np.linalg.norm( x_gt[ix, iy, iz, i_fib, :] )
                        if gt_norm>0:
                            x_gt_norm[ix, iy, iz, i_fib]= gt_norm
                            x_gt[ix, iy, iz, i_fib, :]/= gt_norm
                        
                        pr_norm= np.linalg.norm( x_pr[ix, iy, iz, i_fib, :] )
                        if pr_norm>0:
                            x_pr_norm[ix, iy, iz, i_fib]= pr_norm
                            x_pr[ix, iy, iz, i_fib, :]/= pr_norm
    
    for ix in range(sx):
        for iy in range(sy):
            for iz in range(sz):
                
                if mask[ix,iy,iz]:
                    
                    v_gt= x_gt[ix, iy, iz, :, :]
                    v_gt_norm= x_gt_norm[ix, iy, iz, :]
                    v_gt= v_gt[v_gt_norm>0,:]
                    v_gt_norm= v_gt_norm[v_gt_norm>0]
                    if normalize_truth:
                        v_gt_norm/= v_gt_norm.sum()
                    
                    v_pr= x_pr[ix, iy, iz, :, :]
                    v_pr_norm= x_pr_norm[ix, iy, iz, :]
                    v_pr= v_pr[v_pr_norm>0,:]
                    v_pr_norm= v_pr_norm[v_pr_norm>0]
                    if normalize_truth:
                        v_pr_norm/= v_pr_norm.sum()
                        
                    n_gt= v_gt.shape[0]
                    n_pr= v_pr.shape[0]
                    
                    
                    
                    if penalize_miss:
                        
                        if n_pr==0:
                            v_pr= np.ones((1,3))
                            v_pr/= np.linalg.norm(v_pr)
                            n_pr= 1
                    
                    if n_pr>0:
                        
                        error_current= 0
                        
                        for i_fib in range(n_gt):
                            
                            temp= crl_dci.compute_min_angle_between_vector_sets(v_pr.T, v_gt[i_fib:i_fib+1,:].T)
                            
                            error_current+= temp* v_gt_norm[i_fib]
                            
                            Error_matrix[ix,iy,iz, i_fib]= error_current
                        
                        WAAE[n_gt-1]+= error_current
                        WAAE_count[n_gt-1]+= 1
                        
                    '''elif n_pr>0:
                        
                        error_current= 0
                        
                        for i_fib in range(n_pr):
                            
                            temp= crl_dci.compute_min_angle_between_vector_sets(v_gt.T, v_pr[i_fib:i_fib+1,:].T)
                            
                            error_current+= temp* v_pr_norm[i_fib]
                        
                        WAAE[n_gt-1]+= error_current
                        WAAE_count[n_gt-1]+= 1'''
                    
    
    return Error_matrix, WAAE, WAAE_count, WAAE/WAAE_count





















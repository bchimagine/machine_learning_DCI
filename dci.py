#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""



import numpy as np
import scipy.optimize as opt
from scipy.stats import f
from dipy.reconst.recspeed import local_maxima
from dipy.reconst.recspeed import remove_similar_vertices
from scipy.interpolate import SmoothSphereBivariateSpline






def compute_min_angle_between_fibers(R, n_fib):
    
    max_cos= 0
    
    for i in range(n_fib-1):
        for j in range(i+1,n_fib):
            
            v1= np.array( [ np.sin(R[i*5+3])*np.cos(R[i*5+4]) , np.sin(R[i*5+3])*np.sin(R[i*5+4]) , np.cos(R[i*5+3]) ] )
            v2= np.array( [ np.sin(R[j*5+3])*np.cos(R[j*5+4]) , np.sin(R[j*5+3])*np.sin(R[j*5+4]) , np.cos(R[j*5+3]) ] )
            
            cur_cos= np.abs( np.dot(v1, v2) )
            
            if cur_cos>max_cos:
                
                max_cos= cur_cos
    
    min_ang= np.arccos(max_cos)*180/np.pi
    
    return min_ang




def MCMC_prob(R, sigma, b_vals, b_vecs, s, n_fib, sample_stride):
    
    '''prob= 1
    
    for i in range(n_fib):
        
        prob*= -1/( (1-R[5*i]+0.001) * np.log(1-R[5*i]+0.001) )
        
    for i in range(n_fib):
        
        prob*= np.sin( R[5*i+3] )
    
    prob*= 1/ sigma
    
    s_prd= polar_fibers_and_iso_simulate(R, n_fib, b_vals, b_vecs)
    
    prob*= np.prod( np.exp( - (s_prd[::sample_stride]-s[::sample_stride])**2/sigma**2 ) )'''
    
    prob1= 0
    
    for i in range(n_fib):
        
        prob1+= np.log( -1/( (1-(R[5*i]+0.001)) * np.log(1-(R[5*i]+0.001)) ) )
        
    for i in range(n_fib):
        
        prob1+= np.log( np.sin( R[5*i+3] ) )
    
    prob1+= np.log( 1/ sigma )
    
    s_prd= polar_fibers_and_iso_simulate(R, n_fib, b_vals, b_vecs)
    
    prob2= np.sum( - (s_prd[::sample_stride]-s[::sample_stride])**2/sigma**2 )
    
    prob= prob1 + prob2
    
    
    return prob, prob1, prob2







def polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= 45, n_try_angle= 100):
    
    R_0=        np.zeros( 5*n_fib+1 )
    bounds_lo=  np.zeros( 5*n_fib+1 )
    bounds_hi=  np.zeros( 5*n_fib+1 )
    
    good_separation= False
    
    i_try_angle= 0
    
    while not good_separation:
        
        for i in range(n_fib):
            R_0[5*i+3]= np.pi/10 + np.random.rand() * ( np.pi - np.pi/5)
            R_0[5*i+4]= np.random.rand() * 2 * np.pi
        
        i_try_angle+= 1
        separation_temp= compute_min_angle_between_fibers(R_0, n_fib)
        #print(separation_temp)
        
        if separation_temp>min_separation:
            good_separation= True
            
        if i_try_angle==n_try_angle:
            good_separation= True
            print('Fibers are not well-separated!')
        
    R_0[-1]= d_iso
    if n_fib>0:
        bounds_lo[-1]= d_iso*0.90
        bounds_hi[-1]= d_iso*1.10
    else:
        bounds_lo[-1]= d_iso*0.90
        bounds_hi[-1]= d_iso*1.10
    
    for i in range(n_fib):
        
        R_0[5*i+1]= lam_1
        bounds_lo[5*i+1]= lam_1*0.8
        bounds_hi[5*i+1]= lam_1*1.2
        
        R_0[5*i+2]= lam_2
        bounds_lo[5*i+2]= lam_2*0.8
        bounds_hi[5*i+2]= lam_2*1.2
        
        bounds_lo[5*i+3]= 0
        bounds_hi[5*i+3]= np.pi
        bounds_lo[5*i+4]= 0
        bounds_hi[5*i+4]= 2*np.pi
    
    if n_fib==1:
        R_0[0]= 0.7
        bounds_lo[0]= 0.05
        bounds_hi[0]= 1.0
    elif n_fib==2:
        R_0[0]= 0.4
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.75
        R_0[5]= 0.4
        bounds_lo[5]= 0.05
        bounds_hi[5]= 0.75
    elif n_fib==3:
        R_0[0]= 0.30
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.4
        R_0[5]= 0.30
        bounds_lo[5]= 0.05
        bounds_hi[5]= 0.3
        R_0[10]= 0.15
        bounds_lo[10]= 0.05
        bounds_hi[10]= 0.3
    elif n_fib==4:
        R_0[0]= 0.25
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.3
        R_0[5]= 0.25
        bounds_lo[5]= 0.05
        bounds_hi[5]= 0.3
        R_0[10]= 0.25
        bounds_lo[10]= 0.05
        bounds_hi[10]= 0.3
        R_0[15]= 0.25
        bounds_lo[15]= 0.05
        bounds_hi[15]= 0.3
    elif n_fib==5:
        R_0[0]= 0.20
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.3
        R_0[5]= 0.20
        bounds_lo[5]= 0.05
        bounds_hi[5]= 0.3
        R_0[10]= 0.20
        bounds_lo[10]= 0.05
        bounds_hi[10]= 0.3
        R_0[15]= 0.20
        bounds_lo[15]= 0.05
        bounds_hi[15]= 0.3
        R_0[20]= 0.20
        bounds_lo[20]= 0.05
        bounds_hi[20]= 0.3
    
    return R_0, bounds_lo, bounds_hi




def diamond_init_log(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, kappa=100):
    
    R_0=        np.zeros( 6*n_fib+2 )
    bounds_lo=  np.zeros( 6*n_fib+2 )
    bounds_hi=  np.zeros( 6*n_fib+2 )
    
    R_0[-2]= d_iso
    if n_fib>0:
        bounds_lo[-2]= d_iso*0.8
        bounds_hi[-2]= d_iso*1.2
    else:
        bounds_lo[-2]= d_iso*0.8
        bounds_hi[-2]= d_iso*1.2
    
    R_0[-1]= kappa
    if n_fib>0:
        bounds_lo[-1]= kappa*0.01
        bounds_hi[-1]= kappa*100
    else:
        bounds_lo[-1]= kappa*0.01
        bounds_hi[-1]= kappa*100
    
    for i in range(n_fib):
        
        R_0[6*i+1]= np.log(lam_1)
        bounds_lo[6*i+1]= np.log(lam_1*0.95)
        bounds_hi[6*i+1]= np.log(lam_1*1.05)
        
        R_0[6*i+2]= np.log(lam_2)
        bounds_lo[6*i+2]= np.log(lam_2*0.95)
        bounds_hi[6*i+2]= np.log(lam_2*1.05)
        
        R_0[6*i+3]= np.pi/10 + np.random.rand() * ( np.pi - np.pi/5)
        bounds_lo[6*i+3]= 0
        bounds_hi[6*i+3]= np.pi
        
        R_0[6*i+4]= np.random.rand() * 2 * np.pi
        bounds_lo[6*i+4]= 0
        bounds_hi[6*i+4]= 2*np.pi
        
        R_0[6*i+5]= kappa
        bounds_lo[6*i+5]= kappa*0.01
        bounds_hi[6*i+5]= kappa*100
    
    if n_fib==1:
        R_0[0]= 0.6
        bounds_lo[0]= 0.05
        bounds_hi[0]= 1.0
    elif n_fib==2:
        R_0[0]= 0.5
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.95
        R_0[6]= 0.5
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.95
    elif n_fib==3:
        R_0[0]= 0.4
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.85
        R_0[6]= 0.3
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.85
        R_0[12]= 0.3
        bounds_lo[12]= 0.05
        bounds_hi[12]= 0.85
    elif n_fib==4:
        R_0[0]= 0.35
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.6
        R_0[6]= 0.25
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.6
        R_0[12]= 0.20
        bounds_lo[12]= 0.05
        bounds_hi[12]= 0.6
        R_0[18]= 0.15
        bounds_lo[18]= 0.05
        bounds_hi[18]= 0.4
    elif n_fib==5:
        R_0[0]= 0.25
        bounds_lo[0]= 0.25
        bounds_hi[0]= 0.5
        R_0[6]= 0.20
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.5
        R_0[12]= 0.20
        bounds_lo[12]= 0.05
        bounds_hi[12]= 0.5
        R_0[18]= 0.15
        bounds_lo[18]= 0.05
        bounds_hi[18]= 0.5
        R_0[24]= 0.15
        bounds_lo[24]= 0.05
        bounds_hi[24]= 0.5
    
    return R_0, bounds_lo, bounds_hi




def polar_fibers_and_iso_resid(R, n_fib, b, q, data, weight):
    
    y = (1-R[0:-1:5].sum()) * np.exp( - R[-1]*b )
    
    for i in range(n_fib):
        
        y+= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    residuals = data - y
    
    residuals= weight * residuals
    
    return residuals






def polar_fibers_and_iso_simulate(R, n_fib, b, q):
    
    y = (1-R[0:-1:5].sum()) * np.exp( - R[-1]*b )
    
    for i in range(n_fib):
        
        y+= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    return y



def diamond_resid_log(R, n_fib, b, q, data, weight):
    
    y = (1-R[0:-2:6].sum()) * ( 1 + ( R[-2]*b) / R[-1] )** (-R[-1])
    
    for i in range(n_fib):
        
        y+= R[i*6]* ( 1 + b * ( np.exp(R[i*6+2]) + (np.exp(R[i*6+1])-np.exp(R[i*6+2]))* 
             ( q[:,0]*np.sin(R[i*6+3])*np.cos(R[i*6+4]) + q[:,1]*np.sin(R[i*6+3])*np.sin(R[i*6+4]) + q[:,2]*np.cos(R[i*6+3]) )**2 ) / R[i*6+5] ) ** (- R[i*6+5] )
    
    residuals = data - y
    
    residuals= weight * residuals
    
    return residuals




def diamond_simulate_log(R, n_fib, b, q):
    
    y = (1-R[0:-2:6].sum()) * ( 1 + (R[-2]*b) / R[-1]  )** (-R[-1])
    
    for i in range(n_fib):
        
        y+= R[i*6]* ( 1 + b * ( np.exp(R[i*6+2]) + (np.exp(R[i*6+1])-np.exp(R[i*6+2]))* 
             ( q[:,0]*np.sin(R[i*6+3])*np.cos(R[i*6+4]) + q[:,1]*np.sin(R[i*6+3])*np.sin(R[i*6+4]) + q[:,2]*np.cos(R[i*6+3]) )**2 ) / R[i*6+5] ) ** (- R[i*6+5] )
    
    return y




def model_selection_f_test(s, b_vals, b_vecs, m_max= 3, threshold= 20, condition_mode= 'F_val', model='DIAMOND'):
    
    m_selected= False
    m_opt= -1
    m= 0
    wt_temp= s**2
    
    while not m_selected:
        
        true_numbers= m
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = polar_fibers_and_iso_init(true_numbers, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(true_numbers, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = diamond_init_log(true_numbers, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        else:
            print('Model type unidentified.')
            return np.nan
        
        if model=='ball_n_sticks':
            solution = opt.least_squares(polar_fibers_and_iso_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals, b_vecs,
                                                s,
                                                wt_temp**0.0))
            ss= polar_fibers_and_iso_simulate(solution.x, true_numbers, b_vals, b_vecs)
        elif model=='DIAMOND':
            '''solution = opt.least_squares(diamond_resid, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            ss= diamond_simulate(solution.x, true_numbers, b_vals, b_vecs)'''
            solution = opt.least_squares(diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            ss= diamond_simulate_log(solution.x, true_numbers, b_vals, b_vecs)
            
        ss_fit= ( ss-s )**2
        
        if m==0:
            
            RSS1= ss_fit.sum()
            p1= len(R_init)
            
        else:
            
            RSS2= ss_fit.sum()
            p2= len(R_init)
            
            f_val= ( (RSS1-RSS2) / (p2-p1) )   /   ( RSS2 / ( len(b_vals)-p2 ) )
            
            if condition_mode== 'F_prob':
                f_stat= f.cdf( f_val , p2-p1, len(b_vals)-p2 )
                cond= f_stat< 1-threshold or m==m_max
            elif condition_mode== 'F_val':
                cond= f_val< threshold or m==m_max
            else:
                print('Condition mode unidentified!')
                return np.nan
            
            if cond:
                
                m_opt= m-1
                m_selected= True
                
            else:
                
                RSS1= ss_fit.sum()
                p1= len(R_init)
                    
        
        m+= 1
        
    
    return m_opt
                



def model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= 3, threshold= 0.5, model='DIAMOND', delta_mthod= False):
    
    m_selected= False
    m_opt= -1
    m= 0
    wt_temp= s**2
    
    while not m_selected:
        
        true_numbers= m
        
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = polar_fibers_and_iso_init(true_numbers, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(true_numbers, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = diamond_init_log(true_numbers, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
        else:
            print('Model type unidentified.')
            return np.nan
        
        ss_bs=       np.zeros(len(b_vals))
        ss_bs_count= np.zeros(len(b_vals))
        
        ss_bs_matrix=       np.zeros( (len(b_vals), n_bs) )
        ss_bs_count_matrix= np.zeros( (len(b_vals), n_bs) )
        
        for i_bs in range(n_bs):
            
            np.random.seed(i_bs)
            
            ind_bs_train= np.random.randint(0,len(b_vals), b_vals.shape)
            ind_bs_test= [i not in ind_bs_train   for i in range(len(b_vals))]
            ind_counts = [np.sum(ind_bs_train==i) for i in range(len(b_vals))]
            
            if model=='ball_n_sticks':
                
                solution = opt.least_squares(polar_fibers_and_iso_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals[ind_bs_train], b_vecs[ind_bs_train,:],
                                                s[ind_bs_train],
                                                wt_temp[ind_bs_train]**0.0))
            elif model=='DIAMOND':
                
                '''solution = opt.least_squares(diamond_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals[ind_bs_train], b_vecs[ind_bs_train,:],
                                                s[ind_bs_train],
                                                wt_temp[ind_bs_train]**0.0))'''
                
                solution = opt.least_squares(diamond_resid_log, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals[ind_bs_train], b_vecs[ind_bs_train,:],
                                                s[ind_bs_train],
                                                wt_temp[ind_bs_train]**0.0))
                
                '''solution = pybobyqa.solve(polar_fibers_and_iso_resid_bbq, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals[ind_bs_train], b_vecs[ind_bs_train,:],
                                                s[ind_bs_train],
                                                wt_temp[ind_bs_train]**0.5),
                                                rhobeg= 0.002, scaling_within_bounds= True, seek_global_minimum=False)'''
            
            if model=='ball_n_sticks':
                ss= polar_fibers_and_iso_simulate(solution.x, true_numbers, b_vals, b_vecs)
            elif model=='DIAMOND':
                #ss= diamond_simulate(solution.x, true_numbers, b_vals, b_vecs)
                ss= diamond_simulate_log(solution.x, true_numbers, b_vals, b_vecs)
            
            ss_bs[ind_bs_test]+= ( ss[ind_bs_test]-s[ind_bs_test] )**2
            ss_bs_count[ind_bs_test]+= 1
            
            ss_bs_matrix[ind_bs_test,i_bs]= ( ss[ind_bs_test]-s[ind_bs_test] )**2
            ss_bs_count_matrix[:,i_bs]= ind_counts
            
        ss_bs[ss_bs_count>0]= ss_bs[ss_bs_count>0]/ ss_bs_count[ss_bs_count>0]
        #ss_bs= ss_bs[ss_bs_count>0]
        E_bs= ss_bs.mean()
        
        if model=='ball_n_sticks':
            solution = opt.least_squares(polar_fibers_and_iso_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals, b_vecs,
                                                s,
                                                wt_temp**0.0))
            ss= polar_fibers_and_iso_simulate(solution.x, true_numbers, b_vals, b_vecs)
        elif model=='DIAMOND':
            '''solution = opt.least_squares(diamond_resid, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            ss= diamond_simulate(solution.x, true_numbers, b_vals, b_vecs)'''
            solution = opt.least_squares(diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            '''solution = pybobyqa.solve(polar_fibers_and_iso_resid_bbq, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.5),
                                    rhobeg= 0.002, scaling_within_bounds= True, seek_global_minimum=False)'''
                                          
            ss= diamond_simulate_log(solution.x, true_numbers, b_vals, b_vecs)
        
        ss_fit= ( ss-s )**2
        E_fit= ss_fit.mean()
        
        E_632= 0.368*E_fit + 0.632*E_bs
        
        if m==0:
            
            E_632_m_n_1= E_632
            #E_fit_m_n_1= E_fit
            E_bs_m_n_1 = E_bs
            ss_bs_m_n_1= ss_bs.copy()
            ss_bs_matrix_m_n_1= ss_bs_matrix.copy()
            #ss_bs_count_matrix_m_n_1= ss_bs_count_matrix.copy()
            
        else:
            
            del_632= E_632_m_n_1- E_632
            #del_fit= E_fit_m_n_1- E_fit
            del_bs = E_bs_m_n_1 - E_bs
            
            del_bs_ss= ss_bs_m_n_1- ss_bs
            
            if delta_mthod:
                
                del_ss_bs_matrix= ss_bs_matrix_m_n_1- ss_bs_matrix
                
                q_hat= np.sum(del_ss_bs_matrix, axis=0)
                
                N_hat= np.mean( ss_bs_count_matrix, axis=1 )
                
                numer= ss_bs_count_matrix - N_hat[:,np.newaxis]
                
                numer= np.matmul(numer, q_hat)
                
                denom= np.sum( ss_bs_count_matrix==0, axis=1 )
                
                D= (2+ 1/(len(b_vals)-1)) * ( del_bs_ss - del_bs_ss.mean() )/ len(b_vals) + (denom>0) * numer/ (denom+1e-7)
                
                SE_BS= np.linalg.norm(D)
                
            else:
                
                SE_BS= np.sqrt( np.sum( ( del_bs_ss - del_bs_ss.mean() )**2 ) / ( len(b_vals)**2 ) )
            
            SE_632= del_632/del_bs * SE_BS
            
            if del_632 < threshold*SE_632 or m==m_max:
                
                m_opt= m-1
                m_selected= True
                
            else:
                
                E_632_m_n_1= E_632
                #E_fit_m_n_1= E_fit
                E_bs_m_n_1 = E_bs
                ss_bs_m_n_1= ss_bs.copy()
                ss_bs_matrix_m_n_1= ss_bs_matrix.copy()
                #ss_bs_count_matrix_m_n_1= ss_bs_count_matrix.copy()
        
        m+= 1
    
    
    return m_opt






def compute_min_angle_between_fiberset(V_set_orig):
    
    V_set= V_set_orig.copy()
    
    n_fib= V_set.shape[1]
    max_cos= 0
    
    for i in range(V_set.shape[1]):
        V_set[:,i]= V_set[:,i]/ np.linalg.norm(V_set[:,i])
        
    for i in range(n_fib-1):
        for j in range(i+1,n_fib):
            
            v1= V_set[:,i]
            v2= V_set[:,j]
            
            cur_cos= np.abs( np.dot(v1, v2) )
            
            if cur_cos>max_cos:
                
                max_cos= cur_cos
    
    min_ang= np.arccos(max_cos)*180/np.pi
    
    return min_ang






def polar_fibers_from_solution(R_fin, n_fib):
    
    f_fin= np.zeros( (3, n_fib) )
    
    for i in range(n_fib):
        
        f_fin[0,i]= np.sin(R_fin[i*5+3])*np.cos(R_fin[i*5+4]) 
        f_fin[1,i]= np.sin(R_fin[i*5+3])*np.sin(R_fin[i*5+4])
        f_fin[2,i]= np.cos(R_fin[i*5+3])
        f_fin[:,i] /= np.linalg.norm( f_fin[:,i] )
    
    return f_fin



















def compute_angular_feature_vector_cont_full_sphere(sig_orig, b_vecs, v, N= 10, M= 20, full_circ= False):
    
    assert( len(sig_orig)==b_vecs.shape[1] )
    
    s= sig_orig.copy()
    
    f_vec= np.zeros((v.shape[0], 2*N+2))
    
    if full_circ:
        theta= np.arccos( np.clip( np.dot( v, b_vecs ) , -1, 1) )
    else:
        theta= np.arccos( np.clip( np.abs( np.dot( v, b_vecs ) ), 0, 1) )
        
    for i in range(N+1):
        
        if full_circ:
            theta_i= i/N * np.pi
        else:
            theta_i= i/N * np.pi/2
            
        diff= 1/ ( np.abs(theta- theta_i) + 0.2 )
        arg1= np.argsort(diff)[:,-M:]
        
        arg2= [ np.arange(len(v))[:,np.newaxis],arg1 ]
        
        mean= np.sum( diff[arg2]*s[arg1] , axis=1)/ np.sum( diff[arg2], axis=1 )
        var = np.sum( diff[arg2]*((s[arg1]-mean[:,np.newaxis])**2) , axis=1 )/ np.sum( diff[arg2], axis=1 )
        
        f_vec[:, i]=     mean
        f_vec[:, i+N+1]= var
        
    return f_vec










def compute_angular_feature_vector_cont(sig_orig, b_vecs, dir_orig, N= 10, M= 20, full_circ= False):
    
    s= sig_orig.copy()
    d= dir_orig.copy()
    d/= np.linalg.norm(d)
    
    f_vec= np.zeros((N+1,2))
    
    if full_circ:
        theta= np.arccos( np.clip( np.dot( d, b_vecs ) , -1, 1) )
    else:
        theta= np.arccos( np.clip( np.abs( np.dot( d, b_vecs ) ), 0, 1) )
        
    for i in range(N+1):
        
        if full_circ:
            theta_i= i/N * np.pi
        else:
            theta_i= i/N * np.pi/2
            
        diff= 1/ ( np.abs(theta- theta_i) + 0.2 )
        
        if M>0:
            arg= np.argsort(diff)[-M:]
            mean= np.sum( diff[arg]*s[arg] )/ diff[arg].sum()
            var = np.sum( diff[arg]*((s[arg]-mean)**2) )/ diff[arg].sum()
        else:
            mean= np.sum( diff*s )/ diff.sum()
            var = np.sum( diff*((s-mean)**2) )/ diff.sum()
        
        f_vec[i,:]= mean, var
        
    return f_vec




def Cart_2_Spherical(xyz):
    
    xy = xyz[:,0]**2 + xyz[:,1]**2
    r = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2])
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    phi[phi<0]= 2*np.pi+phi[phi<0]
    
    return r, theta, phi








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
        
    








def compute_min_angle_between_vector_sets(V1_orig, V2_orig):
    
    V1= V1_orig.copy()
    V2= V2_orig.copy()
    
    assert(V1.shape[0]==3 and V2.shape[0]==3)
    
    for i in range(V1.shape[1]):
        V1[:,i]= V1[:,i]/ np.linalg.norm(V1[:,i])
    
    for i in range(V2.shape[1]):
        V2[:,i]= V2[:,i]/ np.linalg.norm(V2[:,i])
    
    max_cos= 0
    
    for i in range(V1.shape[1]):
        for j in range(V2.shape[1]):
            
            cur_cos= np.clip( np.abs( np.dot(V1[:,i], V2[:,j]) ), 0, 1)
            
            if cur_cos>max_cos:
                
                max_cos= cur_cos
    
    min_ang= np.arccos(max_cos)*180/np.pi
    
    return min_ang


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
                            
                            temp= compute_min_angle_between_vector_sets(v_pr.T, v_gt[i_fib:i_fib+1,:].T)
                            
                            error_current+= temp* v_gt_norm[i_fib]
                            
                            Error_matrix[ix,iy,iz, i_fib]= error_current
                        
                        WAAE[n_gt-1]+= error_current
                        WAAE_count[n_gt-1]+= 1
                        
                    
                    
    
    return Error_matrix, WAAE, WAAE_count, WAAE/WAAE_count








def find_dominant_fibers_dipy_way(sph, f_n, min_angle, n_fib, peak_thr=.25, optimize=False, Psi= None, opt_tol=1e-7):
    
    if optimize:
        
        v = sph.vertices
        
        '''values, indices = local_maxima(f_n, sph.edges)
        directions= v[indices,:]
        
        order = values.argsort()[::-1]
        values = values[order]
        directions = directions[order]
        
        directions, idx = remove_similar_vertices(directions, 25,
                                                  return_index=True)
        values = values[idx]
        directions= directions.T
        
        seeds = directions
        
        def SHORE_ODF_f(x):
            fx= np.dot(Psi, x)
            return -fx
        
        num_seeds = seeds.shape[1]
        theta = np.empty(num_seeds)
        phi = np.empty(num_seeds)
        values = np.empty(num_seeds)
        for i in range(num_seeds):
            peak = opt.fmin(SHORE_ODF_f, seeds[:,i], xtol=opt_tol, disp=False)
            theta[i], phi[i] = peak
            
        # Evaluate on new-found peaks
        small_sphere = Sphere(theta=theta, phi=phi)
        values = sphere_eval(small_sphere)
    
        # Sort in descending order
        order = values.argsort()[::-1]
        values = values[order]
        directions = small_sphere.vertices[order]
    
        # Remove directions that are too small
        n = search_descending(values, relative_peak_threshold)
        directions = directions[:n]
    
        # Remove peaks too close to each-other
        directions, idx = remove_similar_vertices(directions, min_separation_angle,
                                                  return_index=True)
        values = values[idx]'''
        
    else:
        
        v = sph.vertices
        
        values, indices = local_maxima(f_n, sph.edges)
        directions= v[indices,:]
        
        order = values.argsort()[::-1]
        values = values[order]
        directions = directions[order]
        
        directions, idx = remove_similar_vertices(directions, min_angle,
                                                  return_index=True)
        values = values[idx]
        directions= directions.T
        
    return directions, values








def hardi2013_cylinders_and_iso_simulate(true_fibers_orig, b, q, lam_1=0.0019, lam_2=0.0004, d_iso= 0.003):
    
    true_fibers= true_fibers_orig.copy()
    
    n_f= true_fibers.shape[1]
    
    if n_f>0:
        v= true_fibers[:,0]
        f1= np.linalg.norm(v)
        v/= f1
        y1 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f1= y1= 0
    
    if n_f>1:
        v= true_fibers[:,1]
        f2= np.linalg.norm(v)
        v/= f2
        y2 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f2= y2= 0
    
    if n_f>2:
        v= true_fibers[:,2]
        f3= np.linalg.norm(v)
        v/= f3
        y3 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f3= y3= 0
    
    if n_f>3:
        v= true_fibers[:,3]
        f4= np.linalg.norm(v)
        v/= f4
        y4 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f4= y4= 0
    
    if n_f>4:
        v= true_fibers[:,4]
        f5= np.linalg.norm(v)
        v/= f5
        y5 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f5= y5= 0
    
    y_iso= np.exp( - d_iso*b )
    f_iso= 1- f1- f2- f3- f4- f5
    
    y= f1*y1 + f2*y2 + f3*y3 + f4*y4 + f5*y5 + f_iso*y_iso 
    
    return y







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































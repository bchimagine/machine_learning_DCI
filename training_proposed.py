#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""




import numpy as np
import dci
import matplotlib.pyplot as plt
#from dipy.data import get_sphere
#from sklearn import svm
import tensorflow as tf
import os
from tqdm import tqdm





# these files specifing the gradient table can be downloaded from:
# http://hardi.epfl.ch/static/events/2013_ISBI/testing_data.html

b_vals= np.loadtxt( './hardi-scheme.bval' )
b_vecs= np.loadtxt( './hardi-scheme.bvec' ).T
b_vecs= b_vecs[b_vals>10,:]
b_vals= b_vals[b_vals>10]

b_vals_test= b_vals
b_vecs_test= b_vecs






####    Generate training and test data

# parameters

N= 15
M= 15

min_fiber_separation= 30

D_par_range= [0.0019, 0.00235]
D_per_range= [0.00045, 0.00060]

######

# one fascicle voxels

n_fib= 1
CSF_range= [0.10, 0.80]
snr= 15

n_fib_sim= 100000
n_sig_sim= 10

X_train_1= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_train_1= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    
    s= dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= dci.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
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
    
    true_fibers_polar, _, _= dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    
    s= dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= dci.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_test+= 1
        
        X_test_1[i_test,:]= f_cont.flatten('F')
        
        Y_test_1[i_test,:]= ang
            
############

# two-fascicle voxels

n_fib= 2
CSF_range= [0.0, 0.50]
Fr1_range= [0.15, 0.30]

n_fib_sim= 500000
n_sig_sim= 10

X_train_2= np.zeros( (n_fib_sim*n_sig_sim,2*N+2) )
Y_train_2= np.zeros( (n_fib_sim*n_sig_sim,1) )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
    
    s= dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= dci.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
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
    
    true_fibers_polar, _, _= dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
    
    s= dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= dci.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_test+= 1
        
        X_test_2[i_test,:]= f_cont.flatten('F')
        
        Y_test_2[i_test,:]= ang

############

# three-fascicle voxels

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
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers_polar, _, _= dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])
    
    s= dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= dci.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= dci.compute_angular_feature_vector_cont(s, b_vecs.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
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
    
    true_fibers_polar, _, _= dci.polar_fibers_and_iso_init(n_fib, lam_1=d_par, lam_2=d_per, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    
    true_fibers= dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])
    
    s= dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= dci.add_rician_noise(s, snr=snr)
    
    for i_sim in range(n_sig_sim):
        
        test_dir= np.random.rand(3)
            
        f_cont= dci.compute_angular_feature_vector_cont(s, b_vecs_test.T, test_dir, N= N, M=M, full_circ= False)
        
        ang= dci.compute_min_angle_between_vector_sets(true_fibers, test_dir[:,np.newaxis])
        
        i_test+= 1
        
        X_test_3[i_test,:]= f_cont.flatten('F')
        
        Y_test_3[i_test,:]= ang


X_train= np.concatenate( (X_train_1, X_train_2, X_train_3), axis=0 )
Y_train= np.concatenate( (Y_train_1, Y_train_2, Y_train_3), axis=0 )

X_test= np.concatenate( (X_test_1, X_test_2, X_test_3), axis=0 )
Y_test= np.concatenate( (Y_test_1, Y_test_2, Y_test_3), axis=0 )







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





###  MLP

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

cost= tf.reduce_mean( tf.pow(( Y - Y_p ), 2) )
cost2= tf.reduce_mean( tf.div( tf.pow(( Y - Y_p ), 2) , tf.pow(( Y + 5.0 ), 3) ) )

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

gpu_ind= 1

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)

saver = tf.train.Saver(max_to_keep=50)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



i_global = 0
best_test = 0
i_eval = -1

L_Rate = 1.0e-6


batch_size = 1000
n_epochs = 1000


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



saved_model_path =  '... /model_saved_hardi_5.ckpt'

#saver.save(sess, saved_model_path)

saver.restore(sess, saved_model_path)



###############################################################################

# Check model's accuracy

n_rows, n_cols = 2, 4
fig, ax = plt.subplots(figsize=(16, 10), nrows=n_rows, ncols=n_cols)

Y_test_prd= np.zeros(Y_test.shape)

for i_v in tqdm(range(n_test//batch_size), ascii=True):
                
    batch_x = X_test[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
    
    Y_test_prd[i_v * batch_size:(i_v + 1) * batch_size]= sess.run(Y_p, feed_dict={X: batch_x, p_keep_hidden: 1.0})

plt.subplot(n_rows, n_cols, 1), plt.plot(Y_test[::10], Y_test_prd[::10], '.'), plt.plot(np.arange(100), np.arange(100), '.r')
plt.subplot(n_rows, n_cols, 5), plt.hist(Y_test - Y_test_prd, bins=50);

print( np.sqrt( np.mean(  (  Y_test- Y_test_prd )**2  )) )

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






























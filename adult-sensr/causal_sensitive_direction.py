# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:11:35 2020

@author: maity
"""
import sys
sys.path.insert(1, 'C:/Users/maity/OneDrive/Documents/adult-sensr/adult-sensr/')
import tensorflow as tf
import numpy as np
from adult import preprocess_adult_data, get_sensitive_directions_and_projection_matrix, get_consistency, get_metrics
from train_clp_adult import train_fair_nn
from mle_sgd import proximal_gd_sigmoid

from sklearn.utils.random import sample_without_replacement
from sklearn import linear_model
## Sampling pairs
def generate_pairs(len1, len2, n_pairs=100):
    """
    vanilla sampler of random pairs (might sample same pair up to permutation)
    n_pairs > len1*len2 should be satisfied
    """
    idx = sample_without_replacement(len1*len2, n_pairs)
    return np.vstack(np.unravel_index(idx, (len1, len2)))

seed = 1
X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test, names_income, names_gender = preprocess_adult_data(seed = seed)


## Run linear regression X_gender_train on y_gender_train to get the sensitive direction for the protected attribute gender
sensitive_regression = linear_model.LinearRegression(fit_intercept = True)
sensitive_regression.fit(X_gender_train, y_gender_train)
intercept = np.reshape(sensitive_regression.intercept_, (-1,1))
sensitive_directions = np.concatenate((intercept, sensitive_regression.coef_), axis = 1)

tf.reset_default_graph()
fair_info = [y_gender_train, y_gender_test, names_income[0], names_gender[0], sensitive_directions, None]
weights, train_logits, test_logits, _  = train_fair_nn(X_train, y_train, tf_prefix='sensr', adv_epoch_full=40, l2_attack=0.0001,
                                          adv_epoch=50, ro=0.001, adv_step=10., plot=True, fair_info=fair_info, balance_batch=True, 
                                          X_test = X_test, X_test_counter=None, y_test = y_test, lamb_init=2., 
                                          n_units=[100], l2_reg=0, epoch=12000, batch_size=1000, lr=1e-4, lambda_clp=0.,
                                          fair_start=0., counter_init=False, seed=None)



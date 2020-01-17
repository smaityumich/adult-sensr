import tensorflow as tf
import numpy as np
from adult import preprocess_adult_data, get_sensitive_directions_and_projection_matrix, get_consistency, get_metrics
from train_clp_adult import train_fair_nn
from mle_sgd import proximal_gd_sigmoid

from sklearn.utils.random import sample_without_replacement
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

sensitive_directions, _ = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test)

tf.reset_default_graph()
fair_info = [y_gender_train, y_gender_test, names_income[0], names_gender[0], sensitive_directions, None]
weights, train_logits, test_logits, _  = train_fair_nn(X_train, y_train, tf_prefix='sensr', adv_epoch_full=40, l2_attack=0.0001,
                                          adv_epoch=50, ro=0.001, adv_step=10., plot=True, fair_info=fair_info, balance_batch=True, 
                                          X_test = X_test, X_test_counter=None, y_test = y_test, lamb_init=2., 
                                          n_units=[100], l2_reg=0, epoch=12000, batch_size=1000, lr=1e-4, lambda_clp=0.,
                                          fair_start=0., counter_init=False, seed=None)

## Metrics
preds = np.argmax(test_logits, axis = 1)
gender_race_consistency, spouse_consistency = get_consistency(X_test, weights = weights)
print('gender/race combined consistency', gender_race_consistency)
print('spouse consistency', spouse_consistency)
acc_temp, bal_acc_temp, race_gap_rms_temp, race_max_gap_temp, gender_gap_rms_temp, gender_max_gap_temp = get_metrics(dataset_orig_test, preds)


## EXPLORE metric
K = y_train.shape[1]
comparable_pairs = None
for i in range(K):
    c0_idx = np.where((y_gender_train[:,0] + y_train[:,i])==2)[0]
    c1_idx = np.where((y_gender_train[:,1] + y_train[:,i])==2)[0]
    pairs_idx = generate_pairs(len(c0_idx), len(c1_idx), n_pairs=10000)
    comparable_pairs_now = X_train[c0_idx[pairs_idx[0]]]- X_train[c1_idx[pairs_idx[1]]]
    if comparable_pairs is None:
        comparable_pairs = comparable_pairs_now
    else:
        comparable_pairs = np.vstack((comparable_pairs, comparable_pairs_now))


# incomparable pairs are simply sampled from different classes
c0_idx = np.where(y_train[:,0])[0]
c1_idx = np.where(y_train[:,1])[0]
pairs_idx = generate_pairs(len(c0_idx), len(c1_idx), n_pairs=5000)
incomp_pairs_new = [X_train[c0_idx[pairs_idx[0]]], X_train[c1_idx[pairs_idx[1]]]]
Xnew_incomp = incomp_pairs_new[0] - incomp_pairs_new[1]
       
X_pairs = np.vstack((comparable_pairs, Xnew_incomp))
Y_pairs = np.zeros(X_pairs.shape[0])
Y_pairs[:comparable_pairs.shape[0]] = 1

Sigma_fair_mle = proximal_gd_sigmoid(X_pairs,Y_pairs,mbs=1000,maxiter=10000)

tf.reset_default_graph()
fair_info = [y_gender_train, y_gender_test, names_income[0], names_gender[0], sensitive_directions, Sigma_fair_mle]
weights, train_logits, test_logits, _  = train_fair_nn(X_train, y_train, tf_prefix='explore', adv_epoch_full=10, l2_attack=0.1,
                                          adv_epoch=50, ro=0.01, adv_step=10., plot=True, fair_info=fair_info, balance_batch=True, 
                                          X_test = X_test, X_test_counter=None, y_test = y_test, lamb_init=2., 
                                          n_units=[100], l2_reg=0, epoch=30000, batch_size=1000, lr=1e-4, lambda_clp=0.,
                                          fair_start=0., counter_init=False, seed=None)

## Metrics
preds = np.argmax(test_logits, axis = 1)
gender_race_consistency, spouse_consistency = get_consistency(X_test, weights = weights)
print('gender/race combined consistency', gender_race_consistency)
print('spouse consistency', spouse_consistency)
acc_temp, bal_acc_temp, race_gap_rms_temp, race_max_gap_temp, gender_gap_rms_temp, gender_max_gap_temp = get_metrics(dataset_orig_test, preds)

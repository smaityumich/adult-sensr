import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
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

bios_datafolder = ''
bios_scrapped = np.load(bios_datafolder + 'X_bert_scrapped.npy')
bios_real = np.load(bios_datafolder + 'X_bert_real.npy')
bios_counter = np.load(bios_datafolder + 'X_bert_counter.npy')
y_title = list(map(str, np.load(bios_datafolder + 'bios_titles.npy')))
y_gender = list(map(str, np.load(bios_datafolder + 'bios_gedner.npy')))

## Subsample data
#classes_keep = ['attorney', 'paralegal']
seed = 10
np.random.seed(seed)
classes_keep = ['nurse', 'physician', 'attorney','paralegal']
index_select = np.array([i for i,p in enumerate(y_title) if p in classes_keep])
y_title_new = np.array(y_title)[index_select]
datanew = np.vstack((y_title_new, index_select)).transpose()
df = pd.DataFrame(dict(
        A=datanew[:,0],
        B=datanew[:,1]
    ))
new = df.groupby('A', group_keys=False).apply(lambda x: x.sample(min(len(x), 5000)))
new = new.values
index_select = np.array(new[:,1], dtype = int)

bios_real = bios_real[index_select]
bios_scrapped = bios_scrapped[index_select]
bios_counter = bios_counter[index_select]
y_title = new[:,0]
y_gender = np.array(y_gender)[index_select]

X_real_train, X_real_test, X_counter_train, X_counter_test, X_scrapped_train, X_scrapped_test, y_train, y_test, gender_train, gender_test = train_test_split(bios_real, bios_counter, bios_scrapped, y_title, y_gender, test_size=0.2, random_state=seed)


################## Running metric learning ##################
# comparable pairs are all the pairs with same title but different gender
    
    
def partition(array):
  return {i: (array == i).nonzero()[0] for i in np.unique(array)}

title_partition = partition(y_train)
gender_partition = partition(gender_train)
comparable_pairs = None

for iter in range(len(classes_keep)):
    c0_idx = np.intersect1d(gender_partition["F"], title_partition[classes_keep[iter]])
    c1_idx = np.intersect1d(gender_partition["M"], title_partition[classes_keep[iter]])
    pairs_idx = generate_pairs(len(c0_idx), len(c1_idx), n_pairs=4000)
    comparable_pairs_now = X_real_train[c0_idx[pairs_idx[0]]]- X_real_train[c1_idx[pairs_idx[1]]]
    if comparable_pairs is None:
        comparable_pairs = comparable_pairs_now
    else:
        comparable_pairs = np.vstack((comparable_pairs, comparable_pairs_now))


# incomparable pairs are simply sampled from different classes
Xnew_incomp = None


for iter in range(40):
   indices = np.random.choice(np.arange(len(classes_keep)), 2, replace=False)
   c0_idx = np.where(y_train==classes_keep[indices[0]])[0]
   c1_idx = np.where(y_train==classes_keep[indices[1]])[0]
   pairs_idx = generate_pairs(len(c0_idx), len(c1_idx), n_pairs=400)
   incomp_pairs_new = [X_real_train[c0_idx[pairs_idx[0]]], X_real_train[c1_idx[pairs_idx[1]]]]
   Xnew1 = incomp_pairs_new[0] - incomp_pairs_new[1]
   if Xnew_incomp is None:
       Xnew_incomp = Xnew1
   else:
       Xnew_incomp = np.vstack((Xnew1, Xnew_incomp))
       
       

X_pairs = np.vstack((comparable_pairs, Xnew_incomp))
Y_pairs = np.zeros(X_pairs.shape[0])
Y_pairs[:comparable_pairs.shape[0]] = 1

Sigma_fair_mle = proximal_gd_sigmoid(X_pairs,Y_pairs,mbs=5000,maxiter=5000)


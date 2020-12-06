# import package
from MDAnalysis.lib.distances import distance_array
from tensorflow.keras.utils import HDF5Matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from ivis import Ivis
from collections import defaultdict

import MDAnalysis as mda
import numpy as np
import mdtraj as md

import itertools
import pickle as pk
import h5py


class LinearExplainer(object):
"""
The ivis component 1 & 2 fit the important feature best
"""
    def __init__(self, model, data=None):
        self.model = model
        self.data = data

    def feature_importances_(self, X, **kwargs):
        if self.data is None:
            embeddings = self.model.transform(X)
        else:
            embeddings = self.data

        score = np.empty(shape=(X.shape[1],))
        for i in range(X.shape[1]):
            est = LinearRegression(**kwargs).fit(embeddings, X[:, i])
            score[i] = est.score(embeddings, X[:, i])

        return score


def BuildAngles(cases):
"""
Just Add the left middle points
"""
    case_ls = [case for case in cases]
    case_combinations = list(itertools.combinations(cases, 2))
    angle_ls = []
    
    for i in case_combinations:
        ex_ls = []
        ex_ls.append(i[0])
        ex_ls.append(i[1])
        left_ls = [item for item in case_ls if not item in ex_ls]
        for j in left_ls:
            i_ls = list(i)
            i_ls.insert(1, j)
            i_tuple = tuple(i_ls)
            angle_ls.append(i_tuple)
    return angle_ls
 
 
 def BuildDihedrals(cases):
 """
 Three ponits difine a plane
 Two combinations is Ok
 """
    dihedral_ls = []
    
    planes = list(itertools.combinations(cases, 3))
    PlaneVSPlanes = itertools.combinations(planes, 2)
    for plane in PlaneVSPlanes:
        dihedral = tuple(set(i[0] + i[1]))
        dihedral_ls.append(dihedral)
    return dihedral_ls
    
    
# System

# specify trajectory and parameters of protein files (Amber Examples)
traj = "protein.nc"
parm = "protein.prmtop"

# load Trajectory as t
t = md.load(traj, top = parm )

# record the index of each CA
# so we can only recognize the amino acid residues, not ions
CAids = t.top.select("name CA")
aa_len = len(CAids)

# record the index of each amino acid
reslist = [a for a in t.topology.residues]
aalist = reslist[:aa_len]

# Distance

# build distance pairs
ca_dist_pairs = list(itertools.combinations(CAids, 2))
aa_dist_pairs = list(itertools.combinations(aalist, 2))

# compute CA distances
dist = md.compute_distances(t, ca_dist_pairs)
distances = np.where(dist <= 1.0, dist, 0) # Inspired by the activation function (1 nm)
D_scaled = MinMaxScaler().fit_transform(distances) # Necessary

# hdf5 file is needed for handling large datasets
hf = h5py.File('distances.h5', 'w')
hf.create_dataset('distances', data=D_scaled)

# Angle

# build angle triplets
ca_ang_pairs = BuildAngles(CAids)
aa_ang_pairs =  BuildAngles(aalist)

# compute angle triplets
ang = md.compute_angles(t, ca_ang_pairs)
A_scaled = MinMaxScaler().fit_transform(ang)

# always too large 
# (167 residues & 30,000 frames need about 500 GiB)
hf = h5py.File('angles.h5', 'w')
hf.create_dataset('angles', data=A_scaled)

# Dihedral

# build dihedral qartets
ca_dih_pairs = BuildDihedrals(CAids)
aa_dih_pairs = BuildDihedrals(aalist)

# compute dihedral qartets
dihs = md.compute_dihedrals(t, ca_dih_pairs)
D_scaled = MinMaxScaler().fit_transform(dihs)

# always too large 
# (167 residues & 30,000 frames need about 81.9 TiB)
hf = h5py.File('dihedrals.h5', 'w')
hf.create_dataset('dihedrals', data=D_scaled)

# compute Phi
phis = md.compute_phi(t)
H_scaled = MinMaxScaler().fit_transform(phis[1])

hf = h5py.File('phis.h5', 'w')
hf.create_dataset('phis', data=H_scaled)

# compute Psi
psis = md.compute_psi(t)
S_scaled = MinMaxScaler().fit_transform(psis[1])

hf = h5py.File('psis.h5', 'w')
hf.create_dataset('psis', data=S_scaled)

# load data
X = HDF5Matrix('distances.h5', 'distances')

# define ivis model
metrics_ivis = {
    "n_epochs_without_progress" : 10,
    "supervision_weight": 0.1,
    "embedding_dims": 2,
    "distance": "euclidean",
    "margin": 1,
    "model":"maaten",
    "epochs":200,
    "batch_size": 128,
    "supervision_metric": "sparse_categorical_crossentropy",
    "k" : 100
}

# ivis training 
model = Ivis(**metrics_ivis)
model.fit(X, shuffle_mode='batch')
model.save_model('iris.ivis')

# original feature importances
explainer = LinearExplainer(model)
X_value = np.array(X)
feature_importances = explainer.feature_importances_(X_value)

# some features are activated to 0, which fit definitely excellent
feature_importances_zero = []
for i in list(feature_importances):
    if i == 1:
        feature_importances_zero.append(float(0)) 
    else:
        feature_importances_zero.append(i)

# aa_pairs_importance & each_aa_importance
each_aa_importance_list = defaultdict(list)
each_aa_importance = {}

aa_pairs_importance = dict(zip(aa_dist_pairs, feature_importances_zero))
for aa_pairs, pairs_importance in aa_pairs_importance.items():
    each_aa_importance_list[aa_pairs[0]].append(float(pairs_importance))
    each_aa_importance_list[aa_pairs[1]].append(float(pairs_importance))
for aa1, importance1 in each_aa_importance_list.items():
    each_aa_importance[aa1] = sum(importance1)
    
sorted(aa_pairs_importance.items(), key = lambda kv:kv[1], reverse=True)   

# each_aa_importance in a percentage form
Total = sum(each_aa_importance.values())
each_aa_importance_percentage = {}

for key, value in each_aa_importance.items():
    percentage = value / Total * 100
    each_aa_importance_percentage[key] = percentage

sorted(each_aa_importance_percentage.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

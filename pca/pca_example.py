##
import numpy as np
from sklearn.decomposition import PCA
##

# generate data
data = np.zeros([1000, 200, 100])
data[:, 10, 10] = np.random.choice([0,1], size=len(data))
data[:, 11, 11] = np.random.choice([0,0,0,0,1], size=len(data))

# reshape data
flat = data.reshape([len(data), -1])
flat = flat - flat.mean(axis=0)

# run pca
pca = PCA(n_components=5)
pca.fit(flat)

# inspect results
var = pca.explained_variance_ratio_
var[var<1e-10] = 0
print(var)
##

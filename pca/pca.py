##
from soup import *
from sklearn.decomposition import PCA, FastICA
from scipy.ndimage import zoom

## load some data
mov = np.load('example.npy')

## downsample the data in space
mov = zoom(mov, (1, .5, .5), order=1)
mov_orig = mov.copy()

## shape the data to (n_samples x n_features)
# here, samples are time points, and features are pixels
mov = mov.reshape([len(mov), -1])

## mean-subtract
mov = mov - mov.mean(axis=0)

##
n_components = 50
pca = PCA(n_components)

##
pca.fit(mov)

##
pca.explained_variance_
pca.explained_variance_ratio_

##
comps = pca.components_

pl.imshow(comps[5].reshape([256,256]))

## reduced movie
# in an ideal world, where PCA works perfectly for segmentation, these are your time traces for cellular activity
reduced = comps @ mov.T
reduced = comps.dot(mov.T)
# this step is achieved in one using:
reduced = pca.fit_transform(mov).T

## go backwards: make a movie using *only* the reduced movie and the components
eigen_mov = (reduced.T @ comps).reshape(mov_orig.shape)

## ica
t = np.arange(0,10,0.01)
x1 = 1*np.sin(t)
x2 = 2*np.sin(t+1)
x3 = 3*np.sin(t+2)
x = np.array([x1,x2,x3]).T
x = (x - x.mean(axis=0)) / x.std(axis=0)
ica_in = x * np.roll(x, -1, axis=1)
ica = FastICA(max_iter=500, tol=0.001)
result = ica.fit_transform(ica_in)

## k-means


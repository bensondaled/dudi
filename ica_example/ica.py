##
from soup import *
from scipy.io import wavfile
from sklearn.decomposition import FastICA

##
fs,a1 = wavfile.read('strings.wav')
fs,a2 = wavfile.read('cello.wav')
fs,a3 = wavfile.read('strings3.wav')

n = 150000
a1 = a1[:n,0]
a2 = a2[:n,0]
a3 = a3[:n,0]
orig = np.array([a1,a2,a3]).T.astype(float)
X = orig.copy()
X = X - X.mean(axis=0)
X = X / X.std(axis=0)

# mix
M = np.array([  [.3, .2, .3], 
                [.3, .4, .1], 
                [.4, .4, .6]])
X = np.dot(X, M.T)

ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)

## play originals
play(a1)
play(a2)
play(a3)

## play mixed
play(X[:,0])
play(X[:,1])
play(X[:,2])

## play unmixed
play(100*S_[:,0])
play(100*S_[:,1])
play(100*S_[:,2])

## plot them
fig,axs = pl.subplots(3,1,sharex=True)
for ax,signal,name in zip(axs,(orig,X,S_), ['original','mixed','unmixed']):
    pf.Series(norm(signal,axis=0)).astype(float).plot(ax=ax)
    ax.set_title(name)
    pretty(ax=ax)

##

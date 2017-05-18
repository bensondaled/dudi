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

# shuffle the signals independently of one another
shuffler = [np.random.choice(np.arange(len(sig)), replace=False, size=len(sig)) for sig in X.T]
X_shuffled = np.array([sig[shuf] for sig,shuf in zip(X.T, shuffler)]).T

# mix
M = np.array([  [.3, .2, .3], 
                [.3, .4, .1], 
                [.4, .4, .6]])
X = X @ M.T
X_shuffled = X_shuffled @ M.T
# if you were to shuffle at this point, ICA would fail

# ICA model on data
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)
mixing = ica.mixing_
comps = ica.components_

# ICA model on shuffled data
ica_shuf = FastICA(n_components=3)
S_shuf = ica_shuf.fit_transform(X_shuffled)
mixing_shuf = ica_shuf.mixing_
comps_shuf = ica_shuf.components_

# check result
recovered = X @ comps.T # equivalent to transform, but you can sub out comps
assert np.allclose(S_, recovered)
recovered_shuf = X @ comps_shuf.T # use components discovered from shuffled data to transform original unshuffled data

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
signals = [orig, X, X_shuffled, recovered, recovered_shuf]
names = ['Original', 'Mixed', 'Shuffled-Mixed', 'Recovered using ICA components', 'Recovered using Shuffle-ICA components']
fig,axs = pl.subplots(len(signals),1,sharex=True)
for ax,signal,name in zip(axs, signals, names):
    pf.Series(norm(signal,axis=0)).astype(float).plot(ax=ax)
    ax.set_title(name)
    pretty(ax=ax)

pl.tight_layout()


##

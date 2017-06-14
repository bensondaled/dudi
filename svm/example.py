from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import zoom
##
digits = datasets.load_digits()

imgs = digits.images
labs = digits.target

#imgs = zoom(imgs, (1,.5,.5))

ndim = imgs.shape[1]

##
which = (labs==1) | (labs==8)
imgs = imgs[which]
labs = labs[which]

##
imgs = imgs.reshape([len(imgs), -1]).astype(int)

##
lr = LogisticRegression()
lr.fit(imgs[:250], labs[:250])

print( np.mean( lr.predict(imgs[250:]) == labs[250:] ) )

coef_orig = lr.coef_.copy().squeeze()
args = np.argsort(np.abs(coef_orig))[::-1]
##
results = []
for i in range(1,len(args)):
    lr = LogisticRegression()
    lr.fit(imgs[:, :i], labs)
    pred = lr.predict(imgs[:, :i])
    score = np.mean(pred==labs)
    results.append(score)

##

fig,axs = pl.subplots(2,2); axs=axs.ravel()

cmap = pl.cm.Greys_r

axs[0].imshow(imgs[labs==1].mean(axis=0).reshape([8,8]), cmap=cmap)
axs[0].set_title('Average 1')
axs[1].imshow(imgs[labs==8].mean(axis=0).reshape([8,8]), cmap=cmap)
axs[1].set_title('Average 8')
axs[2].imshow(lr.coef_.reshape([8,8]), cmap=cmap)
axs[2].set_title('Weights')
axs[3].imshow((imgs[labs==1].mean(axis=0)-imgs[labs==8].mean(axis=0)).reshape([8,8]), cmap=cmap)
axs[3].set_title('Average 1-8')

for ax in axs:
    ax.axis('off')

##

all_combs = []
all_results = []
for npix in range(1,ndim**2):
    print(npix)
    combs = np.array(list(it.combinations(np.arange(ndim**2), npix)))
    all_combs.append(combs)
    results = []
    for c in combs:
        imgs_i = imgs[:,c]
        lr = LogisticRegression()
        lr.fit(imgs_i, labs)
        pred = lr.predict(imgs_i)
        score = np.mean(pred==labs)
        results.append(score)
    all_results.append([combs[np.argmax(results)], np.max(results), np.argmax(results)])
##

##

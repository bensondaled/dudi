##
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

##

n_clusters = 5
n_members = [100 for i in range(n_clusters)]
dim = 1000
std = 30

space = np.zeros([dim,dim])
centers = [np.random.randint(0,dim,size=2) for i in range(n_clusters)]

pts = np.empty([np.sum(n_members), 3]) # y, x, cluster_id
i = 0
for clust_id,(c,n) in enumerate(zip(centers, n_members)):
    clust_id = np.array([clust_id]*n)
    cy = np.random.normal(c[0], std, size=n)
    cx = np.random.normal(c[1],std, size=n)
    pts[i:i+n] = np.array([cy, cx, clust_id]).T
    i += n


##

km = KMeans(n_clusters=5)
km.fit(pts[:,:-1])

centers_ = km.cluster_centers_
labs_ = km.labels_


gm = GaussianMixture(n_components=5)
gm.fit(pts[:,:-1])

means_ = gm.means_
glabs = gm.predict(pts[:,:-1])

##

fig,axs = pl.subplots(1,3)

axs[0].scatter(pts.T[1], pts.T[0], c=pts.T[2])
axs[0].set_title('True')

axs[1].scatter(pts.T[1], pts.T[0], c=labs_)
axs[1].scatter(centers_.T[1], centers_.T[0], marker='x', color='k', lw=4, s=100)
axs[1].set_title('K-means')

axs[2].scatter(pts.T[1], pts.T[0], c=glabs)
axs[2].scatter(means_.T[1], means_.T[0], marker='x', color='k', lw=4, s=100)
axs[2].set_title('GMM')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

##

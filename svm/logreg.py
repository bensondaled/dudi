from sklearn.linear_model import LogisticRegression

##

n = 5000
x_pos = np.random.choice(np.arange(1000), size=[n,2])
x_neg = np.random.choice(np.arange(-1000,0), size=[n,2])
x = np.concatenate([x_pos, x_neg])

np.random.shuffle(x)

##

labels = (x[:,0] > 0).astype(int)
np.random.shuffle(labels)

##

lr = LogisticRegression()
lr.fit(x[:5000], labels[:5000])

true = labels[5000:]
guess = lr.predict(x[5000:])

##

coef = lr.coef_.squeeze()

np.sum(x*coef, axis=1)

##

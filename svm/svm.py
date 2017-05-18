##
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

##
data = pd.read_excel('spine_data.xlsx')
morphs = data['morphology'].str.strip(' ').values
morph_ids, morphs = np.unique(morphs, return_inverse=True)
data = data[[c for c in data.columns if 'spine' in c and 'ID' not in c and 'position' not in c and 'branchings' not in c]].values

## preprocessing
data = scale(data)

## SVM
fig,axs = pl.subplots(1,2)
for C in np.arange(.001,.5,.01):
    s = svm.SVC(C, kernel='linear')
    s.fit(data, morphs)
    accuracy = (s.predict(data) == morphs).mean()
    axs[0].plot(C, accuracy, 'ko')

    # get the separating hyperplane
    w = s.coef_[0]
    a = -w[0] / w[1]
    xx = 0
    yy = a * xx - (s.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = s.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = s.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    d = dist(yy_up, yy_down)
    axs[1].plot(C, d, 'bo')

for cv in range(2,30,2):
    scores = cross_val_score(s, data, morphs, cv=cv)
    pl.plot(cv, np.mean(scores), 'ko')

print(scores)

s = svm.SVC(C, kernel='linear', probability=True)
s.fit(data, morphs)


##

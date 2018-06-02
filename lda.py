from sklearn.decomposition import LatentDirichletAllocation, PCA
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


d = loadmat('bbc_train_data.mat')

data = d['data']

# labels = []
#
# with open('bbcsport.classes', 'r') as fp:
#     l = fp.readline()
#     while l != "":
#         try:
#             labels.append(int(l.split(' ')[1][0]))
#             l = fp.readline()
#         except:
#             l = fp.readline()
#
# labels.pop(0)
#
# labels = np.array(labels)
labels = np.squeeze(d['labels'].T)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels,
                                                    random_state=42,
                                                    test_size=0.3)

n_components = 64

lda = LatentDirichletAllocation(n_components=n_components, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0,
                                batch_size=25)

lda.fit(X_train)



lda_txed = lda.transform(X_train)


rf = RandomForestClassifier()

rf.fit(lda_txed, Y_train)
print(rf.score(lda_txed, Y_train))

lda_test_txed = lda.transform(X_test)
print(rf.score(lda_test_txed, Y_test))

pca = PCA(n_components=4)
PCA_train = pca.fit_transform(lda_txed)

names = ['athletics', 'cricket', 'football', 'rugby', 'tennis']
#for name, label in zip(names, range(5)):
markers = ['o', '*', '^', 'v', '<']

plts = [(0,0),(0,1), (1,0), (1,1)]

fig, axarr = plt.subplots(2,2)
for i, color in enumerate(['red', 'blue', 'green', 'yellow', 'cyan']):
    idx = np.where(Y_train==i)

    axarr[0,0].scatter(PCA_train[idx,0], PCA_train[idx,1], c=color,
                    label=names[i],
               alpha=0.3, edgecolors='k', s=50, marker=markers[i])

    axarr[0, 1].scatter(PCA_train[idx, 0], PCA_train[idx, 2], c=color,
                        label=names[i],
                        alpha=0.3, edgecolors='k', s=50, marker=markers[i])

    axarr[1, 0].scatter(PCA_train[idx, 1], PCA_train[idx, 2], c=color,
                        label=names[i],
                        alpha=0.3, edgecolors='k', s=50, marker=markers[i])

    axarr[1, 1].scatter(PCA_train[idx, 1], PCA_train[idx, 3], c=color,
                        label=names[i],
                        alpha=0.3, edgecolors='k', s=50, marker=markers[i])

axarr[0,0].legend()
axarr[0,1].legend()
axarr[1,0].legend()
axarr[1,1].legend()

axarr[0,0].set_xlabel('PCA1')
axarr[0,0].set_ylabel('PCA2')

axarr[0,1].set_xlabel('PCA1')
axarr[0,1].set_ylabel('PCA3')

axarr[1,0].set_xlabel('PCA2')
axarr[1,0].set_ylabel('PCA3')

axarr[1,1].set_xlabel('PCA2')
axarr[1,1].set_ylabel('PCA4')

plt.show()

# terms = []
#
# with open('bbcsport.terms', 'r') as fp:
#     t = fp.readline()
#     while t != "":
#         terms.append(t)
#         t = fp.readline()
#
# print_top_words(lda, terms ,n_top_words=10)
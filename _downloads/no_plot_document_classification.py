# -*- coding: utf-8 -*-
from operator import itemgetter

import numpy as np
import pylab as pl

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import Vectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics



###############################################################################
# Load some categories from the training set
categories = ['comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc',
              'sci.electronics']

data_train = fetch_20newsgroups(subset='train', categories=categories,
                               shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories,
                              shuffle=True, random_state=42)

categories = data_train.target_names
print "%d documents (training set)" % len(data_train.data)
print "%d documents (testing set)" % len(data_test.data)
print "%d categories" % len(categories)

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

# Extracting features from the training dataset using a sparse vectorizer
vectorizer = Vectorizer(use_idf=True)
X_train = vectorizer.fit_transform(data_train.data)

# Extracting features from the test dataset using the same vectorizer
X_test = vectorizer.transform(data_test.data)

vocabulary = np.array([t for t, i in sorted(vectorizer.vocabulary.iteritems(),
                                            key=itemgetter(1))])
def bag(X, features):
    X_red = list()
    for bag_of_words in features:
        X_red.append(sum(X[:, 
                            np.where(vocabulary == word)[0]].toarray()[:, 0]
                        for word in bag_of_words))
    return np.array(X_red).T

features = np.array([
            ['amp', 'bus', 'circuit', 'circuits', 'controller',
            'detector', 'ground', 'power', 'voltage', 'cooling',
            'motorola', 'amplifier', 'seagate', 'toshiba', 's3',
            'antenna', 'led', 'batteries', 'battery', 'analog',
            'jumpers', 'traffic',
            ],
            ['dma', 'drive', 'driver', 'drives', 'file', 'files', 'tape',
            'monitor', 'monitors', 'directory', 'floppy', 'font',
            'fonts', 'truetype', 'ide', 'ini', 'motherboard', 'nt',
            'win', 'windows', 'cursor', 'desktop', 'icon', 'icons']
        ], dtype=object)

X_train = bag(X_train, features)
X_test  = bag(X_test, features)


###############################################################################
# Benchmark classifier

knn10 = KNeighborsClassifier(n_neighbors=10)
knn10.fit(X_train, y_train)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)

svm = SVC(C=1000, kernel='linear')
svm.fit(X_train, y_train)

svm_rbf = SVC(C=200, gamma=2)
svm_rbf.fit(X_train, y_train)

pred = knn10.predict(X_test)

score = metrics.f1_score(y_test, pred)
print "f1-score:   %0.3f" % score

print "classification report:"
print metrics.classification_report(y_test, pred,
                                        target_names=categories)

print "confusion matrix:"
print metrics.confusion_matrix(y_test, pred)

###############################################################################
# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].

x_min, x_max = X_train[:,0].min(), .85*X_train[:,0].max() + .02
y_min, y_max = X_train[:,1].min(), .85*X_train[:,1].max() + .02
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z10 = knn10.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z10 = Z10.reshape(xx.shape)

Z1 = knn1.predict(np.c_[xx.ravel(), yy.ravel()])
Z1 = Z1.reshape(xx.shape)

Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)

Z_rbf = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = Z_rbf.reshape(xx.shape)

pl.figure(1, figsize=(4.6, 5.06))
for name, X, y in [('train', X_train, y_train),
                   ('test', X_test, y_test),
                  ]:
    pl.clf()
    pl.axes([.08, .07, .91, .827])
    pl.set_cmap(pl.cm.Spectral)

    # Plot also the training points
    symbols = 'o^s'
    for label, symbol in zip(np.unique(y), symbols):
        this_pts = X[y==label]
        pl.plot(this_pts[:, 0],
                this_pts[:, 1], symbol,
                color=pl.cm.Spectral(.5*label),
                label=categories[label],
                markeredgewidth=1)

    pl.xlim(xx.min()-.01, xx.max())
    pl.ylim(yy.min()-.01, yy.max())
    pl.xlabel(u"Mots relatifs à l'électronique",
            weight='bold', verticalalignment='center', size=13)
    pl.ylabel(u"Mots relatifs à l'informatique",
            weight='bold', horizontalalignment='center', size=13)
    pl.xticks((0.,))
    pl.yticks((0.,))
    pl.legend(numpoints=1, handlelength=0)

    if name == 'train':
        pl.title("Messages dans la base d'apprentissage", size=15)
        title = "base d'apprentissage"
    else:
        pl.title(u"Messages à prédire", size=15)
        title = u"messages à prédire"
    pl.savefig('document_word_distrib_%s.png' % name, dpi=200)
    mesh = pl.pcolormesh(xx, yy, Z1)
    pl.xlim(xx.min()-.01, xx.max())
    pl.ylim(yy.min()-.01, yy.max())
    pl.title(u"Prédiction au plus proche voisin\n et %s" % title, size=15)
    pl.savefig('document_word_knn1_%s.png' % name, dpi=200)


pl.figure(2, figsize=(3.62, 4.048))
pl.clf()
pl.axes([.01, .01, .98, .897])
for label, symbol in zip(np.unique(y_test), symbols):
    this_pts = X_test[y_test==label]
    pl.plot(this_pts[:, 0],
            this_pts[:, 1], symbol,
            color=pl.cm.Spectral(.5*label),
            label=categories[label],
            markeredgewidth=1)

mesh = pl.pcolormesh(xx, yy, Z10)
pl.xlim(xx.min()-.01, xx.max())
pl.ylim(yy.min()-.01, yy.max())
pl.xticks(())
pl.yticks(())
pl.title(u"10 plus proches voisins", size=16)
pl.savefig('document_word_knn10_test.png', dpi=200)

mesh.remove()
mesh = pl.pcolormesh(xx, yy, Z_svm)
pl.xlim(xx.min()-.01, xx.max())
pl.ylim(yy.min()-.01, yy.max())
pl.title(u"Décisions linéaires", size=15)
pl.savefig('document_word_svm_test.png', dpi=200)

mesh.remove()
mesh = pl.pcolormesh(xx, yy, Z_rbf)
pl.xlim(xx.min()-.01, xx.max())
pl.ylim(yy.min()-.01, yy.max())
pl.title(u"Décisions courbes", size=15)
pl.savefig('document_word_svm_rbf_%s.png' % name, dpi=200)
pl.show()
pl.draw()

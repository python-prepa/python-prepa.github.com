# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import Vectorizer
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
vectorizer = Vectorizer()
X_train = vectorizer.fit_transform(data_train.data)

# Extracting features from the test dataset using the same vectorizer
X_test = vectorizer.transform(data_test.data)

###############################################################################
# Benchmark classifier

# sklearn.naive_bayes.MultinomialNB(alpha=310) gives 84.3%
# LogisticRegression(C=1.9) gives 85.9%
#clf = LogisticRegression(C=1.9, penalty='l2')
# LinearSVC(C=.17) gives 85.9%

from sklearn.svm.sparse import LinearSVC
clf = LinearSVC(C=.17)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
for category, words in zip(categories,
                            vectorizer.inverse_transform(
                                    clf.coef_ > .7)):
    print category, words.ravel()

score = metrics.f1_score(y_test, pred)
print "f1-score:   %0.3f" % score

print "classification report:"
print metrics.classification_report(y_test, pred,
                                        target_names=categories)

print "confusion matrix:"
print metrics.confusion_matrix(y_test, pred)


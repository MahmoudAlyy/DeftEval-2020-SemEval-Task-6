import pickle
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import WordEmbeddingHelperFunctions as help
#"Linear SVM", "RBF SVM",
names = ["MLP", "Logistic Regression", "SGD", "Nearest Neighbors",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

# SVC(kernel="linear", C=0.025),
# SVC(gamma=2, C=1),
classifiers = [

    MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
                  solver='lbfgs'),
    LogisticRegression(
        random_state=1, multi_class='multinomial', solver='saga'),
    SGDClassifier(loss='hinge',
                  penalty='l2',
                  verbose=1,
                  random_state=1,
                  learning_rate='invscaling',
                  eta0=1),
    DecisionTreeClassifier(max_depth=5),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
"""
# (Optional) Classification via stochastic gradient descent classifier.
sgd = SGDClassifier(loss='hinge',
                    penalty='l2',
                    verbose=1,
                    random_state=1,
                    learning_rate='invscaling',
                    eta0=1)


# Classification via Logistic Model
logistic = LogisticRegression(
    random_state=1, multi_class='multinomial', solver='saga')
"""


def sk_evaluate(model, feature, label, label_names):
    pred = model.predict(feature)
    true = np.array(label)

    print('Score on dataset...\n')
    print('Confusion Matrix:\n', confusion_matrix(true, pred))
    print('\nClassification Report:\n', classification_report(
        true, pred, target_names=label_names))
    print('\naccuracy: {:.3f}'.format(accuracy_score(true, pred)))
    print('f1 score: {:.3f}'.format(f1_score(true, pred, average='weighted')))

    return pred, true


# Get train data
save_dir = "data/task1/combined/train_data_w2v.pkl"
strings = pickle.load(open(save_dir, "rb"))
with open("data/task1/combined/train_data_w2v.txt") as f:
    sentences = f.readlines()
newArr = []
for sen in sentences:
    # print(sen)
    newSen = sen.replace("\t0\n", "").replace("\t1\n", "")
    # print(newSen)
    newSen = newSen.split(" ")
    newArr.append(newSen)
# print(newArr)


save_dir = "data/task1/combined/train_data_w2v.pkl"
strings = pickle.load(open(save_dir, "rb"))
dfWithoutEmbedding = pd.DataFrame(strings, columns=['sentence', 'value'])


test_dir = "data/task1/combined/dev_data_w2v.pkl"
testStrings = pickle.load(open(test_dir, "rb"))
testData = pd.DataFrame(testStrings, columns=['sentence', 'value'])
test_sentences = testData.sentence
#print(test_sentences.split(" "))
test_y = testData.value
with open("data/task1/combined/dev_data_w2v.txt") as f:
    testsentences = f.readlines()
newArr2 = []
for sen in testsentences:
    # print(sen)
    newSen = sen.replace("\t0\n", "").replace("\t1\n", "")
    # print(newSen)
    newSen = newSen.split(" ")
    newArr2.append(newSen)
# print(newArr2)
# print("print 5 lines from train data:\n")
# for sentence, val in strings:
#     # sentence : string of words
#     # vale : 1 -> has def , 0 -> no def
#     # ur works start from here gg,hf,gl
#     #print(type(sentence))
#     #print(sentence,val)
#     loop_counter = loop_counter + 1
#     #if loop_counter > 5:
#     #    break

# print(loop_counter)
# loop_counter = 0
# print("#######################################")
# print("print 5 lines from test data:\n")


def prepareTrainAndtest(df, yValues):
    # Specify train/valid/test size.
    # Prepare test dataset.
    train_size = math.floor(len(df) * 0.7)
    test_size = len(df) - train_size
    train_X, test_X, train_y, test_y = train_test_split(df, yValues,
                                                        test_size=test_size,
                                                        random_state=1,)

    print('Shape of train_X: {}'.format(train_X.shape))
    print('Shape of test_X: {}'.format(test_X.shape))
    return train_X, test_X, train_y, test_y


# train models
word_model = Word2Vec(newArr, min_count=2,
                      window=5,
                      size=300,
                      workers=4,
                      iter=200)
mean_vec_tr = help.MeanEmbeddingVectorizer(word_model)
doc_vec = mean_vec_tr.transform(newArr)

testWords_model = Word2Vec(newArr2, min_count=2,
                           window=5,
                           size=300,
                           workers=4,
                           iter=200)
mean_vec_tr2 = help.MeanEmbeddingVectorizer(testWords_model)
testDoc_vec = mean_vec_tr2.transform(newArr2)
# Save word averaging doc2vec.
print('Shape of word-mean doc2vec...')
# display(doc_vec.shape)

print('Save word-mean doc2vec as csv file...')
np.savetxt('doc_vec.csv', doc_vec, delimiter=',')
doc_vec = pd.read_csv('doc_vec.csv', header=None)

train_X = doc_vec
test_X = testDoc_vec
train_y = dfWithoutEmbedding.value
print('Shape of train_X: {}'.format(train_X.shape))
print(train_X[0:5])
print('Shape of test_X: {}'.format(test_X.shape))
print(train_y[0:5])
#train_X, test_X, train_y, test_y = prepareTrainAndtest(doc_vec, dfWithoutEmbedding.value)
# summarize the loaded model
print(word_model)
# summarize vocabulary
words = list(word_model.wv.vocab)
# print(words)
# access vector for one word
# print(word_model['biology'])
# save model
word_model.save('word_model.bin')
# load model
new_model = Word2Vec.load('word_model.bin')
print(new_model)
# fit a 2d PCA model to the vectors
X = word_model[word_model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
"""
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
"""
"""
#logistic
print("logistic")
logistic.fit(train_X, train_y)
# Score on test dataset.
print('Performance of Mean Word Vector on testing dataset...')
_, _ = sk_evaluate(logistic, test_X, test_y, label_names=None)
#SGD
print("SGD")
sgd.fit(train_X, train_y)
# Score on test dataset.
print('Performance of Mean Word Vector on testing dataset...')
_, _ = sk_evaluate(sgd, test_X, test_y, label_names=None)
"""
"""
for name, clf in zip(names, classifiers):
    print(name)
    clf.fit(train_X, train_y)
    _, _ = sk_evaluate(clf, test_X, test_y, label_names=None)
"""

# tfidf
tfidf_vec_tr = help.TfidfEmbeddingVectorizer(word_model)

tfidf_vec_tr.fit(newArr)  # fit tfidf model first
tfidf_doc_vec = tfidf_vec_tr.transform(newArr)


test_tfidf_vec_tr = help.TfidfEmbeddingVectorizer(testWords_model)

test_tfidf_vec_tr.fit(newArr2)  # fit tfidf model first
test_tfidf_doc_vec = test_tfidf_vec_tr.transform(newArr2)

print('Shape of tfidf-word-mean doc2vec...')
print('Save tfidf-word-mean doc2vec as csv file...')
np.savetxt('tfidf_doc_vec.csv', tfidf_doc_vec, delimiter=',')
tfidf_doc_vec = pd.read_csv('tfidf_doc_vec.csv', header=None)

train_X = tfidf_doc_vec
test_X = test_tfidf_doc_vec

for name, clf in zip(names, classifiers):
    print(name)
    clf.fit(train_X, train_y)
    _, _ = sk_evaluate(clf, train_X, train_y, label_names=None)

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, train_X, train_y,
                                     display_labels=[0, 1],
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
"""
"""
clf = MLPClassifier(alpha=10, max_iter=1000, hidden_layer_sizes=(5, 2), random_state=1,
                    solver='lbfgs')
# print(alpha)
clf.fit(train_X, train_y)
_, _ = sk_evaluate(clf, test_X, test_y, label_names=None)

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, test_X, test_y,
                                 display_labels=[0, 1],
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

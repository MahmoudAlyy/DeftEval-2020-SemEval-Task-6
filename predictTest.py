from sklearn.externals import joblib
import pickle
from sklearn.externals import joblib
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
from sklearn.model_selection import KFold

# Load the model from the file
choice = 'MLP Neural Net'
clf = joblib.load('models/'+choice+'.pkl')

test_dir = "data/task1/combined/dev_data_w2v.pkl"
testStrings = pickle.load(open(test_dir, "rb"))
testData = pd.DataFrame(testStrings, columns=['sentence', 'value'])
test_sentences = testData.sentence
# print(test_sentences.split(" "))
y_test = testData.value
with open("data/task1/combined/dev_data_w2v.txt") as f:
    testsentences = f.readlines()
newArr2 = []
for sen in testsentences:
    # print(sen)
    newSen = sen.replace("\t0\n", "").replace("\t1\n", "")
    # print(newSen)
    newSen = newSen.split(" ")
    newArr2.append(newSen)

testWords_model = Word2Vec.load('testwords_model.bin')
mean_vec_tr2 = help.MeanEmbeddingVectorizer(testWords_model)
testDoc_vec = mean_vec_tr2.transform(newArr2)

test_tfidf_vec_tr = help.TfidfEmbeddingVectorizer(testWords_model)

test_tfidf_vec_tr.fit(newArr2)  # fit tfidf model first
test_tfidf_doc_vec = test_tfidf_vec_tr.transform(newArr2)

X_test = test_tfidf_doc_vec


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


_, _ = sk_evaluate(clf, X_test, y_test, label_names=None)
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=[0, 1],
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
figPath = "confusionMatrices/test/"+choice+".png"
plt.savefig(figPath)

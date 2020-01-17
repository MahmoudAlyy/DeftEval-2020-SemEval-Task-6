import pickle
import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix


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
# print(dfWithoutEmbedding.shape)
print(dfWithoutEmbedding.info())
print(dfWithoutEmbedding.head())


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
"""
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
"""

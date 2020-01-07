from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
from collections import Counter
from nltk.corpus import stopwords



from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


### Get train data 
here = os.path.dirname(os.path.realpath(__file__))
save_dir = here + "\\sentences&def.pkl"
strings = pickle.load(open(save_dir, "rb"))

# dict = {}

# for sentence,val in strings:
#     # sentence : string of words
#     # vale : 1 -> has def , 0 -> no def
#     # ur works start from here gg,hf,gl

#     for word in sentence:
#         if word not in dict:
#             dict[word] = 1
#         else :
#             dict[word] = dict[word]  + 1
   
# ### print most common word
# d = Counter(dict)
# print(type(d))
# for k, v in d.most_common(int(len(dict)/2)):
#     print (k," : ", v)

# print("##################################################################################")


### use panda data frame
df = pd.DataFrame(strings, columns=['sentence', 'value'])

#####################################  max feautres ????????????#########################
max_features_value = 4000

vectorizer = CountVectorizer(analyzer="word",preprocessor=None,stop_words="english", max_features=max_features_value) 

train_x = vectorizer.fit_transform(df.sentence)
train_x = train_x.toarray()

train_y = df.value

print(train_x.shape)
print(train_y.shape)



### Get test data
save_dir = here + "\\TEST_sentences&def.pkl"
test_strings = pickle.load(open(save_dir, "rb"))

test_df = pd.DataFrame(test_strings, columns=['sentence', 'value'])
vectorizer = CountVectorizer(analyzer="word",preprocessor=None,stop_words="english", max_features=max_features_value) 

test_x = vectorizer.fit_transform(test_df.sentence)
test_x = test_x.toarray()

test_y = test_df.value

print(test_x.shape)
print(test_y.shape)



naive = MultinomialNB()
classifier = naive.fit(train_x, train_y)
predict = classifier.predict(test_x)


cm = confusion_matrix(predict, test_y)

accuracy = cm.trace()/cm.sum()
print(accuracy)

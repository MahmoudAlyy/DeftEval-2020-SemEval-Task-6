from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
save_dir = "data/task1/combined/train_data.pkl"
strings = pickle.load(open(save_dir, "rb"))

dict = {}


### FINDER ###
tot_key=0
key_def=0
key = "<chem>"

for sentence,val in strings:
    # sentence : string of words
    # vale : 1 -> has def , 0 -> no def
    # ur works start from here gg,hf,gl
    #print(type(sentence))

    if key in sentence:
        tot_key = tot_key + 1
        if val:
            key_def = key_def +1

    for word in sentence.split():
        if word in dict:
            dict[word] = dict[word] + 1
        else:
            dict[word] = 1

#print(tot_key,key_def)
   
#print(dict)

### print 20 most common word
w_name = "testing/common_words.txt"

f= open(w_name, "w+")
d = Counter(dict)
for k, v in d.most_common(20):
    f.write (k+" : "+str(v)+"\n")

# print("##################################################################################")


### use panda data frame
df = pd.DataFrame(strings, columns=['sentence', 'value'])

#####################################  max feautres ????????????#########################
def try_loop(i,n):
    max_features_value = i

    vectorizer = CountVectorizer( max_features=max_features_value,ngram_range=(1,n)) 
    #vectorizer = TfidfVectorizer( max_features=max_features_value ,  ngram_range=(1,n))
    
    train_x = vectorizer.fit_transform(df.sentence)
    

    w_name = "testing/vectorizer vocab.txt"
    d2 = vectorizer.vocabulary_
    f = open(w_name, "w+")
    for k in d2.keys():
        f.write(k+" : " + str(d2[k]) +"\n")

    train_x = train_x.toarray()

    train_y = df.value


    ### Get test data
    save_dir = "data/task1/combined/dev_data.pkl"
    test_strings = pickle.load(open(save_dir, "rb"))

    test_df = pd.DataFrame(test_strings, columns=['sentence', 'value'])

    test_x = vectorizer.fit_transform(test_df.sentence)
    test_x = test_x.toarray()

    test_y = test_df.value


    naive = MultinomialNB()
    classifier = naive.fit(train_x, train_y)
    predict = classifier.predict(test_x)

    cm = confusion_matrix(predict, test_y)


    accuracy = cm.trace()/cm.sum()
    #print(accuracy,i,n)
    return accuracy



# w_name = "testing/logs/log 200-300 1-4 chem.txt"
# f = open(w_name, "w+")

# for i in range(200,300):
#     print(i)
#     for n in range(1,5):
#         try:
#             acc = try_loop(i,n)
#         except:
#             print("Something went wrong")
#         else:
#             #print("Nothing went wrong")
#             f.write(str(acc)+" i = "+str(i)+"  n = "+str(n)+"\n")

#             #print(acc)

print(try_loop(267,2))

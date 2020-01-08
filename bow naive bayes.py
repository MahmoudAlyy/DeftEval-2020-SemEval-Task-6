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

dict = {}

for sentence,val in strings:
    # sentence : string of words
    # vale : 1 -> has def , 0 -> no def
    # ur works start from here gg,hf,gl

    for word in sentence.split():
        if word not in dict:
            dict[word] = 1
        else :
            dict[word] = dict[word]  + 1
   
#print(dict)

### print most common word
w_name = here + "\\common_words.txt"

f= open(w_name, "w+")
d = Counter(dict)
print(type(d))
for k, v in d.most_common(20):
    f.write (k+" : "+str(v)+"\n")

# print("##################################################################################")


### use panda data frame
df = pd.DataFrame(strings, columns=['sentence', 'value'])

#####################################  max feautres ????????????#########################
def try_loop(i,n):
    max_features_value = i

    vectorizer = CountVectorizer(analyzer="word", max_features=max_features_value,ngram_range=(1,n) ) 



    train_x = vectorizer.fit_transform(df.sentence)
    print(vectorizer.vocabulary_)

    train_x = train_x.toarray()

    train_y = df.value

    #print(train_x.shape)
    #print(train_y.shape)



    ### Get test data
    save_dir = here + "\\TEST_sentences&def.pkl"
    test_strings = pickle.load(open(save_dir, "rb"))

    test_df = pd.DataFrame(test_strings, columns=['sentence', 'value'])

    test_x = vectorizer.fit_transform(test_df.sentence)
    test_x = test_x.toarray()

    test_y = test_df.value

    #print(test_x.shape)
    #print(test_y.shape)



    naive = MultinomialNB()
    classifier = naive.fit(train_x, train_y)
    predict = classifier.predict(test_x)

    cm = confusion_matrix(predict, test_y)


    accuracy = cm.trace()/cm.sum()
    print(accuracy,i,n)
    return accuracy



# w_name = here + "\\log.txt"
# f = open(w_name, "w+")

# for i in range(1,200):
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

try_loop(50,1)

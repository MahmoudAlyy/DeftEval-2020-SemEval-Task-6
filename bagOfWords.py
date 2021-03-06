import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
import os
from collections import Counter
from nltk.corpus import stopwords



from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


### Get train data 
here = os.path.dirname(os.path.realpath(__file__))
save_dir = here+ "/data/task1/combined/train_data_bow.pkl"
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

w_name = here + "/testing/common_words.txt"

f= open(w_name, "w+")
d = Counter(dict)
for k, v in d.most_common(20):
    f.write (k+" : "+str(v)+"\n")

# print("##################################################################################")


### use panda data frame
df = pd.DataFrame(strings, columns=['sentence', 'value'])

#####################################  max feautres ????????????#########################
def try_loop(i,n,mindf):
    max_features_value = i

    
    vectorizer = CountVectorizer( max_features=max_features_value,ngram_range=(1,n), min_df=mindf) 
    print("i = ", i, " n-grams 1 -", n, " :")


    #vectorizer = CountVectorizer(ngram_range=(1, n), min_df=mindf)

    #vectorizer = TfidfVectorizer( max_features=max_features_value ,  ngram_range=(1,n))
    
    train_x = vectorizer.fit_transform(df.sentence)
    print("len of features: ", len(vectorizer.vocabulary_))
    
    
    w_name = here +"/testing/vectorizer vocab.txt"
    d2 = vectorizer.vocabulary_
    f = open(w_name, "w+")
    for k in d2.keys():
        f.write(k+" : " + str(d2[k]) +"\n")

    train_x = train_x.toarray()

    train_y = df.value


    ### Get test data
    save_dir = here +"/data/task1/combined/dev_data_bow.pkl"
    test_strings = pickle.load(open(save_dir, "rb"))

    test_df = pd.DataFrame(test_strings, columns=['sentence', 'value'])

    test_x = vectorizer.fit_transform(test_df.sentence)
    test_x = test_x.toarray()

    test_y = test_df.value


    naive = MultinomialNB()
    classifier = naive.fit(train_x, train_y)
    predict = classifier.predict(test_x)

    cm = confusion_matrix(test_y,predict)
    #print(cm)


    accuracy = cm.trace()/cm.sum()
    #print(accuracy,i,n)
    # print(cm)
    # print("accuracy =",accuracy_score(test_y,predict))
    # print("f1 =",f1_score(test_y, predict, average='weighted'))

    print('Score on dataset...\n')
    print('Confusion Matrix:\n', confusion_matrix(test_y, predict))
    print('\nClassification Report:\n', classification_report(
        test_y, predict))
    print('\naccuracy: {:.3f}'.format(accuracy_score(test_y, predict)))
    print('f1 score: {:.3f}'.format(f1_score(test_y, predict, average='weighted')))

    return(f1_score(test_y,predict,average='weighted'))



# w_name = here + "/testing/logs/f1 final 1-250.txt"
# f = open(w_name, "w+")

# for i in range(2,250):
#     print(i)
#     for n in range(1,4):
#         try:
#             acc = try_loop(i,n,1)
#             #acc = try_loop(1, n, i/1000)

#         except:
#             print("Something went wrong")
#         else:
#             #print("Nothing went wrong")
#             f.write("f1 : "+str(acc)+" i = "+str(i)+"  n = "+str(n)+"\n")
#             #f.write("f1 : "+str(acc)+" min df = "+str(i/1000)+"  n = "+str(n)+"\n")

try_loop(98,2,1)


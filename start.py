import pickle
import os
from collections import Counter
from nltk.corpus import stopwords



here = os.path.dirname(os.path.realpath(__file__))
save_dir = here + "\\sentences&def.pkl"
strings = pickle.load(open(save_dir, "rb"))

words = 0
dict = {}

for sentence,val in strings:
    # sentence : list of words
    # vale : 1 -> has def , 0 -> no def
    # ur works start from here gg,hf,gl

    for word in sentence:
        if word not in dict:
            dict[word] = 1
        else :
            dict[word] = dict[word]  + 1
   
### print most common word
d = Counter(dict)
for k, v in d.most_common(25):
    print (k," : ", v)

print("##################################################################################")


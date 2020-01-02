from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import re
import nltk

here = os.path.dirname(os.path.realpath(__file__))
here = here + "\\temp"
#here = here + "\\task1_converted"
strings = []
all_words = []
str
for filename in os.listdir(here):
    f_name = here + "\\" + filename
    f=open(f_name,"r",encoding="utf-8")
    line = f.readline()
    dl = 1
    while line and dl<5 :
        x = line.split("\"") # split by ("")
        
        x[1] = x[1].replace('( [ link ] )','')             ### might not remove idk yet
        x[1] = x[1].lower()     # switch to lower

        x[1] = re.sub('[^a-zA-Z]', ' ', x[1])
        x[1]  = re.sub(r'\s+', ' ', x[1] )


        all_words =all_words [nltk.word_tokenize(x[1])]
        print(all_words)
        
        strings.append( (x[1] , int(x[len(x)-2])))
        dl = dl +1
        line = f.readline()

defc = 0
nodefc = 0
for st,v in strings:
    if v == 1 :
        defc  = defc + 1
    elif v == 0:
        nodefc = nodefc + 1
    else:
        print("?????????????????????")

print("def = ",defc)
print("no def = ",nodefc)


### TF IDF CODE ###
# tfidf = TfidfVectorizer(min_df=2 , max_df=0.5 , ngram_range=(1,3))
# features = tfidf.fit_transform(st for st,v in strings)

# pd.DataFrame(
#     features.todense(),
#     columns=tfidf.get_feature_names()
# )


# print(pd.DataFrame(
#     features.todense(),
#     columns=tfidf.get_feature_names()
# ))




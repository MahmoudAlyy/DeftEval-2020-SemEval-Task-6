#from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import re
here = os.path.dirname(os.path.realpath(__file__))
#here = here + "\\temp"
here = here + "\\task1_converted"
strings = []
str
for filename in os.listdir(here):
    f_name = here + "\\" + filename
    f=open(f_name,"r",encoding="utf-8")
    line = f.readline()
    dl = 1
    while line: # and dl<5 :
        x = line.split("\"") # split by ("")
        #print(x)
        if x[1] == " " or "":
            print("error in file name : ",filename, "line = ",dl)
            print("error in string",x)
            line = f.readline()
            continue

        if x[1][1].isdigit() :
            x[1] = x[1][x[1].find(".")+1:]
        
        x[1] = x[1].replace('( [ link ] )','')             ### might not remove idk yet
        x[1] = x[1].strip()     # remove white spavce
        x[1] = x[1].lower()     # switch to lower
        x[1] = x[1][0:-1]       # remove last dot
        #print(x[1])
        #print(x[len(x)-2])
        
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

### reads form task1_converted and output sentance and value in obj.pkl ###
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import re
import pickle

here = os.path.dirname(os.path.realpath(__file__))
here2 = here + "\\New"
#here2 = here + "\\temp"
#here2 = here + "\\task1_converted"




### check if number exist in a string an replace it with <num>
### dk if its a problem or not if 123str123 ->  123<num>123 
def numString(str):
    x = re.search(r'\d+', str)
    while x != None:
        str = str[0:x.span()[0]] + "<num>" + str[x.span()[1]:]
        x = re.search(r'\d+', str)
    return str

### i am from u.s. -> i am from us ###
def dotString(str):
    str = str.replace(".", "")
    return str


strings = []
all_words = []
str
for filename in os.listdir(here2):
    f_name = here2 + "\\" + filename
    f=open(f_name,"r",encoding="utf-8")
    line = f.readline()
    dl = 1
    while line:# and dl<10 :
        x = line.split("\"")        # split by ("")
        if x[1] == "" or x[1] == " ":                ###  " "" 1 The concept of “ specific use ” involves some sort of commercial application ."	"0"    <--- this line cause an error so i just skip it    
            line = f.readline()                 
            continue

        if x[1][1].isdigit():
            x[1] = x[1][x[1].find('.')+1:]      ### check if line starts with number if so remove the dot after it
        
        x[1] = numString(x[1])
        x[1] = re.sub(r'http\S+', "<link>", x[1])     ### remove links
        x[1] = dotString(x[1])
        
        x[1] = x[1].replace('( [ link ] )','<link>')  
        x[1] = re.sub(r"\b[a-zA-Z]\b", "", x[1])        ### remove single char form string
        x[1] = x[1].lower()     # switch to lower

        x[1] = re.sub('[^a-zA-Z<>]', ' ', x[1])             ### might not remove
        x[1] = x[1].strip()
        
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

for item in strings:
   print(item[0])

save_dir = here + "\\sentences&def.pkl"
pickle.dump( strings, open(save_dir, "wb" ) )


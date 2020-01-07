### reads form task1_converted and output sentance and value in obj.pkl ###
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import re
import pickle
from nltk.corpus import stopwords


here = os.path.dirname(os.path.realpath(__file__))
#here2 = here + "\\New"
#here2 = here + "\\temp"
#here2 = here + "\\task1_converted"
here2 = here + "\\test_data_converted"




### check if number exist in a string an replace it with <num>
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
       
        sentence = []
        sentf=" "
        x = line.split("\"")        # split by ("")
        
        if x[1] == "" or x[1] == " " or x[1].isspace():                ###  " "" 1 The concept of “ specific use ” involves some sort of commercial application ."	"0"    <--- this line cause an error so i just skip it    
            line = f.readline()                 
            continue
        
        ### switch to lower
        x[1] = x[1].lower()     

        ### check if line starts with number if so remove the dot after it
        if x[1][1].isdigit():
            x[1] = x[1][x[1].find('.')+1:] 

        
        ### replace floating point (dots)
        x[1] = re.sub('\d+\.\d+', '<num>', x[1])

        ### replace chemisty values
        x[1] = re.sub(r"\b[a-zA-Z][a-zA-z0-9]*[0-9]\b", "<chem>", x[1])

        ### replace intergers
        x[1] = numString(x[1])

        ### replace links (dots)
        x[1] = re.sub(r'http\S+', "<link>", x[1])

        ### replace i.e  & i.e . (dots)
        x[1] = x[1].replace("i.e ","")
        x[1] = x[1].replace("i.e . ","")

        ### replace u.s. (dots) # only u.s. exists in corpus
        x[1] = x[1].replace("u.s.", "us")

        ### for names like S. I. Tomonaga
        x[1] = x[1].replace(". ", "")

        ### so they all links have same format         
        x[1] = x[1].replace('( [ link ] )','<link>') 
        x[1] = x[1].replace('[ link ]','<link>') 



        ### remove single char form string
        x[1] = re.sub(r"\b[a-zA-Z]\b", "", x[1])   
        # Water ’s   ->  Water ’
        # Sentence example that needs single char removal -> “ ( a ) whether the average person
        # " 830 . Most fungal hyphae are divided into separate cells by endwalls called septa ( singular , septum ) ( [ link ] a , c ) .

        x[1] = re.sub('[^a-zA-Z<>]', ' ', x[1])    

        stopWords = set(stopwords.words('english'))

        for w in x[1].split():
            if w not in stopWords:
                sentence.append(w)

        sentf = sentf.join(sentence)

        strings.append( (sentf, int(x[len(x)-2])))
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

### output to txt file so i can read it with my eyes, the start.py reads from pickle obj 
#w_name = here + "\\sentences&def.txt"
w_name = here + "\\TEST_sentences&def.txt"

f= open(w_name,"w+")
for item in strings:
   f.write(item[0]+"\t"+str(item[1])+"\n")



### pickle save
#save_dir = here + "\\sentences&def.pkl"
save_dir = here + "\\TEST_sentences&def.pkl"

pickle.dump( strings, open(save_dir, "wb" ) )


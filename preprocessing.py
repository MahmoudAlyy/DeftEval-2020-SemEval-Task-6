### reads form task1_converted and output sentance and value in obj.pkl ###
import matplotlib.pyplot as plt  # nltk.download('stopwords')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import re
import pickle
import nltk
import matplotlib.pyplot as plt
plt.rcdefaults()
#nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords

show_variable = -1


def transform(path, w_name_bow, save_dir_bow, w_name_w2v, save_dir_w2v):
    bow_strings = []
    w2v_strings=[]
    for filename in os.listdir(path):
        f_name = path + "/" + filename
        f = open(f_name, "r", encoding="utf-8")

        line = f.readline()
        while line:

            sentence = []
            sentf = " "
            x = line.split("\"")        # split by ("")
            # " "" 1 The concept of “ specific use ” involves some sort of commercial application ."	"0"    <--- this line cause an error so i just skip it
            if x[1] == "" or x[1] == " " or x[1].isspace():
                line = f.readline()
                continue

            # switch to lower
            x[1] = x[1].lower()

            # check if line starts with number if so remove the dot after it
            if x[1][1].isdigit():
                x[1] = x[1][x[1].find('.')+1:]

            # replace floating point (dots)
            x[1] = re.sub(r"\d+\.\d+", '<num>', x[1])

            # replace chemisty values
            x[1] = re.sub(r"\b[a-zA-Z][a-zA-z0-9]*[0-9]\b", "<chem>", x[1])

            # replace intergers
            #x[1] = numString(x[1])
            x[1] = re.sub(r"\b[0-9].*[0-9]\b", '<num>', x[1])

            # replace links (dots)
            x[1] = re.sub(r'http\S+', "<link>", x[1])

            # replace i.e  & i.e . (dots)
            x[1] = x[1].replace("i.e ", "")
            x[1] = x[1].replace("i.e . ", "")

            # replace u.s. (dots) # only u.s. exists in corpus
            x[1] = x[1].replace("u.s.", "us")

            # for names like S. I. Tomonaga
            x[1] = x[1].replace(". ", "")

            # so they all links have same format
            x[1] = x[1].replace('( [ link ] )', '<link>')
            x[1] = x[1].replace('[ link ]', '<link>')

            ### REMOVING link and num ##############################
            x[1] = x[1].replace('<link>', ' ')  # 1191 413
            x[1] = x[1].replace('<num>', ' ')   # 2713 682
            x[1] = x[1].replace('<chem>', ' ')  # 1191 413

            # remove single char form string
            x[1] = re.sub(r"\b[a-zA-Z]\b", "", x[1])
            # Water ’s   ->  Water ’
            # Sentence example that needs single char removal -> “ ( a ) whether the average person
            # " 830 . Most fungal hyphae are divided into separate cells by endwalls called septa ( singular , septum ) ( [ link ] a , c ) .

            # might remove one
            x[1] = re.sub(r"\b[;]*.[;]", " <scolon>", x[1])
            #x[1] = re.sub(r"\b[:]*.[:]", " <colon>", x[1])
            #x[1] = re.sub(r"\b[,]*.[,]", " <comma>", x[1])

            x[1] = re.sub('[^a-zA-Z<>]', ' ', x[1])

            stopWords = set(stopwords.words('english'))

            for w in x[1].split():
                if w not in stopWords:
                    sentence.append(w)

            sentf = sentf.join(sentence)

            ### adding tarek stuff

            final = nltk.word_tokenize(sentf)
            tsentence = []
            final = nltk.pos_tag(final)
            for word, tag in final:
                    tsentence.append(tag)
            final = " ".join(tsentence)

            ### end of tarek

            #final = sentf

            bow_strings.append((final, int(x[len(x)-2])))
            w2v_strings.append((sentf,int(x[len(x)-2])))
            line = f.readline()

    defc = 0
    nodefc = 0
    for st, v in bow_strings:
        if v == 1:
            defc = defc + 1
        elif v == 0:
            nodefc = nodefc + 1

    print("def = ", defc)
    print("no def = ", nodefc)

    ### just ploting

    global show_variable
    show_variable = show_variable + 1
    print(show_variable)

    if show_variable == 0:
        objects = ('Train Def', 'Train NoDef')
        plt.figure("Train Data")

    elif show_variable == 1:
        objects = ('Test Def', 'Test NoDef')
        plt.figure("Test Data")

    y_pos = np.arange(len(objects))
    performance = [defc,nodefc]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.title('Definiton / No Definition Count')

    plt.ion()
    plt.show()
    plt.pause(0.001)

    if show_variable == 1:
        plt.show(block="True")

    ### plot end

    # output to txt file so i can read it with my eyes 0.0 , the start.py reads from pickle obj
    f = open(w_name_bow, "w+")
    for item in bow_strings:
        f.write(item[0]+"\t"+str(item[1])+"\n")

    f = open(w_name_w2v, "w+")
    for item in w2v_strings: 
        f.write(item[0]+"\t"+str(item[1])+"\n")

    # pickle save

    pickle.dump(bow_strings, open(save_dir_bow, "wb"))
    pickle.dump(w2v_strings, open(save_dir_w2v, "wb"))

### END ###


if __name__ == "__main__":

    here = os.path.dirname(os.path.realpath(__file__))
    folderPath = here + "/data/task1/train"
    generatedTxtFile = here + "/data/task1/combined/train_data_bow.txt"
    generatedPklFile = here + "/data/task1/combined/train_data_bow.pkl"
    generatedTxtFilew2v = here + "/data/task1/combined/train_data_w2v.txt"
    generatedPklFilew2v = here + "/data/task1/combined/train_data_w2v.pkl"

    transform(folderPath, generatedTxtFile, generatedPklFile,generatedTxtFilew2v, generatedPklFilew2v)

    folderPath = here + "/data/task1/dev"
    generatedTxtFile = here + "/data/task1/combined/dev_data_bow.txt"
    generatedPklFile = here + "/data/task1/combined/dev_data_bow.pkl"
    generatedTxtFilew2v = here + "/data/task1/combined/dev_data_w2v.txt"
    generatedPklFilew2v = here + "/data/task1/combined/dev_data_w2v.pkl"

    transform(folderPath, generatedTxtFile, generatedPklFile, generatedTxtFilew2v, generatedPklFilew2v)

    # objects = ('Train Def', 'Train NoDef', 'Test Def', 'Test NoDef')
    # y_pos = np.arange(len(objects))
    # performance = [1]
    #plt.show(block="True")

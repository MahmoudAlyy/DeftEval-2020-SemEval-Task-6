### reads form task1_converted and output sentance and value in obj.pkl ###
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import re
import pickle
from nltk.corpus import stopwords


def transform(path, w_name, save_dir):
    strings = []
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
            # x[1] = x[1].replace('<chem>', ' ')  # 1191 413

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

            strings.append((sentf, int(x[len(x)-2])))
            line = f.readline()

    defc = 0
    nodefc = 0
    for st, v in strings:
        if v == 1:
            defc = defc + 1
        elif v == 0:
            nodefc = nodefc + 1

    print("def = ", defc)
    print("no def = ", nodefc)

    # output to txt file so i can read it with my eyes 0.0 , the start.py reads from pickle obj
    f = open(w_name, "w+")
    for item in strings:
        f.write(item[0]+"\t"+str(item[1])+"\n")

    # pickle save

    pickle.dump(strings, open(save_dir, "wb"))

### END ###


if __name__ == "__main__":

    folderPath = "data/task1/train"
    generatedTxtFile = "data/task1/combined/train_data.txt"
    generatedPklFile = "data/task1/combined/train_data.pkl"

    transform(folderPath, generatedTxtFile, generatedPklFile)

    folderPath = "data/task1/dev"
    generatedTxtFile = "data/task1/combined/dev_data.txt"
    generatedPklFile = "data/task1/combined/dev_data.pkl"

    transform(folderPath, generatedTxtFile, generatedPklFile)

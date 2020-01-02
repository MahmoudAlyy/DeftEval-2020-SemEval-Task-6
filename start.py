import pickle
import os


here = os.path.dirname(os.path.realpath(__file__))
save_dir = here + "\\sentences&def.pkl"
strings = pickle.load(open(save_dir, "rb"))


for sentence,val in strings:
    # sentence : self explanatory
    # sentence.split() : retunrs list of words in a sentence
    # vale : 1 -> has def , 0 -> no def
    # ur works start from here gg,hf,gl

    if len(sentence.split()) != 0:      # error in data set mrg3 sentence fadia, no worries
        print(sentence.split())
        print(val)

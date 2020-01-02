import pickle
import os


here = os.path.dirname(os.path.realpath(__file__))
save_dir = here + "\\sentences&def.pkl"
strings = pickle.load(open(save_dir, "rb"))

#for item in strings:
 #   print (item)

#print(strings[-2][0].split())

for word in strings[-2][0].split():
    print(word)

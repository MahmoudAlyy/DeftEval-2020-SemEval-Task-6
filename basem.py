import pickle
import os


### Get train data
here = os.path.dirname(os.path.realpath(__file__))
save_dir = here + "\\sentences&def.pkl"
strings = pickle.load(open(save_dir, "rb"))

### Get test data
save_dir = here + "\\TEST_sentences&def.pkl"
test_strings = pickle.load(open(save_dir, "rb"))

loop_counter = 0

print("print 5 lines from train data:\n")
for sentence, val in strings:
    # sentence : string of words
    # vale : 1 -> has def , 0 -> no def
    # ur works start from here gg,hf,gl
    #print(type(sentence))
    print(sentence,val)
    loop_counter = loop_counter + 1
    if loop_counter > 5:
        break

loop_counter = 0
print("#######################################")
print("print 5 lines from test data:\n")

for sentence, val in test_strings:
    # sentence : string of words
    # vale : 1 -> has def , 0 -> no def
    # ur works start from here gg,hf,gl
    #print(type(sentence))
    print(sentence, val)
    loop_counter = loop_counter + 1
    if loop_counter > 5:
        break




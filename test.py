import os

here = os.path.dirname(os.path.realpath(__file__))
path = here + "/data/task1/dev"
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

            print(x)
            # switch to lower
            x[1] = x[1].lower()

            # check if line starts with number if so remove the dot after it
            if x[1][1].isdigit():
                x[1] = x[1][x[1].find('.')+1:]
            

            line = f.readline()

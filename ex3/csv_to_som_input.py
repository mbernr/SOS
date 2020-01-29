import csv
import numpy as np


dataset = "mice"
column_names = []
data = []
target = []


# read CSV

with open("data/"+dataset+"/"+dataset+".csv") as file:
    csv_reader = csv.reader(file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            column_names = row[1:-4]
            line_count += 1
        else:
            data.append(row[1:-4])
            target.append(row[-1])
            line_count += 1



# replace missing values

def mean_of_column(data, column):
    s = 0
    n = 0
    for i in range(len(data)):
        if data[i][column] != '?':
            s += float(data[i][column])
            n += 1
    return s / n

def std_of_column(data, column):
    temp = [float(data[i][column]) for i in range(len(data))]
    c = np.array(temp, dtype=float)
    return np.std(c)

column_means = [mean_of_column(data, i) for i in range(len(data[0]))]


# scale

for i in range(len(data)):
    for column in range(len(data[i])):
        if data[i][column] == '?':
            data[i][column] = column_means[column]
        else:
            data[i][column] = float(data[i][column])

column_means = [mean_of_column(data, i) for i in range(len(data[0]))]
column_stds = [std_of_column(data, i) for i in range(len(data[0]))]

for i in range(len(data)):
    for column in range(len(data[i])):
        data[i][column] = (data[i][column] - column_means[column]) / column_stds[column]
        data[i][column] = str(data[i][column])


# write .vec file

with open("data/"+dataset+"/"+dataset+".vec", 'w') as file:
    file.write("$TYPE vec \n")
    file.write("$XDIM {} \n".format(len(data)))
    file.write("$YDIM 1 \n")
    file.write("$VEC_DIM {} \n".format(len(data[0])))
    for i in range(len(data)):
        file.write(" ".join(data[i]) + " vec" + str(i+1) + " \n")


# write .cls file

with open("data/"+dataset+"/"+dataset+".cls", 'w') as file:
    file.write("$TYPE class_information \n")
    file.write("$NUM_CLASSES {} \n".format(len(set(target))))
    file.write("$CLASS_NAMES {} \n".format(" ".join(sorted(list(set(target))))))
    file.write("$XDIM 2 \n")
    file.write("$YDIM {} \n".format(len(target)))
    for i in range(len(data)):
        file.write("vec{} {} \n".format(i+1, target[i]))


#write .tv file
with open("data/"+dataset+"/"+dataset+".tv", 'w') as file:
    file.write("$TYPE template \n")
    file.write("$XDIM 2 \n")
    file.write("$YDIM {} \n".format(len(data)))
    file.write("$VEC_DIM {} \n".format(len(data[0])))
    for i in range(len(data[0])):
        file.write("{} {} \n".format(i, column_names[i]))

# write .prop file

with open("data/"+dataset+"/"+dataset+".prop", 'w') as file:
    file.write("""outputDirectory=../maps/{0} 
workingDirectory=./
outputDirectory=../maps/{0}
namePrefix={0}
vectorFileName={0}.vec
sparseData=yes
isNormalized=yes
templateFileName={0}.tv
#cacheSize=

randomSeed=42
xSize=18
ySize=18
learnRate=0.75
sigma=5
#tau=
#metricName=
numIterations=16000

""".format(dataset))

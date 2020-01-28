import csv


dataset = "led"
column_names = []
data = []
target = []


# read CSV

with open("data/"+dataset+"/"+dataset+".csv") as file:
    csv_reader = csv.reader(file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            column_names = row[:-1]
            line_count += 1
        else:
            data.append(row[:-1])
            target.append(row[-1])
            line_count += 1


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
    for i in range(len(column_names)):
        file.write("{} {} \n".format(i, column_names[i]))

# write .prop file

with open("data/"+dataset+"/"+dataset+".prop", 'w') as file:
    file.write("""outputDirectory=../maps/{0} 
namePrefix={0} 
vectorFileName={0}.vec 
templateFileName={0}.tv 
isNormalized=false 
randomSeed=7 
workingDirectory = ./   

useDatabase=false 
databaseServerAddress= 
databaseName=vectors 
databaseUser= 
databasePassword= 
databaseTableNamePrefix={0}

xSize=10 
ySize=10 
learnRate=0.7 
#metricName= 
numIterations=10000 
""".format(dataset))

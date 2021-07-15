import sys, csv ,operator

def reorder_csv(filename):
    data = csv.reader(open(filename),delimiter=',')
    sortedlist = sorted(data, key=operator.itemgetter(21))    # 0 specifies according to first column we want to sort
    #now write the sorte result into new CSV file
    newFileName = filename[:-4] + "_sorted.csv"
    with open(newFileName, "w", newline='') as f:
        fileWriter = csv.writer(f, delimiter=',')
        for row in sortedlist:
            fileWriter.writerow(row)

reorder_csv("./samples/2021_03_23_20_28_39/data.csv")
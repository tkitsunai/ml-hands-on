import csv
import random

with open('iris.csv') as fin:
    reader = csv.reader(fin)
    header = next(reader)
    data = list(reader)

random.shuffle(data)
n_train = int(len(data) * 0.7)

with open('iris_train.csv', 'w') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(data[:n_train])

with open('iris_test.csv', 'w') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(data[n_train:])
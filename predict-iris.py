import csv
from sklearn import svm
import pickle

with open('iris.csv') as fin:
    reader = csv.reader(fin)
    header = next(reader)
    X = []
    y = []
    for d in reader:
        X.append([float(v) for v in d[:-1]])
        y.append(d[-1])

with open('iris-model2.pkl', 'rb') as fin:
        model = pickle.load(fin)

for y_true, y_pred in zip(y, model.predict(X)):
    print(y_true, y_pred)
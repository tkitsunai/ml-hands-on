import csv
from sklearn import svm
import pickle

with open("iris_train.csv") as fin:
    reader = csv.reader(fin)
    header = next(reader)
    X = []
    y = []
    for d in reader:
        X.append([float(v) for v in d[:-1]])
        y.append(d[-1])

model = svm.SVC()

model.fit(X, y)

with open('model/iris_model_train.pkl', 'wb') as fout:
        pickle.dump(model, fout)
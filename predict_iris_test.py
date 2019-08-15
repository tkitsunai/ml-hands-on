import csv
from sklearn import svm
import pickle

with open('iris_test.csv') as fin:
    reader = csv.reader(fin)
    header = next(reader)
    X = []
    y = []
    for d in reader:
        X.append([float(v) for v in d[:-1]])
        y.append(d[-1])

with open('model/iris_model_train.pkl', 'rb') as fin:
        model = pickle.load(fin)

for y_true, y_pred in zip(y, model.predict(X)):
    print(y_true, y_pred)
from datasetup import load_data

train, val, test = load_data()

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

features = ['HR', 'O2Sat', 'Temp', 'Resp', 'Age'] #what should be the baseline features? Predicting SepsisLabel

# I think we agreed on SVM but we can change this to something else
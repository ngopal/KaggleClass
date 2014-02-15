
# Load Data

#from sklearn import datasets
#iris = datasets.load_iris()

import sys
import numpy as np

data = open(sys.argv[1], "r") #open data file for reading
targets = []
inputs = []
for line in data.readlines():
	what_we_see = line.strip('\r\n').split(',')
	print what_we_see
	if 'Choice' not in what_we_see[0]:
		targets.append([float(what_we_see[0])])
		inputs.append([float(i) for i in what_we_see[1:]])
	else:
		continue

print "finished sucking in data"



# Train Data
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(np.array(inputs), np.array(targets)).predict(np.array(inputs))

print y_pred

print("Number of mislabeled points : %d" % (targets != y_pred).sum())

# Make Prediction

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# print("Number of mislabeled points : %d" % sum(iris.target != y_pred))

# Load Data

#from sklearn import datasets
#iris = datasets.load_iris()

import sys
import numpy as np

data = open(sys.argv[1], "r") #open data file for reading
inputs_zero = []
inputs_one = []
for line in data.readlines():
	what_we_see = line.strip('\r\n').split(',')
	if '0' in what_we_see[0]:
		inputs_zero.append([float(i) for i in what_we_see[1:]])
	elif '1' in what_we_see[0]:
		inputs_one.append([float(i) for i in what_we_see[1:]])
	else:
		pass

print "finished sucking in data"



# Train Data
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(np.array(inputs_zero), np.array([float(0) for i in range(len(inputs_zero))])).predict(np.array(inputs_one))

print y_pred

print("Number of mislabeled points : %d" % sum([float(0) for i in range(len(y_pred))] != y_pred))

# Make Prediction

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# print("Number of mislabeled points : %d" % sum(iris.target != y_pred))
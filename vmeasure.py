from pathlib import Path

import numpy as np
from sklearn import metrics

import json
#v_measure_score([0, 0, 1, 1], [0, 0, 1, 1])
#print("%.6f" % v_measure_score([4, 7, 4, 8], [2, 1, 2, 1]))
#y_true = [1, 1, 2, 2]
#y_pred = [2, 2, 1, 87]
#print("v meas %.3f " %  v_measure_score(y_true, y_pred))
#print("homog  %.3f " %  homogeneity_score(y_true, y_pred))
#print("completeness %.3f " %  completeness_score(y_true, y_pred))

data_folder = Path("C:/Users/lauri/IdeaProjects/eSQ/")
classesFile = open( data_folder /"classes.txt", "r")
clustersFile = open( data_folder /"clusters.txt", "r")

classes = classesFile.read()
clusters = clustersFile.read()

classesL = classes.split(",")
clustersL = clusters.split(",")

print ("classes len: ", len(classesL) )
print("clusters len: " , len(clustersL))

v = metrics.v_measure_score (classesL, clustersL)
h = metrics.homogeneity_score(classesL, clustersL)
c = metrics.completeness_score(classesL, clustersL)

print("v measure:    %.4f" % v)
print("homogenity:   %.4f" % h)
print("completeness: %.4f" % c)

resultsFile = open (data_folder /"results.csv" , "w")
resultsFile.write(str(v))
resultsFile.close()


#print("%.6f  clax clust" % v_measure_score(["a", "b"], ["a", "a"]))
#print ("%.6f  clax clust" % v_measure_score(["a", "b"], ["a", "c"])

#print("f1 %.3f " % f1_score(y_true, y_pred, average='macro'))
#print (precision_score(y_true, y_pred, average='macro'))

#y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
#y_pred = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'cat'])
#print (precision_recall_fscore_support(y_true, y_pred, average='macro'))

#print (precision_score(y_true, y_pred, average='macro'))

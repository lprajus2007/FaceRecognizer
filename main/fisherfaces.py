# -*- coding: utf-8 -*-
"""
Created on Tue May 06 11:35:06 2014

@author: lprajus2007
"""
import sys,os
sys.path.append(os.getcwd())
sys.path.append("..")
import numpy as np
from libraries.utility import vectorize, read_images, randomgenerator
from libraries.algorithms import fisherfaces, ffknnsearch
import matplotlib.pyplot as plt

#os.chdir(os.path.dirname(__file__))
print os.getcwd()

c = 15  #Number of classes

# Input N value to read 15N images for training
N = input("Enter N value:")
misrate = [[0.0 for x in xrange(3*N+1)] for x in xrange(10)]
misrate = np.asarray(misrate)
total = [0.0 for x in xrange(10)]
y = []
k = 1
for i in xrange(15):
    for j in xrange(N):
        y.append(k)
    k = k+1

percent = 0
for ei in xrange(10) :
    tr,t = randomgenerator(N)    
    x,yyy = read_images(os.getcwd()+"\..\data", N, tr)
    Xtest,ytest = read_images(os.getcwd()+"\..\data", 11-N, t)
    x = np.asarray(vectorize(x))
    y = np.asarray(y).reshape(-1,1)
    Xtest = np.asarray(vectorize(Xtest))
    x = x.T
    ii = 0
    for num_components in xrange(5,15) :
        [W,mu] = fisherfaces(x,y,num_components)
        #W = np.asarray(W)
        w = np.dot(x,W)
        wp = np.dot(Xtest.T,W)
        idx = ffknnsearch(w,wp)
        idx = [k+1 for k in idx]
        idx.insert(0,0)                 
        count = 0
        for j in xrange(1,16) :
            for k in xrange(1,12-N) :
                if idx[(j-1)*(11-N)+k] < (j-1)*N+1 or idx[(j-1)*(11-N)+k] > j*N :
                    count = count+1
        misrate[ei,ii] = count/float(15*(11-N))
        ii = ii+1
    percent = percent + 10
    print percent,"% complete"

total = np.sum(np.asarray(misrate), axis=0)
total = [x/10 for x in total]
print "Average misclassification rates",total[0:10]

plt.plot([x for x in range(5,15)],total[0:10])
plt.xlabel('d\'')
plt.ylabel('Average Misclassification Rates')
plt.show()
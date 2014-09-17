# -*- coding: utf-8 -*-
"""
Created on Fri May 02 01:35:06 2014

@author: lprajus2007
"""

import sys,os
sys.path.append(os.getcwd())
sys.path.append("..")
import numpy as np
from libraries.utility import vectorize, read_images, randomgenerator
from libraries.algorithms import pca,test, eigenface, knnsearch
import matplotlib.pyplot as plt

#os.chdir(os.path.dirname(__file__))
print os.getcwd()


# Input N value to read 15N images for training
N = input("Enter N value:")

############################################################################################
####################################  EIGEN FACES TRAINING AND TESTING #####################
############################################################################################

misrate = [[0.0 for x in xrange(3*N+1)] for x in xrange(10)]
total = [0.0 for x in xrange(10)]
percent = 0
for i in xrange(10) :
    tr,t = randomgenerator(N)    
    #print r,test
    Xtrain,ytrain = read_images(os.getcwd()+"\..\data", N, tr)
    Xtest,ytest = read_images(os.getcwd()+"\..\data", 11-N, t)    
    # Princical Component Analysis for feature reduction 
    eigenvectors, Z, mu = pca(vectorize(Xtrain), ytrain)
    l = 0
    for num_components in range(0,15*N+1,5):
        W,M = eigenface(eigenvectors,Z,num_components)            
        Wnew = test(vectorize(Xtest), ytest, W, M, mu)
        count = 0
        idx = knnsearch(W,Wnew)        
        idx = [k+1 for k in idx]                   
        idx.insert(0,0)
        count = 0
        for j in xrange(1,16) :
            for k in xrange(1,12-N) :
                if idx[(j-1)*(11-N)+k] < (j-1)*N+1 or idx[(j-1)*(11-N)+k] > j*N :
                    count = count+1
        misrate[i][l] = count/float(15*(11-N))
        l = l+1
    percent = percent + 10
    print percent,"% complete"

total = np.sum(np.asarray(misrate), axis=0)
total = [x/10 for x in total]
print "Average misclassification rates",total

plt.plot([x for x in range(0,15*N+1,5)],total)
plt.xlabel('d\'')
plt.ylabel('Average Misclassification Rates')
plt.show()
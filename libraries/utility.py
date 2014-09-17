# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:35:06 2014

@author: lprajus2007
"""

import os
import PIL.Image as Image
import random
import numpy as np

def randomgenerator(n):
    allimages = []
    allimages.extend(range(1,12))
    r = random.sample(xrange(1,12),n)            
    test = [item for item in allimages if item not in r]
    return r,test
    
def read_images(path, n, r):
    c = 1
    temp = 0
    X = []
    y = [] 
    for dirpath,dirnames,filenames in os.walk(path):
        for filename in filenames:                   
            temp = temp+1
            if (any(r[i]==temp for i in range(n))):
                im = Image.open(os.path.join(dirpath,filename))
                im = im.convert("L")
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(c);
            c = c+1;
            temp=temp%11;
    return [X,y]
 
def vectorize(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1,1)))
    return mat
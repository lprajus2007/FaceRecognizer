# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:35:06 2014

@author: lprajus2007
"""
import numpy as np

def pca(X, y):
    [d,n] = X.shape
    mu = X.mean(axis=1)
    mut = np.tile(mu,(n,1))
    mut = mut.T
    Z = X - mut;
    if n>d:
        C = np.dot(Z,Z.T)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(Z.T,Z)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(Z,eigenvectors)
        for i in xrange(n):
            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
    
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    return eigenvectors,Z,mu
    
def eigenface(eigenvectors,Z,num_components) :
    # select only num_components
    M = eigenvectors[:, 0:num_components]
    W = np.dot(M.T,Z)
    return W, M
    
def euclideandist(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sqrt(np.sum(np.power((p-q),2)))
        
def knnsearch(X, q):
    q = np.asarray(q)
    idx = []
    d1 = []
    for qi in q.T:
        for xi in X.T:
            xi = xi.reshape(-1,1)
            qi = qi.reshape(-1,1)
            d = euclideandist(xi, qi)
            d1.append(d)
        d1 = np.asarray(d1)
        idx_int = np.argsort(d1)
        idx.append(idx_int[0])
        d1 = []
        idx_int = []
    return idx
        
def ffknnsearch(X, q):
    X = np.asarray(X)    
    q = np.asarray(q)
    idx = []
    d1 = []
    for qi in q:
        for xi in X:
            xi = xi.reshape(-1,1)
            qi = qi.reshape(-1,1)
            d = euclideandist(xi, qi)
            d1.append(d)
        idx_int = np.argsort(d1)
        idx.append(idx_int[0])
        d1 = []
        idx_int = []
    return idx

def test(X, y, W, M, mu) :
    [d,n] = X.shape
    mut = X.mean(axis=1)    
    mut = np.tile(mut,(n,1))
    mut = mut.T
    p = X-mut
    Wnew = np.dot(M.T,p)
    return Wnew
    
def lda(X, y, num_components=0) :
    y = np.asarray(y)
    [n,d] = X.shape
    c = np.unique(y)
    meanTotal = X.mean(axis=0)
    Sw = np.zeros((d, d), dtype=np.float32)
    Sb = np.zeros((d, d), dtype=np.float32)
    for i in c:
        Xi = X[np.where(y==i)[0],:]
        meanClass = Xi.mean(axis=0).reshape(1,-1)
        Sw = Sw + np.dot((Xi-meanClass).T, (Xi-meanClass))
        Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
    idx = np.argsort(-eigenvalues.real)
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
    eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
    eigenvectors = np.array(eigenvectors[0:,0:num_components].real, dtype=np.float32, copy=True)
    return [eigenvectors, meanTotal]

def ffpca(X, y, num_components=0) :
    [n,d] = X.shape
    mu = X.mean(axis=0)
    X = X - mu
    if n>d:
        C = np.dot(X.T,X)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X,X.T)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T,eigenvectors)
        for i in xrange(n):
            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    # select only num_components
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:,0:num_components].copy()
    return [eigenvectors, mu]

def fisherfaces(X,y,k) :
    N = X.shape[0]
    labels = np.unique(y)
    c = len(labels)
    k = min(k,c-1)
    [Wpca, mu] = ffpca(X, y, (N-c))
    M = np.dot(X-np.tile(mu,(X.shape[0],1)),Wpca)
    [Wlda, mu_lda] = lda(M, y, k)
    W = np.dot(Wpca,Wlda)
    return W,mu

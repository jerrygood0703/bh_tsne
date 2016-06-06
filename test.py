#!/usr/bin/env python
import sys
import os
from argparse import ArgumentParser, FileType
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from tempfile import mkdtemp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random
from scipy.spatial.distance import pdist, squareform
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import metrics
'''def readInChunks(fileObj, chunkSize=36864):
    """
    Lazy function to read a file piece by piece.
    Default chunk size: 2kB.
    """
    while True:
        data = fileObj.read(chunkSize)
        if not data:
            break
        yield data

data = []
sample_data = []
count = 0
    #for line in fileinput.input(['features_0326.txt']):
with open('features_0326.txt', 'r') as f:
    #    lastlines = collections.deque(f, 10)
    for line in f:
        print count
        count += 1
        if random.sample(xrange(0, 3), 1)[0] == 1:
            sample_data = line.split('\t')
            data.append([float(e) for e in sample_data])
            print len(data), len(data[len(data)-1])'''
#####################################################################3
from sklearn.cluster import DBSCAN
def cluster_DBSCAN(X):
    db = DBSCAN(eps=0.02, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    #print labels
# Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    return labels
##############################################################################
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
def cluster_meanshift(X):
    # Compute clustering with MeanShift
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    return labels
##############################################################################
from sklearn.cluster import AffinityPropagation
def cluster_affinity(norm_res):
    # Compute Affinity Propagation
    af = AffinityPropagation().fit(norm_res)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    #print labels
    n_clusters_ = len(cluster_centers_indices)
    plt.close('all')
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = norm_res[cluster_centers_indices[k]]
        plt.plot(norm_res[class_members, 0], norm_res[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
        for x in norm_res[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    return labels
def labelling(picked, l, img_name, outcount, count):
    print picked
    if picked != None:
        for n in picked:
	    print n
	    if l == n:
	        return picked
    else:
        picked = []
    print img_name
    img = cv2.imread(img_name)
    cv2.imwrite('pool/' + str(outcount) + '_' + str(count) + '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    picked.append(l)
    return picked

data = []
labels_p = []
with open('predict_xy.txt', 'r') as f:
    for line in f:
        sample_data = line.split(' ')
        data.append([float(e) for e in sample_data])
        #print data[len(data)-1]
	if len(data) == 2000:
	    pred_res = np.array(data)
	    max_scaler = preprocessing.MinMaxScaler()
	    norm_pred_res = max_scaler.fit_transform(pred_res)
	    labels_p.append(cluster_meanshift(norm_pred_res))
	    data = []
	    print len(labels_p)
pred_res = np.array(data)
data = []
max_scaler = preprocessing.MinMaxScaler()
norm_pred_res = max_scaler.fit_transform(pred_res)
labels_p.append( cluster_meanshift(norm_pred_res) )
print len(labels_p)
seq_txt = open( '03-26seq.txt', 'r')
outcount = 0
for label in labels_p:
    count = 0
    picked = []
    for l in label:
        print str(l) + ' l '	
        img_name = seq_txt.readline()[:-3] 
        picked = labelling(picked, l, img_name, outcount, count)
        count += 1
    outcount += 1
'''for label in labels_p:
    img_name = seq_txt.readline()[:-3] 
    if label == outcount:  
        print img_name     
        img = cv2.imread(img_name)
        cv2.imwrite('pool/' + str(outcount) + '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    outcount += 1'''























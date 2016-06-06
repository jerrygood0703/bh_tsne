#!/usr/bin/env python

'''
A simple Python wrapper for the bh_tsne binary that makes it easier to use it
for TSV files in a pipeline without any shell script trickery.

Note: The script does some minimal sanity checking of the input, but don't
    expect it to cover all cases. After all, it is a just a wrapper.

[Sylar]Example:
    ./bhtsne.py -i output2.txt -v -p 45 -o res.txt

Example:

    > echo -e '1.0\t0.0\n0.0\t1.0' | ./bhtsne.py -d 2 -p 0.1
    -2458.83181442  -6525.87718385
    2458.83181442   6525.87718385

The output will not be normalised, maybe the below one-liner is of interest?:

    python -c 'import numpy; d = numpy.loadtxt("/dev/stdin");
        d -= d.min(axis=0); d /= d.max(axis=0);
        numpy.savetxt("/dev/stdout", d, fmt='%.8f', delimiter="\t")'

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2013-01-22
'''

# Copyright (c) 2013, Pontus Stenetorp <pontus stenetorp se>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from argparse import ArgumentParser, FileType
from os.path import abspath, dirname, isfile, join as path_join
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from sys import stderr, stdin, stdout
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

### Constants
BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'bh_tsne')
assert isfile(BH_TSNE_BIN_PATH), ('Unable to find the bh_tsne binary in the '
    'same directory as this script, have you forgotten to compile it?: {}'
    ).format(BH_TSNE_BIN_PATH)
# Default hyper-parameter values from van der Maaten (2013)
DEFAULT_NO_DIMS = 2
DEFAULT_PERPLEXITY = 30
DEFAULT_THETA = 0.5
EMPTY_SEED = -1

###

def _argparse():
    argparse = ArgumentParser('bh_tsne Python wrapper')
    argparse.add_argument('-d', '--no_dims', type=int,
                          default=DEFAULT_NO_DIMS)
    argparse.add_argument('-p', '--perplexity', type=float,
            default=DEFAULT_PERPLEXITY)
    # 0.0 for theta is equivalent to vanilla t-SNE
    argparse.add_argument('-t', '--theta', type=float, default=DEFAULT_THETA)
    argparse.add_argument('-r', '--randseed', type=int, default=EMPTY_SEED)
    
    argparse.add_argument('-v', '--verbose', action='store_true')
    argparse.add_argument('-i', '--input', type=FileType('r'), default=stdin)
    argparse.add_argument('-o', '--output', type=FileType('w'),
            default=stdout)
    return argparse


class TmpDir:
    def __enter__(self):
        self._tmp_dir_path = mkdtemp()
        return self._tmp_dir_path

    def __exit__(self, type, value, traceback):
        rmtree(self._tmp_dir_path)


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))

def bh_tsne(samples, no_dims=DEFAULT_NO_DIMS, perplexity=DEFAULT_PERPLEXITY, theta=DEFAULT_THETA, randseed=EMPTY_SEED,
        verbose=False):
    # Assume that the dimensionality of the first sample is representative for
    #   the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)

    # bh_tsne works with fixed input and output paths, give it a temporary
    #   directory to work in so we don't clutter the filesystem
    with TmpDir() as tmp_dir_path:
        # Note: The binary format used by bh_tsne is roughly the same as for
        #   vanilla tsne
        with open(path_join(tmp_dir_path, 'data.dat'), 'wb') as data_file:
            # Write the bh_tsne header
            data_file.write(pack('iiddi', sample_count, sample_dim, theta, perplexity, no_dims))
            # Then write the data
            for sample in samples:
                data_file.write(pack('{}d'.format(len(sample)), *sample))
            # Write random seed if specified
            if randseed != EMPTY_SEED:
                data_file.write(pack('i', randseed))

        # Call bh_tsne and let it do its thing
        with open('/dev/null', 'w') as dev_null:
            bh_tsne_p = Popen((abspath(BH_TSNE_BIN_PATH), ), cwd=tmp_dir_path,
                    # bh_tsne is very noisy on stdout, tell it to use stderr
                    #   if it is to print any output
                    stdout=stderr if verbose else dev_null)
            bh_tsne_p.wait()
            assert not bh_tsne_p.returncode, ('ERROR: Call to bh_tsne exited '
                    'with a non-zero return code exit status, please ' +
                    ('enable verbose mode and ' if not verbose else '') +
                    'refer to the bh_tsne output for further details')

        # Read and pass on the results
        with open(path_join(tmp_dir_path, 'result.dat'), 'rb') as output_file:
            # The first two integers are just the number of samples and the
            #   dimensionality
            result_samples, result_dims = _read_unpack('ii', output_file)
            # Collect the results, but they may be out of order
            results = [_read_unpack('{}d'.format(result_dims), output_file)
                for _ in xrange(result_samples)]
            # Now collect the landmark data so that we can return the data in
            #   the order it arrived
            results = [(_read_unpack('i', output_file), e) for e in results]
            # Put the results in order and yield it
            results.sort()
            for _, result in results:
                yield result
            # The last piece of data is the cost for each sample, we ignore it
            #read_unpack('{}d'.format(sample_count), output_file)
def draw_map(norm_res, labels, data_name):    
    seq_txt = open( data_name + '.txt', 'r')
    map_size = 10000
    roi_size = 200
    #norm_res = np.ceil(norm_res * 1000)
    # full map size = 1000 + roi.size()
    label_file = open( data_name + '_labels.txt', 'w')
    final_map = np.zeros((map_size, map_size, 3), np.uint8)
    for x, label in zip(norm_res, labels):
        img_name = seq_txt.readline()[:-3]        
        img = cv2.imread(img_name)
        label_file.write('%s %d\n' % (img_name, label))
        #cv2.imshow("res", img)
        #cv2.waitKey(10)
        # resize roi to 100x100
        roi = cv2.resize(img,(roi_size, roi_size), interpolation = cv2.INTER_CUBIC)
        a = np.ceil(x[0] * (map_size - roi_size))
        b = np.ceil(x[1] * (map_size - roi_size))
        a = a-((a) % roi_size)
        b = b-((b) % roi_size)
        #print a,b
        final_map[map_size - (b + roi_size):map_size - b, a:a + roi_size, :] = roi
    #print res
    #cv2.imshow("res", final_map)
    #cv2.waitKey()
    cv2.imwrite(data_name + '.jpeg', final_map, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
def draw_concatmap(norm_res, labels, data_name):
    seq_txt = open( data_name + '.txt', 'r')
    map_size = 6000 # size of final image
    roi_size = 100 # size of every image thumbnail
    final_map = np.zeros((map_size, map_size, 3), np.uint8)
    img_list = []
    for it in range(len(norm_res)):
        img_name = seq_txt.readline()[:-3]
        img_list.append(img_name)        
        
    used = np.zeros((len(norm_res), 1))    
    qq = map_size/roi_size
    abes = np.zeros((qq*qq,2))
    i=0
    for a in range(0, map_size, roi_size):
        for b in range(0, map_size, roi_size):
            abes[i,0] = a
            abes[i,1] = b
            i=i+1
    for i in range(len(abes)):
        a = abes[i,0]
        b = abes[i,1]
        xf = a/map_size;
        yf = b/map_size;
        dd = np.sum(np.square(np.subtract(norm_res, [xf,yf])), axis=1)
        index = 0
        for inf in np.nditer(dd, op_flags=['readwrite']):
            if used[index,0] > 0:
                inf[...] = 1000
            index+=1  
        dd_index = np.argmin(dd)
        used[dd_index,0] = 1
        
        img = cv2.imread(img_list[dd_index])
        roi = cv2.resize(img,(roi_size, roi_size), interpolation = cv2.INTER_CUBIC)
        final_map[map_size - (b + roi_size):map_size - b, a:a + roi_size, :] = roi
        # not working!
        #cv2.rectangle(final_map, (int(map_size - (b + roi_size)), int(a)), (int(map_size - b), int(a + roi_size)), (0,0,255), 20)

    cv2.imwrite(data_name + '.jpeg', final_map, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
       
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
##############################################################################
from sklearn.cluster import DBSCAN
def cluster_DBSCAN(X):
    db = DBSCAN(eps=0.1, min_samples=5).fit(X)
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
###############################################################################
def svm_reg(res, new_data):
    clfx = svm.SVR(C=1.0, cache_size=500, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    clfx.fit(new_data, res[:,0]) 
    clfy = svm.SVR(C=1.0, cache_size=500, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    clfy.fit(new_data, res[:,1]) 
    return clfx, clfy
def svm_predict(clfx, clfy, data_name):
    output_xy = []
    data = []
    res_x = []
    res_y = []
    with open(data_name) as f:
        for sample_line in f:
            sample_data = sample_line.split('\t')
            data.append([float(e) for e in sample_data])
            if len(data) == 1000:
                res_x.extend(clfx.predict(data))
                res_y.extend(clfy.predict(data))
                data = []
                print 'predict one batch...'
    res_x.extend(clfx.predict(data))
    res_y.extend(clfy.predict(data))
    return res_x, res_y
################################################################################
import operator
def similarity(data, mean):
    distance_mean = 0
    distance_array = []
    out = []
    for single_data in data:
        dis = math.sqrt(np.sum(np.square(np.subtract(single_data, mean)), axis=0))
        distance_mean += dis
        distance_array.append((single_data, dis))
    # sort tuples by dis
    distance_array.sort(key=operator.itemgetter(1), reverse=True)
    #print distance_array[0]
    #distance_mean = distance_mean / len(data)
    '''for d, da in zip(distance_array, data):
        if d > distance_mean:
            out.append(da)'''
    for i in xrange(20):
        out.append(distance_array[i][0])
    return out
    
def sampling(data_name, mean):
    mean = mean
    new_data = []
    batch_data = []
# first mean[4096]
    with open(data_name, 'r') as f:
        for line in f:           
            sample_data = line.split('\t')
            batch_data.append([float(e) for e in sample_data])
            if len(batch_data) == 100 :
                out = similarity(batch_data, mean)
                for o in out:
                    new_data.append(o)
                np_data = np.array(new_data)
                mean = np.mean(np_data, axis=0)
                batch_data = []
                print len(new_data)
    return new_data

def read_data_random(data_name):
    new_data = []
    with open(data_name, 'r') as f:
        for line in f:           
            if random.sample(xrange(0, 10), 1)[0] == 1:
                sample_data = line.split('\t')
                new_data.append([float(e) for e in sample_data])
                print len(new_data), len(new_data[len(new_data)-1])
    return new_data
############################################################################
def main(args):
    argp = _argparse().parse_args(args[1:])
    data_name = 'features_0326.txt'
    # Read the data, with some sanity checking
    data = []
    data = read_data_random(data_name)
    np_data = np.array(data)
    mean = np.mean(np_data, axis=0)
    del data[:]
    del np_data
    data = sampling(data_name, mean)
    print len(data)
    '''            single_data = [float(e) for e in line.split('\t')]
                if similarity(new_data, single_data):
                    new_data.append(single_data)   '''      
    '''for sample_line_num, sample_line in enumerate((l.rstrip('\n')
            for l in argp.input), start=1):
        sample_data = sample_line.split('\t')
        try:
            assert len(sample_data) == dims, ('Input line #{} of '
                    'dimensionality {} although we have previously observed '
                    'lines with dimensionality {}, possible data error or is '
                    'the data sparsely encoded?'
                    ).format(sample_line_num, len(sample_data), dims)
        except NameError:
            # First line, record the dimensionality
            dims = len(sample_data)
        data.append([float(e) for e in sample_data])'''
    #norm_data = preprocessing.scale(data, axis = 0)
    #max_abs_scaler = preprocessing.MinMaxScaler()
    #new_data = max_abs_scaler.fit_transform(norm_data)
    #pca = PCA(n_components=512)
    #new_data = pca.fit_transform(data)
    #norm_data = preprocessing.scale(new_data, axis = 0)
    #max_abs_scaler = preprocessing.MinMaxScaler()
    #new_data = max_abs_scaler.fit_transform(norm_data)
    #new_data = data
    res = []
    for result in bh_tsne(data, no_dims=argp.no_dims, perplexity=argp.perplexity, theta=argp.theta, randseed=argp.randseed,
            verbose=argp.verbose):
        fmt = ''
        for i in range(1, len(result)):
            fmt = fmt + '{}\t'            
        fmt = fmt + '{}\n'
        argp.output.write(fmt.format(*result))    
        res.append(result)
    res = np.array(res)
    scale_res = preprocessing.scale(res, axis = 0)
    max_abs_scaler = preprocessing.MinMaxScaler()
    norm_res = max_abs_scaler.fit_transform(scale_res)   
    '''d = squareform(pdist(norm_res, 'euclidean'))
    for zero in np.nditer(d, op_flags=['readwrite']):
        if zero[...] == 0:
            zero[...] = 1000
    d_index = np.argmin(d, axis=1) '''
    print "training svm..."
# feed scaled result
    clfx, clfy = svm_reg(scale_res, data)
# trained svm to do regression on extra data
    print 'svm predicting...'
    x, y = svm_predict(clfx, clfy, data_name)
    pred_res = np.array(zip(x,y))
    wf = open('predict_xy.txt', 'w') 
    for xx, yy in zip(x,y):
        wf.write(str(xx) + ' ' + str(yy) + '\n')
# normalize scaled prediction
    max_scaler = preprocessing.MinMaxScaler()
    norm_pred_res = max_scaler.fit_transform(pred_res) 
    #print norm_pred_res 
    #np.set_printoptions(threshold='nan')
    #print d
    #print norm_res
    #plt.plot(norm_res[:,0], norm_res[:,1], 'ro')
    #plt.show()
    #labels = cluster_affinity(norm_res)
    #labels = cluster_DBSCAN(norm_res)
    labels = cluster_meanshift(norm_res)
    labels_p = cluster_meanshift(norm_pred_res) 
    #draw_map(norm_res, labels, data_name = '03-26seq')
    draw_map(norm_pred_res, labels_p, data_name = '03-26seq')
if __name__ == '__main__':
    from sys import argv
    exit(main(argv))

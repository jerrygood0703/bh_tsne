
This software package contains a Barnes-Hut implementation of the t-SNE algorithm. The implementation is described in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf).


[Sylar] clone from https://github.com/lvdmaaten/bhtsne/

# Installation #

Compile the source using the following command:

```
g++ sptree.cpp tsne.cpp -o bh_tsne -O2
```

That's all!

# Usage #

The code comes with wrappers for Matlab and Python. These wrappers write your data to a file called `data.dat`, run the `bh_tsne` binary, and read the result file `result.dat` that the binary produces. There are also external wrappers available for [Torch](https://github.com/clementfarabet/manifold) and [R](https://github.com/jkrijthe/Rtsne). Writing your own wrapper should be straightforward; please refer to one of the existing wrappers for the format of the data and result files.

Demonstration of usage in Matlab:

```matlab
filename = websave('mnist_train.mat', 'https://github.com/awni/cs224n-pa4/blob/master/Simple_tSNE/mnist_train.mat?raw=true');
load(filename);
numDims = 2; pcaDims = 50; perplexity = 50; theta = .5;
map = fast_tsne(digits', numDims, pcaDims, perplexity, theta);
gscatter(map(:,1), map(:,2), labels');
```

# Sylar for CNN feature extraction

prepare image list

copy the list to caffe/cmake_build/examples/_temp/ and /bh_tsne
```bash
cd caffe/
./cmake_build/tools/extract_features models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel cmake_build/examples/_temp/imagenet_val.prototxt fc7 cmake_build/examples/_temp/features 12 leveldb GPU 0 >> feature_human.txt
```
remember to modify imagenet_val.prototxt
```bash
cp feature_human.txt /bh_tsne
```
In bhtsne.py:
```python
seq_txt = open('INIRAtest_list.txt', 'r')
map_size = 2000
cv2.imwrite("map5.jpeg", final_map, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
```
```bash
can be modify
./bhtsne.py -i feature_human.txt -v -p 50 -o res
```

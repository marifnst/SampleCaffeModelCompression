SampleCaffeModelCompression
===========================

This is a really simple code for compressing caffemodel (no algorithm and technique). This code only copy array from pre-trained mode to new model (not all layer).

Prerequisite:

1. you have [caffe](https://github.com/BVLC/caffe) installed in your pc.

How to do compression in terminal ?
-----------------------------------

Quick demo:

    bash run.sh

or:

1. export CAFFE_ROOT = your_caffe_path

2. python sample_compress.py --before_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy.prototxt --after_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy_compress.prototxt --net $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_beforecaffemodel --output_path $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_compressed.caffemodel


To test the model:

    examples$:python 00-classification.py
    
    

Some example results:
---------------------
The testing picture:

![](examples/images/cat.jpg?raw=true)
   
**Original bvlc_reference_caffenet model**:  
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'  
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'



mean-subtracted values: [('B', 104.0069879317889), ('G', 116.66876761696767), ('R', 122.6789143406786)]
predicted class is: 281
output label: n02123045 **tabby, tabby cat**
probabilities and labels:



**Original bvlc_alexnet model**:  
model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'  
model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'

mean-subtracted values: [('B', 104.0069879317889), ('G', 116.66876761696767), ('R', 122.6789143406786)]
predicted class is: 285
output label: n02124075 **Egyptian cat**
probabilities and labels:

**Compressed bvlc_alexnet model**:  
model_def = caffe_root + 'models/bvlc_alexnet/deploy_compress.prototxt'    
model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet_compressed.caffemodel'

mean-subtracted values: [('B', 104.0069879317889), ('G', 116.66876761696767), ('R', 122.6789143406786)]
predicted class is: 254
output label: n02110958 **pug, pug-dog**
probabilities and labels:


Size comparison:
----------------

	cpchung:examples$ ls ../models/bvlc_alexnet/ -lrth
\total 291M  
-rw-rw-r-- 1 cpchung cpchung **233M** Aug 22  2014 bvlc_alexnet.caffemodel  
-rw-rw-r-- 1 cpchung cpchung **3.6K** Feb 22 22:18 deploy.prototxt  
-rw-rw-r-- 1 cpchung cpchung **3.9K** Feb 22 22:18 deploy_compress.prototxt  
-rw-rw-r-- 1 cpchung cpchung  **59M** Feb 22 23:05 bvlc_alexnet_compressed.caffemodel  

Thank you very much for all references below:

1. https://github.com/BVLC/caffe
2. https://github.com/yuanyuanli85/CaffeModelCompression
3. https://github.com/songhan/Deep-Compression-AlexNet
4. https://github.com/rbgirshick/fast-rcnn

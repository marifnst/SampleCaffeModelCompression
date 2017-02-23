#!/bin/bash

# export CAFFE_ROOT=/home/cpchung/temp/caffe

# python sample_compress.py --before_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy.prototxt --after_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy_compress.prototxt --net $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_beforecaffemodel --output_path $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_compressed.caffemodel



# python sample_compress.py 

# --before_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy.prototxt 

# --after_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy_compress.prototxt 


# --net $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_beforecaffemodel 

# --output_path $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_compressed.caffemodel

export CAFFE_ROOT=./models/bvlc_alexnet

# cp /home/cpchung/temp/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel .

# file="/etc/hosts"
file=$CAFFE_ROOT/bvlc_alexnet.caffemodel
if [ -f "$file" ]
then
	echo "$file found."
else
	echo "$file not found."
	wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel -P $CAFFE_ROOT
fi

python sample_compress.py --before_prototxt $CAFFE_ROOT/deploy.prototxt --after_prototxt  $CAFFE_ROOT/deploy_compress.prototxt --net $CAFFE_ROOT/bvlc_alexnet.caffemodel --output_path  $CAFFE_ROOT/bvlc_alexnet_compressed.caffemodel

ls -lrth $CAFFE_ROOT

# SampleCaffeModelCompression

This is a really simple code for compressing caffemodel (no algorithm and technique). This code only copy array from pre-trained mode to new model (not all layer).

prerequisite:
1. you have understood about caffe (https://github.com/BVLC/caffe)

How to use:
1. export CAFFE_ROOT=<your_caffe_path>
2. python sample_compress.py --before_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy.prototxt --after_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy_compress.prototxt --net $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_beforecaffemodel --output_path $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_compressed.caffemodel

thank you very much for all references below:
1. Caffe (https://github.com/BVLC/caffe)
2. https://github.com/yuanyuanli85/CaffeModelCompression
3. https://github.com/songhan/Deep-Compression-AlexNet
4. https://github.com/rbgirshick/fast-rcnn

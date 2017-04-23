#EXAMPLE : 
#1. export CAFFE_ROOT=<your_caffe_path>
#2. python sample_compress.py --before_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy.prototxt --after_prototxt $CAFFE_ROOT/models/bvlc_alexnet/deploy_compress.prototxt --net $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_beforecaffemodel --output_path $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet_compressed.caffemodel

import os, sys
import argparse
import caffe

sys.path.insert(0, 'python')

def parse_args():
    parser = argparse.ArgumentParser(description='Sample CaffeModel Compression')
    parser.add_argument('--before_prototxt', dest='before_prototxt', default=None, type=str)
    parser.add_argument('--after_prototxt', dest='after_prototxt', default=None, type=str)
    parser.add_argument('--net', dest='caffemodel', default=None, type=str)
    parser.add_argument('--output_path', dest='output_path', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    net_before = caffe.Net(args.before_prototxt, args.caffemodel, caffe.TEST)
    net_after = caffe.Net(args.after_prototxt, caffe.TEST)

    layerName = 'conv1'
    data_temp = net_before.params[layerName][0].data[:len(net_after.params[layerName][0].data)]
    data_temp_bias = net_before.params[layerName][1].data[:len(net_after.params[layerName][0].data)]
    net_after.params[layerName][0].data[...] = data_temp
    net_after.params[layerName][1].data[...] = data_temp_bias

    layerName = 'conv2'
    data_temp = net_before.params[layerName][0].data[:len(net_after.params[layerName][0].data),:len(net_after.params[layerName][0].data[0,:])]
    data_temp_bias = net_before.params[layerName][1].data[:len(net_after.params[layerName][1].data)]
    net_after.params[layerName][0].data[...] = data_temp
    net_after.params[layerName][1].data[...] = data_temp_bias

    layerName = 'conv3'
    data_temp = net_before.params[layerName][0].data[:len(net_after.params[layerName][0].data),:len(net_after.params[layerName][0].data[0,:])]
    data_temp_bias = net_before.params[layerName][1].data[:len(net_after.params[layerName][1].data)]
    net_after.params[layerName][0].data[...] = data_temp
    net_after.params[layerName][1].data[...] = data_temp_bias

    layerName = 'conv4'
    data_temp = net_before.params[layerName][0].data[:len(net_after.params[layerName][0].data),:len(net_after.params[layerName][0].data[0,:])]
    data_temp_bias = net_before.params[layerName][1].data[:len(net_after.params[layerName][1].data)]
    net_after.params[layerName][0].data[...] = data_temp
    net_after.params[layerName][1].data[...] = data_temp_bias

    layerName = 'conv5'
    data_temp = net_before.params[layerName][0].data[:len(net_after.params[layerName][0].data),:len(net_after.params[layerName][0].data[0,:])]
    data_temp_bias = net_before.params[layerName][1].data[:len(net_after.params[layerName][1].data)]
    net_after.params[layerName][0].data[...] = data_temp
    net_after.params[layerName][1].data[...] = data_temp_bias

    layerName = 'fc6'
    data_temp = net_before.params[layerName][0].data[:len(net_after.params[layerName][0].data),:len(net_after.params[layerName][0].data[0,:])]
    data_temp_bias = net_before.params[layerName][1].data[:len(net_after.params[layerName][1].data)]
    net_after.params[layerName][0].data[...] = data_temp
    net_after.params[layerName][1].data[...] = data_temp_bias

    layerName = 'fc7'
    data_temp = net_before.params[layerName][0].data[:len(net_after.params[layerName][0].data),:len(net_after.params[layerName][0].data[0,:])]
    data_temp_bias = net_before.params[layerName][1].data[:len(net_after.params[layerName][1].data)]
    net_after.params[layerName][0].data[...] = data_temp
    net_after.params[layerName][1].data[...] = data_temp_bias

    layerName = 'fc8'
    data_temp = net_before.params[layerName][0].data[:len(net_after.params[layerName][0].data),:len(net_after.params[layerName][0].data[0,:])]
    data_temp_bias = net_before.params[layerName][1].data[:len(net_after.params[layerName][1].data)]
    net_after.params[layerName][0].data[...] = data_temp
    net_after.params[layerName][1].data[...] = data_temp_bias

    net_after.save(args.output_path)
    print 'compressed successful'


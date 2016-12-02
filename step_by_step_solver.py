#EXAMPLE : 
#1. export CAFFE_ROOT=<your_caffe_path>
#2. python step_by_step_solver.py --net examples/mnist/lenet_solver.prototxt 

import os, sys
import argparse

sys.path.insert(0, 'python')
import caffe
import google.protobuf as gp
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Sample learning by manual SGD')
    parser.add_argument('--solver', dest='solver', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # define solver type --> please refer to http://caffe.berkeleyvision.org/tutorial/solver.html
    solver = caffe.SGDSolver(args.solver)
    #solver = caffe.AdaDeltaSolver(args.solver)

    # read solver file
    solver_param = caffe.proto.caffe_pb2.SolverParameter()
    with open(args.solver, 'rt') as f:
        gp.text_format.Merge(f.read(), solver_param)

    # print weight & bias of InnerProduct
    print solver.net.params["ip1"][0].data
    print "======================================================================================================"
    print solver.net.params["ip1"][1].data
    print "======================================================================================================"

    # learning with one step
    solver.step(1)
    # or you can used --> uncomment to 2 script below
    # solver.net.forward()
    # solver.net.backward()

    # print after 
    print solver.net.params["ip1"][0].data
    print "======================================================================================================"
    print solver.net.params["ip1"][1].data
    print "======================================================================================================"

    solver.net.save('output.caffemodel')
    print 'step by step with solver success'

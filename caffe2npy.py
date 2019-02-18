# code to convert MSGNET from caffe to npy
import caffe
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--proto-path', type=str, default="./models/NYUV2/deploy.prototxt")
parser.add_argument('--caffemodel-path', type=str, default="./models/NYUV2/cvpr_nyuv2.caffemodel")
parser.add_argument('--output-path', type=str, default="./caffe_params.npy")

import argparse
# PROTO_PATH = 'MSGNet_x8_deploy.prototxt'
# CAFFEMODEL_PATH = 'MSGNet_x8.caffemodel'

if __name__ == '__main__':
    args = parser.parse_args()
    net = caffe.Net(args.proto_path, args.caffemodel_path, caffe.TRAIN)

    layers = {}
    for name, params in net.params.items():
        layers[name] = {}
        # probably only works for DORN's caffe model...
        if "bn" in name: # batchnorm
            print("{} left: {}".format(name, len(params) - 4))
            layers[name]["scale"] = params[0].data
            layers[name]["shift"] = params[1].data
            layers[name]["mean"] = params[2].data
            layers[name]["var"] = params[3].data
        elif "ip" in name: # inner product (i.e. fully connected)
            print("{} left: {}".format(name, len(params) - 2))
            layers[name]["weight"] = params[0].data
            layers[name]["bias"] = params[1].data
        else: # convolution
            layers[name]["weight"] = params[0].data
            if len(params) == 2:
                print("{} left: {}".format(name, len(params) - 2))
                layers[name]["bias"] = params[1].data
            else:
                print("{} left: {}".format(name, len(params) - 1))
    # Save layers to file
    with open(args.output_path, "w") as f:
        print("saving to {}".format(args.output_path))
        np.save(f, layers)


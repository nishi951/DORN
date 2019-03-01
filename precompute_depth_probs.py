import cv2
import caffe
import numpy as np
import scipy.io as sio
import argparse
import os
import pdb
import json

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--filename', type=str, default='./data/NYUV2/demo_01.png', help='path to an image')
parser.add_argument('--rootdir', type=str, default="/home/markn1/spad_single/data/nyu_depth_v2_processed",
                    help="rootdir of dataset")
parser.add_argument('--blacklist', type=str,
                    default="/home/markn1/spad_single/data/nyu_depth_v2_processed/blacklist.txt",
                    help="images to not calculate losses on")
parser.add_argument('--inindexfile', type=str, default="/home/markn1/spad_single/data/nyu_depth_v2_processed/train.json",
                    help="index of dataset to load")
parser.add_argument('--outindexfile', type=str, default="./train_with_probs.json", help="output modified index")
parser.add_argument('--outputroot', type=str, default='/home/markn1/spad_single/data/nyu_depth_v2_processed')


def get_probs(filename, net, pixel_means):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    H = img.shape[0]
    W = img.shape[1]
    img -= pixel_means
    img = cv2.resize(img, (353, 257), interpolation=cv2.INTER_LINEAR)
    data = img.copy()
    data = data[None, :]
    data = data.transpose(0,3,1,2)
    blobs = {}
    blobs['data'] = data
    net.blobs['data'].reshape(*(blobs['data'].shape))
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}

    net.forward(**forward_kwargs)

    pred = net.blobs['log_probs'].data.copy()
    return pred



if __name__ == '__main__':
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net('models/NYUV2/deploy_probs.prototxt', 'models/NYUV2/cvpr_nyuv2.caffemodel', caffe.TEST)
    pixel_means = np.array([[[103.0626, 115.9029, 123.1516]]])

    with open(args.inindexfile, 'r') as f:
        index = json.load(f)

    if args.blacklist is not None:
        # print("Loading blacklist from {}".format(args.blacklist))
        with open(args.blacklist, "r") as f:
            blacklist = [line.strip() for line in f.readlines()]

    for entry in index:
        if entry in blacklist:
            continue
        print(entry)
        if not os.path.exists(args.outputroot):
            os.makedirs(args.outputroot)
        probs_file = entry + "_probs.npy"
        index[entry]["probs"] = probs_file
        if os.path.isfile(os.path.join(args.outputroot, probs_file)):
            # Skip files we've already created
            print("\tskipping...")
            continue
        rgb_file = os.path.join(args.rootdir, index[entry]["rgb"])
        # depth_truth_file = os.path.join(args.rootdir, index[entry]["rawdepth"])

        # import sys
        # sys.exit()
        probs = get_probs(rgb_file, net, pixel_means)


        # img_id = entry.replace("/", "_")

        # Write output to file
        np.save(os.path.join(args.outputroot, probs_file), probs)
        # Write ground truth to file
        # TESTING
        # break
    # Save new index as a json
    with open(args.outindexfile, 'w') as f:
        json.dump(index, f)
    print("done.")




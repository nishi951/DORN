import cv2
import caffe
import numpy as np
import scipy.io as sio
import argparse
import os
import pdb
import json

from split_utils import build_index
from loss import delta, mse, rel_abs_diff, rel_sqr_diff

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--filename', type=str, default='./data/NYUV2/demo_01.png', help='path to an image')
parser.add_argument('--rootdir', type=str, default="/home/markn1/spad_single/data/nyu_depth_v2_processed",
                    help="rootdir of dataset")
parser.add_argument('--blacklist', type=str,
                    default="/home/markn1/spad_single/data/nyu_depth_v2_processed/blacklist.txt",
                    help="images to not calculate losses on")
parser.add_argument('--indexfile', type=str, default="/home/markn1/spad_single/data/nyu_depth_v2_processed/test.json",
                    help="index of dataset to load")
parser.add_argument('--outputroot', type=str, default='./result/NYUV2/nohints', help='output path')
parser.add_argument('--outputlosses', type=str, default='./result/NYUV2/nohints/losses.json',
                    help="records average losses on whole dataset")
parser.add_argument('--max-depth', type=float, default=10.0)
parser.add_argument('--min_depth', type=float, default=0.0)

def depth_prediction(filename, net, pixel_means):
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

    pred = net.blobs['decode_ord'].data.copy()
    pred = pred[0,0,:,:] - 1.0
    pred = pred/25.0 - 0.36
    pred = np.exp(pred)
    ord_score = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
    return ord_score
    #ord_score = ord_score*256.0

def build_index_with_blacklist(rootdir, blacklistfile):
    index = build_index(args.rootdir, ["rgb", "depth"])
    blacklist_file = args.blacklist
    if blacklist_file is not None:
        print("Loading blacklist from {}".format(os.path.join(blacklist_file)))
        with open(blacklist_file, "r") as f:
            blacklist = [line.strip() for line in f.readlines()]
    newindex = {k: index[k] for k in index if k not in blacklist}
    return newindex

class AddDepthMask(): # pylint: disable=too-few-public-methods
    """Creates a mask that is 1 where actual depth values were recorded and 0 where
    the inpainting algorithm failed to inpaint depth.

    eps - small positive number to assign to places with missing depth.

    works in numpy.
    """
    def __call__(self, sample, eps=1e-6):
        closest = (sample["depth"] == np.min(sample["depth"]))
        zero_depth = (sample["rawdepth"] == 0.)
        mask = (zero_depth & closest)
        # print(sample["rawdepth"])
        sample["mask"] = 1. - mask.astype(np.float32) # Logical NOT
        # Set all unknown depths to be a small positive number.
        sample["depth"] = sample["depth"]*sample["mask"] + (1 - sample["mask"])*eps
        sample["eps"] = np.array([eps])
        return sample


def convert_to_uint8(img, min_val, max_val):
    return np.uint8((img - min_val)/(max_val - min_val)*255.0)

if __name__ == '__main__':
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net('models/NYUV2/deploy.prototxt', 'models/NYUV2/cvpr_nyuv2.caffemodel', caffe.TEST)
    pixel_means = np.array([[[103.0626, 115.9029, 123.1516]]])

    with open(args.indexfile, 'r') as f:
        index = json.load(f)

    if args.blacklist is not None:
        # print("Loading blacklist from {}".format(args.blacklist))
        with open(args.blacklist, "r") as f:
            blacklist = [line.strip() for line in f.readlines()]

    print("Running tests...")
    loss_fns = []
    loss_fns.append(("mse", mse))
    loss_fns.append(("delta1", lambda p, t, m: delta(p, t, m, threshold=1.25)))
    loss_fns.append(("delta2", lambda p, t, m: delta(p, t, m, threshold=1.25 ** 2)))
    loss_fns.append(("delta3", lambda p, t, m: delta(p, t, m, threshold=1.25 ** 3)))
    loss_fns.append(("rel_abs_diff", rel_abs_diff))
    loss_fns.append(("rel_sqr_diff", rel_sqr_diff))
    npixels = 0.
    total_losses = {loss_name: 0. for loss_name, _ in loss_fns}

    for entry in index:
        if entry in blacklist:
            continue
        print(entry)
        rgb_file = os.path.join(args.rootdir, index[entry]["rgb"])
        depth_truth_file = os.path.join(args.rootdir, index[entry]["rawdepth"])

        # import sys
        # sys.exit()
        depth = depth_prediction(rgb_file, net, pixel_means)

        depth_truth = cv2.imread(depth_truth_file, cv2.IMREAD_ANYDEPTH)
        depth_truth = depth_truth/1000.

        boolmask = (depth_truth <= args.min_depth) | (depth_truth >= args.max_depth)
        mask = 1.0 - boolmask.astype(float)
        # Calculate metrics
        npixels += np.sum(mask)
        for loss_name, loss_fn in loss_fns:
            avg_loss = loss_fn(depth, depth_truth, mask)
            total_losses[loss_name] += avg_loss * np.sum(mask)

        img_id = entry.replace("/", "_")
        if not os.path.exists(args.outputroot):
            os.makedirs(args.outputroot)
        # Write output to file
        depth_img = convert_to_uint8(depth, args.min_depth, args.max_depth)
        cv2.imwrite(str(args.outputroot + '/' + img_id + '_pred.png'), depth_img)

        # Write ground truth to file
        truth_img = convert_to_uint8(depth_truth, args.min_depth, args.max_depth)
        cv2.imwrite(str(args.outputroot + '/' + img_id + '_truth.png'), truth_img)

        #TESTING
        # break
    # Save as a json
    avg_losses = {loss_name: total_losses[loss_name]/npixels for loss_name in total_losses}
    if "mse" in avg_losses:
        avg_losses["rmse"] = np.sqrt(avg_losses["mse"])
    with open(args.outputlosses, "w") as f:
        json.dump(avg_losses, f)
    print("avg_losses")
    # print(avg_losses)







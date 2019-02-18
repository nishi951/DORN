import cv2
import torch
import numpy as np
import scipy.io as sio
import argparse
import os
import pdb
import json

from split_utils import build_index
from loss import delta, mse, rel_abs_diff, rel_sqr_diff

from demo_nyuv2_full import parser, depth_prediction, convert_to_uint8
from DORN_pytorch import DORN

if __name__ == '__main__':
    args = parser.parse_args()

    # caffe.set_mode_gpu()
    # caffe.set_device(0)
    # net = caffe.Net('models/NYUV2/deploy.prototxt', 'models/NYUV2/cvpr_nyuv2.caffemodel', caffe.TEST)
    net = DORN
    pixel_means = np.array([[[103.0626, 115.9029, 123.1516]]])

    with open(args.indexfile, 'r') as f:
        index = json.load(f)

    if args.blacklist is not None:
        print("Loading blacklist from {}".format(args.blacklist))
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

        depth = depth_prediction(rgb_file)

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
    print(avg_losses)

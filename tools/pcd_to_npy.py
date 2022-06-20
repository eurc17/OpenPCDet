import numpy as np
import argparse
import os
import glob
import open3d as o3d
from pathlib import Path
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
import time


def main(args):
    for pcd_path in sorted(glob.glob(args.input_dir + "/*.pcd")):
        file_stem = os.path.basename(pcd_path).split(".")[0]
        store_path = args.output_dir + "/" + file_stem + ".npy"
        pcd = o3d.io.read_point_cloud(pcd_path)
        np_arr_pts = np.asarray(pcd.points)
        intensity_pad = np.zeros((np_arr_pts.shape[0], 1))
        np_arr_pts_4 = np.concatenate((np_arr_pts, intensity_pad), axis=1)
        reshaped = np_arr_pts_4.reshape(-1, 1).astype(np.float32)
        # print(np_arr_pts)
        np.save(str(store_path), reshaped)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="convert npy files to pcds",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="input directory to the pcd stored",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="output directory to store the npy files",
    )

    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("input_dir is not a directory!")

    if not os.path.isdir(args.input_dir):
        print("input_dir doesn't exists!")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)

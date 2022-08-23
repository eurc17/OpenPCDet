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


def save_pcd(np_arr_pnts, path):
    np.savetxt("tmp.xyz", np_arr_pnts, "%10.5f")
    pcd = o3d.io.read_point_cloud("tmp.xyz")
    print(pcd)
    o3d.io.write_point_cloud(path, pcd)
    os.remove("tmp.xyz")
    pass


def main(args):
    for npy_file in sorted(glob.glob(args.input_dir + "/*.bin")):
        file_stem = os.path.basename(npy_file).split(".")[0]
        # print(file_stem)
        points = np.fromfile(npy_file, dtype=np.float32).reshape(-1, 4)
        output_path = args.output_dir + "/" + file_stem + ".pcd"
        save_pcd(points, output_path)


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
        help="input directory to the npy stored",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="output directory to store the pcd files",
    )

    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("input_dir is not a directory!")

    if not os.path.isdir(args.input_dir):
        print("input_dir doesn't exists!")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)

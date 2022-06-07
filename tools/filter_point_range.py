import numpy as np
import argparse
import os
import glob
from concurrent.futures import ProcessPoolExecutor


def filter_point(file_path):
    global output_dir
    global limit_range
    base_name = os.path.basename(file_path)
    output_path = output_dir + "/" + base_name
    points = np.load(file_path)
    points = points.reshape(-1, 4)
    mask = (
        (points[:, 0] >= limit_range[0])
        & (points[:, 0] <= limit_range[3])
        & (points[:, 1] >= limit_range[1])
        & (points[:, 1] <= limit_range[4])
    )
    points = points[mask]
    points = points.reshape(-1, 1)
    np.save(output_path, points)
    # print(points.shape)


def get_min_max_height(file_path):
    points = np.load(file_path)
    points = points.reshape(-1, 4)
    points_height = points[:, 2]
    # print(points)
    # print(points_height)
    point_max_height = np.amax(points_height)
    point_min_height = np.amin(points_height)

    # print(point_max_height)
    return (point_min_height, point_max_height)


def main(args):
    global output_dir
    global limit_range
    limit_range = [-10, -9, -3, 42, 13, 2.8]
    output_dir = args.output_dir

    # with ProcessPoolExecutor(max_workers=16) as executor:
    #     executor.map(
    #         filter_point,
    #         sorted(glob.glob(args.input_dir + "/*npy")),
    #         chunksize=1,
    #     )

    with ProcessPoolExecutor(max_workers=16) as executor:
        ret = executor.map(
            get_min_max_height,
            sorted(glob.glob(args.input_dir + "/*npy")),
            chunksize=1,
        )
    list_min_max = list(map(list, zip(*list(ret))))
    list_min = list_min_max[0]
    list_max = list_min_max[1]
    print(np.amin(list_min))
    print(np.amax(list_max))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="filter points stored in .npy in kitti bin style with a given range",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="input directory where the .npy files are stored",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="output directory to store the output files",
    )

    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("input_dir is not a directory!")
        exit(1)

    if not os.path.isdir(args.input_dir):
        print("input_dir doesn't exists!")
        exit(1)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)

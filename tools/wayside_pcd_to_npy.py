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


def process_velodyne(pcd_path):
    global velodyne_bin_dir
    store_path = velodyne_bin_dir / (os.path.basename(pcd_path).split(".")[0] + ".npy")
    pcd = o3d.io.read_point_cloud(pcd_path)
    np_arr_pts = np.asarray(pcd.points)
    intensity_pad = np.zeros((np_arr_pts.shape[0], 1))
    np_arr_pts_4 = np.concatenate((np_arr_pts, intensity_pad), axis=1)
    reshaped = np_arr_pts_4.reshape(-1, 1).astype(np.float32)
    # print(np_arr_pts)
    np.save(str(store_path), reshaped)
    # read_back = np.fromfile(str(store_path), dtype=np.float32).reshape(-1, 4)
    # print(read_back)


def process_obj_pnts(pcd_path):
    global bm_car_pkl_dir
    store_path = bm_car_pkl_dir / (os.path.basename(pcd_path).split(".")[0] + ".pkl")
    pcd = o3d.io.read_point_cloud(pcd_path)
    np_arr_pts = np.asarray(pcd.points)
    with open(store_path, "wb") as f:
        pickle.dump(np_arr_pts.astype(np.float32), f)
    # with open(store_path, "rb") as f:
    #     obj_points = pickle.load(f)
    # obj_points = obj_points.reshape([-1, 3])[:, :3].astype(np.float32)
    # print(obj_points.all() == np_arr_pts.all())


def main(args):
    random.seed(10)
    velodyne_dir = Path(args.input_dir) / "velodyne_pcd"
    if not os.path.exists(velodyne_dir):
        print("velodyne_dir doesn't exists!")
        exit(1)
    if not os.path.isdir(velodyne_dir):
        print("velodyne_dir is not a directory!")
        exit(1)
    # bm_car_dir = Path(args.input_dir) / "bm_50maxdist_2num_Car_pcd"
    # if not os.path.exists(bm_car_dir):
    #     print("bm_car_dir doesn't exists!")
    #     exit(1)
    # if not os.path.isdir(bm_car_dir):
    #     print("bm_car_dir is not a directory!")
    #     exit(1)

    # process velodyne into bins
    global velodyne_bin_dir
    velodyne_bin_dir = Path(args.input_dir) / "training" / "velodyne"
    if not os.path.exists(velodyne_bin_dir):
        os.makedirs(velodyne_bin_dir)

    # start = time.time()
    # for pcd_vel_path in sorted(glob.glob(str(velodyne_dir / "*.pcd"))):
    #     process_velodyne(pcd_vel_path)
    # end = time.time()
    # print("for loop time:", end - start)
    # break
    start = time.time()

    with ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(
            process_velodyne,
            sorted(glob.glob(str(velodyne_dir / "*.pcd"))),
            chunksize=1,
        )

    end = time.time()
    print("executor time:", end - start)
    # process occ target
    # global bm_car_pkl_dir
    # bm_car_pkl_dir = Path(args.input_dir) / "bm_50maxdist_2num_Car"
    # if not os.path.exists(bm_car_pkl_dir):
    #     os.makedirs(bm_car_pkl_dir)
    # # for obj_pnts_pcd_path in sorted(glob.glob(str(bm_car_dir / "*.pcd"))):
    # #     process_obj_pnts(obj_pnts_pcd_path)
    # #     break
    # with ProcessPoolExecutor(max_workers=16) as executor:
    #     executor.map(
    #         process_obj_pnts,
    #         sorted(glob.glob(str(bm_car_dir / "*.pcd"))),
    #         chunksize=1,
    #     )

    # create ImageSets
    # image_sets_dir = Path(args.input_dir) / "ImageSets"
    # if not os.path.exists(image_sets_dir):
    #     os.makedirs(image_sets_dir)
    # all_frame_ids = []
    # if args.class_only == None:
    #     for pcd_vel_path in sorted(glob.glob(str(velodyne_dir / "*.pcd"))):
    #         frame_id = os.path.basename(pcd_vel_path).split(".")[0]
    #         all_frame_ids.append(frame_id)
    # else:
    #     bm_class_path = (
    #         Path(args.input_dir) / f"bm_50maxdist_2num_{args.class_only}_pcd"
    #     )
    #     if not os.path.exists(bm_class_path):
    #         print("The bm path to the specified class does not exist!")
    #     for bm_path in sorted(glob.glob(str(bm_class_path / "*.pcd"))):
    #         bm_basename = os.path.basename(bm_path)
    #         frame_id_raw = int(bm_basename.split("_")[0])
    #         frame_id = f"{frame_id_raw:06d}"
    #         if frame_id not in all_frame_ids:
    #             all_frame_ids.append(frame_id)
    #     assert len(all_frame_ids) == len(set(all_frame_ids))
    # random.shuffle(all_frame_ids)
    # # print(all_frame_ids)
    # train_split_size = int(len(all_frame_ids) * (1 - args.val_ratio))
    # # print(train_split_size)
    # # print(val_split_size)
    # train_inds = sorted(all_frame_ids[:train_split_size])
    # val_inds = sorted(all_frame_ids[train_split_size:])
    # # print(len(train_inds))
    # # print(len(val_inds))
    # # print(train_inds)
    # # print(val_inds)
    # train_split_path = image_sets_dir / "train.txt"
    # print(train_split_path)
    # val_split_path = image_sets_dir / "val.txt"
    # with open(train_split_path, "w") as f:
    #     for item in train_inds:
    #         f.write("%s\n" % item)
    # with open(val_split_path, "w") as f:
    #     for item in val_inds:
    #         f.write("%s\n" % item)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="convert wayside generated dataset to kitti bin & pkl format",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="input directory, named detection3d",
    )
    # parser.add_argument(
    #     "-v",
    #     "--val_ratio",
    #     type=float,
    #     required=False,
    #     default=0.2,
    #     help="input directory, named detection3d",
    # )
    # parser.add_argument(
    #     "-c",
    #     "--class_only",
    #     type=str,
    #     required=False,
    #     default="Car",
    #     help="Select only the specified class in ImageSet generation",
    # )

    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("input_dir is not a directory!")

    if not os.path.isdir(args.input_dir):
        print("input_dir doesn't exists!")
    main(args)

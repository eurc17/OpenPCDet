import pcdet.datasets.kitti.kitti_object_eval_python.kitti_common as kitti
from pcdet.datasets.kitti.kitti_object_eval_python.eval import (
    get_coco_eval_result,
    get_official_eval_result,
)
import argparse
import os
from pcdet.utils import calibration_kitti
from pathlib import Path


def get_calib(calib_file):
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)


def _read_imageset_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(
    label_path,
    result_path,
    calib,
    current_class=0,
    coco=False,
    score_thresh=-1,
):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    gt_annos = kitti.get_label_annos(label_path)
    if coco:
        return get_coco_eval_result(
            gt_annos, dt_annos, current_class, calib, lidar_id=args.lidar_id
        )
    else:
        return get_official_eval_result(
            gt_annos, dt_annos, current_class, calib, lidar_id=args.lidar_id
        )


def main(args):
    calib = get_calib(Path(args.calib_file_path))
    result, result_dict = evaluate(
        args.gt_label_dir, args.predict_label_dir, calib, coco=False
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert vector of bboxpvrcnn to kitti labels",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--predict_label_dir",
        type=str,
        required=True,
        help="the path to the directory storing prediction labels",
    )
    parser.add_argument(
        "-g",
        "--gt_label_dir",
        type=str,
        required=True,
        help="the path to the directory storing gt labels",
    )
    parser.add_argument(
        "-c",
        "--calib_file_path",
        type=str,
        required=True,
        help="Path to the calibration file",
    )
    parser.add_argument(
        "-l",
        "--lidar_id",
        type=int,
        required=True,
        choices=range(1, 4),
        help="Select the wayside lidar to evaluate, this is used for difficulty separation",
    )

    args = parser.parse_args()
    if not os.path.exists(args.predict_label_dir):
        print("predict_label_dir doesn't exists!")

    if not os.path.isdir(args.predict_label_dir):
        print("predict_label_dir is not a directory!")

    if not os.path.exists(args.gt_label_dir):
        print("gt_label_dir doesn't exists!")

    if not os.path.isdir(args.gt_label_dir):
        print("gt_label_dir is not a directory!")

    main(args)

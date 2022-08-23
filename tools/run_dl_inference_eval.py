import os
import glob
import argparse


def main(args):
    # run demo.py
    if not "lidar{}".format(args.lidar_select) in args.input_dir:
        print(
            "WARNING: Lidar select and input dir might be different, comment exit(1) in code to continue"
        )
        exit(1)
    input_dir_len = len(
        [
            name
            for name in os.listdir(args.input_dir)
            if os.path.isfile(os.path.join(args.input_dir, name))
        ]
    )
    print("Input dir length = ", input_dir_len)
    model_name = os.path.basename(args.cfg_file).split(".")[0]
    inference_folder_name = "eval2_with_gen_{}_lidar{}".format(
        model_name, args.lidar_select
    )
    output_dir_inference = "./official_transform/wayside_labels/{}".format(
        inference_folder_name
    )
    cmd = "poetry run python3 ./demo.py --cfg_file {} --data_path {} --ckpt {} --output_dir {}".format(
        args.cfg_file, args.input_dir, args.ckpt, output_dir_inference
    )
    os.system(cmd)
    if (
        len(
            [
                name
                for name in os.listdir(output_dir_inference)
                if os.path.isfile(os.path.join(output_dir_inference, name))
            ]
        )
        != input_dir_len
    ):
        print(
            "Number of files generated not matching that in input_dir after running demo.py, please check before continuing"
        )
        exit(1)
    # run bboxpvtcnn_to_kitti.py (no merge class)
    output_dir_bboxpvtcnn = "./official_transform/class_{}_kitti".format(
        inference_folder_name
    )
    cmd = "poetry run python3 ./bboxpvtcnn_to_kitti.py -i {} -o {} -c {}".format(
        output_dir_inference, output_dir_bboxpvtcnn, args.calib_file
    )
    os.system(cmd)
    if (
        len(
            [
                name
                for name in os.listdir(output_dir_bboxpvtcnn)
                if os.path.isfile(os.path.join(output_dir_bboxpvtcnn, name))
            ]
        )
        != input_dir_len
    ):
        print(
            "Number of files generated not matching that in input_dir after running bboxpvtcnn_to_kitti.py, please check before continuing"
        )
        exit(1)
    # run evaluate_with_labels
    gt_label_dir = "./official_transform/class_gt_eval2_revised_l{}_5p_kitti/".format(
        args.lidar_select
    )
    cmd = (
        "poetry run python3 ./evaluate_with_labels.py -p {}  -g {} -c {} -l {}".format(
            output_dir_bboxpvtcnn, gt_label_dir, args.calib_file, args.lidar_select
        )
    )
    os.system(cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run demo.py, bboxpvtcnn_to_kitti.py, and evaluate_with_labels.py",
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
        "-f",
        "--cfg_file",
        type=str,
        required=True,
        help="The path to the model config file",
    )
    parser.add_argument(
        "-k",
        "--ckpt",
        type=str,
        required=True,
        help="The path to the model checkpoint file",
    )
    parser.add_argument(
        "-l",
        "--lidar_select",
        type=int,
        required=True,
        choices=range(1, 4),
        help="The LiDAR number to select",
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default="./label_transform_test/calib.txt",
        help="The path to the calib.txt",
    )

    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("input_dir doesn't exists!")
        exit(1)

    if not os.path.isdir(args.input_dir):
        print("input_dir is not a directory!")
        exit(1)

    if not os.path.exists(args.cfg_file):
        print("cfg_file doesn't exists!")
        exit(1)

    if not os.path.isfile(args.cfg_file):
        print("cfg_file is not a file!")
        exit(1)

    if not os.path.exists(args.ckpt):
        print("ckpt doesn't exists!")
        exit(1)

    if not os.path.isfile(args.ckpt):
        print("ckpt is not a file!")
        exit(1)

    if not os.path.exists(args.calib_file):
        print("calib_file doesn't exists!")
        exit(1)

    if not os.path.isfile(args.calib_file):
        print("calib_file is not a file!")
        exit(1)

    main(args)

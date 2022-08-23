import argparse
import glob
from pathlib import Path
import os


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="specify the directory storing the point cloud data",
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        help="specify the config for demo",
        required=True,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="specify the path to the pretrained model",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="specify the output dir to store the bboxpvrcnn labels",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_config()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for data_path in sorted(glob.glob(args.data_dir + "/*.npy")):
        # print(data_path)
        file_stem = os.path.basename(data_path).split(".")[0]
        # print(file_stem)
        os.system(
            "poetry run python3 demo.py --cfg_file "
            + args.cfg_file
            + " --ckpt "
            + args.ckpt
            + " --data_path "
            + data_path
            + "  --output_path "
            + args.output_dir
            + "/"
            + file_stem
            + ".json"
        )


if __name__ == "__main__":
    main()

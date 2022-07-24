import argparse
import glob
from pathlib import Path
import json
import os

try:
    import open3d
    from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        ext=".bin",
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = (
            glob.glob(str(root_path / f"*{self.ext}"))
            if self.root_path.is_dir()
            else [self.root_path]
        )

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == ".bin":
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32
            ).reshape(-1, 4)
        elif self.ext == ".npy":
            # print(self.sample_file_list[index])
            # points = np.load(self.sample_file_list[index])
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32
            ).reshape(-1, 4)
            # print(points)
        else:
            raise NotImplementedError

        input_dict = {
            "points": points,
            "frame_id": index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/kitti_models/second.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data",
        help="specify the point cloud data file or directory",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="specify the pretrained model"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".npy",
        help="specify the extension of your point cloud data file",
    )
    parser.add_argument(
        "--draw_vis", action="store_true", help="Specify to draw visualization results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="specify the output directory of the bboxpvrcnn labels",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.output_dir != None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    logger = common_utils.create_logger()
    logger.info("-----------------Quick Demo of OpenPCDet-------------------------")
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
    )
    logger.info(f"Total number of samples: \t{len(demo_dataset)}")

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        # print(demo_dataset[0])
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f"Visualized sample index: \t{idx + 1}")
            data_dict = demo_dataset.collate_batch([data_dict])
            print("batch collated!")
            load_data_to_gpu(data_dict)
            print("data loaded to gpu!")
            pred_dicts, _ = model.forward(data_dict)
            print("prediction ran!")
            # print(pred_dicts[0]["pred_boxes"])
            # print(pred_dicts[0]["pred_scores"])
            # print(pred_dicts[0]["pred_labels"])
            mask = (
                (pred_dicts[0]["pred_boxes"][:, 0] != torch.inf)
                & (pred_dicts[0]["pred_boxes"][:, 1] != torch.inf)
                & (pred_dicts[0]["pred_boxes"][:, 2] != torch.inf)
                & (pred_dicts[0]["pred_boxes"][:, 3] != torch.inf)
                & (pred_dicts[0]["pred_boxes"][:, 4] != torch.inf)
                & (pred_dicts[0]["pred_boxes"][:, 5] != torch.inf)
                & (pred_dicts[0]["pred_boxes"][:, 6] != torch.inf)
            )
            # print(mask)
            pred_dicts[0]["pred_boxes"] = pred_dicts[0]["pred_boxes"][mask]
            pred_dicts[0]["pred_scores"] = pred_dicts[0]["pred_scores"][mask]
            pred_dicts[0]["pred_labels"] = pred_dicts[0]["pred_labels"][mask]

            if args.draw_vis:
                V.draw_scenes(
                    points=data_dict["points"][:, 1:],
                    ref_boxes=pred_dicts[0]["pred_boxes"],
                    ref_scores=pred_dicts[0]["pred_scores"],
                    ref_labels=pred_dicts[0]["pred_labels"],
                )

                if not OPEN3D_FLAG:
                    mlab.show(stop=True)

            output_dict = dict()
            output_dict["pred_boxes"] = (
                pred_dicts[0]["pred_boxes"].cpu().detach().numpy()
            )
            output_dict["pred_scores"] = (
                pred_dicts[0]["pred_scores"].cpu().detach().numpy()
            )
            output_dict["pred_labels"] = (
                pred_dicts[0]["pred_labels"].cpu().detach().numpy()
            )

            output_dict["class_names"] = [
                cfg.CLASS_NAMES[pred_label - 1]
                for pred_label in output_dict["pred_labels"]
            ]

            # print(output_dict)
            bbox_list = list()
            for (i, pred_box) in enumerate(output_dict["pred_boxes"]):
                bbox_pvrcnn = dict()
                bbox_pvrcnn["x"] = float(pred_box[0])
                bbox_pvrcnn["y"] = float(pred_box[1])
                bbox_pvrcnn["z"] = float(pred_box[2])
                bbox_pvrcnn["dx"] = float(pred_box[3])
                bbox_pvrcnn["dy"] = float(pred_box[4])
                bbox_pvrcnn["dz"] = float(pred_box[5])
                bbox_pvrcnn["heading"] = float(pred_box[6])
                bbox_pvrcnn["score"] = float(output_dict["pred_scores"][i])
                bbox_pvrcnn["label"] = int(output_dict["pred_labels"][i])
                bbox_pvrcnn["cluster_id"] = i
                bbox_pvrcnn["devices"] = []
                bbox_pvrcnn["class_name"] = output_dict["class_names"][i]
                bbox_list.append(bbox_pvrcnn)

            print(demo_dataset.sample_file_list[idx])
            file_stem = os.path.basename(demo_dataset.sample_file_list[idx]).split(".")[
                0
            ]

            if args.output_dir != None:
                output_path = args.output_dir + "/" + file_stem + ".json"

                with open(output_path, "w") as fp:
                    json.dump(bbox_list, fp, indent=4)

    logger.info("Demo done.")


if __name__ == "__main__":
    main()

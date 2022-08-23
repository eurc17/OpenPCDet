import os
import numpy as np
from pcdet.utils import box_utils, calibration_kitti
import json
import argparse
import glob
from pathlib import Path


def get_calib(calib_file):
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)


def get_template_prediction(num_samples):
    ret_dict = {
        "name": np.zeros(num_samples),
        "truncated": np.zeros(num_samples),
        "occluded": np.zeros(num_samples),
        "alpha": np.zeros(num_samples),
        "bbox": np.zeros([num_samples, 4]),
        "dimensions": np.zeros([num_samples, 3]),
        "location": np.zeros([num_samples, 3]),
        "rotation_y": np.zeros(num_samples),
        "score": np.zeros(num_samples),
        "boxes_lidar": np.zeros([num_samples, 7]),
    }
    return ret_dict


def generate_single_sample_dict(batch_index, box_dict, calib, class_names):
    pred_scores = box_dict["pred_scores"].cpu().numpy()
    pred_boxes = box_dict["pred_boxes"].cpu().numpy()
    pred_labels = box_dict["pred_labels"].cpu().numpy()
    pred_dict = get_template_prediction(pred_scores.shape[0])
    if pred_scores.shape[0] == 0:
        return pred_dict

    image_shape = batch_dict["image_shape"][batch_index].cpu().numpy()
    pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
    pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
        pred_boxes_camera, calib, image_shape=image_shape
    )

    pred_dict["name"] = np.array(class_names)[pred_labels - 1]
    pred_dict["alpha"] = (
        -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
    )
    pred_dict["bbox"] = pred_boxes_img
    pred_dict["dimensions"] = pred_boxes_camera[:, 3:6]
    pred_dict["location"] = pred_boxes_camera[:, 0:3]
    pred_dict["rotation_y"] = pred_boxes_camera[:, 6]
    pred_dict["score"] = pred_scores
    pred_dict["boxes_lidar"] = pred_boxes

    return pred_dict


def generate_prediction_dicts(
    batch_dict, pred_dicts, class_names, calib, output_path=None
):
    """
    Args:
        batch_dict:
            frame_id:
        pred_dicts: list of pred_dicts
            pred_boxes: (N, 7), Tensor
            pred_scores: (N), Tensor
            pred_labels: (N), Tensor
        class_names:
        output_path:

    Returns:

    """

    annos = []
    for index, box_dict in enumerate(pred_dicts):
        frame_id = batch_dict["frame_id"][index]

        single_pred_dict = generate_single_sample_dict(
            index, box_dict, calib, class_names
        )
        single_pred_dict["frame_id"] = frame_id
        annos.append(single_pred_dict)

        if output_path is not None:
            cur_det_file = output_path / ("%s.txt" % frame_id)
            with open(cur_det_file, "w") as f:
                bbox = single_pred_dict["bbox"]
                loc = single_pred_dict["location"]
                dims = single_pred_dict["dimensions"]  # lhw -> hwl

                for idx in range(len(bbox)):
                    print(
                        "%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
                        % (
                            single_pred_dict["name"][idx],
                            single_pred_dict["alpha"][idx],
                            bbox[idx][0],
                            bbox[idx][1],
                            bbox[idx][2],
                            bbox[idx][3],
                            dims[idx][1],
                            dims[idx][2],
                            dims[idx][0],
                            loc[idx][0],
                            loc[idx][1],
                            loc[idx][2],
                            single_pred_dict["rotation_y"][idx],
                            single_pred_dict["score"][idx],
                        ),
                        file=f,
                    )

    return annos


def generate_single_label_with_score(box_dict, calib, class_names):
    pred_scores = box_dict["pred_scores"]
    pred_boxes = box_dict["pred_boxes"]
    pred_labels = box_dict["pred_labels"]
    pred_dict = get_template_prediction(pred_scores.shape[0])
    if pred_scores.shape[0] == 0:
        print("Error, no score. Template returned!")
        return pred_dict

    # image_shape = batch_dict["image_shape"][batch_index].cpu().numpy()
    pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
    pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
        pred_boxes_camera, calib, image_shape=None
    )

    pred_dict["name"] = np.array(class_names)[pred_labels - 1]
    pred_dict["alpha"] = (
        -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
    )
    pred_dict["bbox"] = pred_boxes_img
    pred_dict["dimensions"] = pred_boxes_camera[:, 3:6]
    pred_dict["location"] = pred_boxes_camera[:, 0:3]
    pred_dict["rotation_y"] = pred_boxes_camera[:, 6]
    pred_dict["score"] = pred_scores
    pred_dict["boxes_lidar"] = pred_boxes

    return pred_dict


def main(args):
    calib = get_calib(Path(args.calib_file_path))
    output_path = Path(args.output_dir)
    for i, wayside_label_path in enumerate(
        sorted(glob.glob(args.input_dir + "/*.json"))
    ):
        print(wayside_label_path)
        box_dict = dict()
        pred_scores_list = list()
        pred_boxes_list = list()
        pred_labels_list = list()
        with open(wayside_label_path, "r") as fp:
            wayside_label = json.load(fp)
        for object in wayside_label:
            pred_scores_list.append(object["score"])
            if object["class_name"] != "":
                if object["class_name"] == "Car":
                    pred_labels_list.append(1)
                elif object["class_name"] == "Truck":
                    pred_labels_list.append(2)
                elif object["class_name"] == "Cyclist":
                    pred_labels_list.append(3)
                else:
                    # print(object["class_name"])
                    pred_labels_list.append(1)

            else:
                continue
            box = []
            box.append(object["x"])
            box.append(object["y"])
            box.append(object["z"])
            box.append(object["dx"])
            box.append(object["dy"])
            box.append(object["dz"])
            box.append(object["heading"])
            pred_boxes_list.append(box)

        pred_scores = np.array(pred_scores_list)
        pred_boxes = np.array(pred_boxes_list)
        pred_labels = np.array(pred_labels_list)

        box_dict["pred_scores"] = pred_scores
        box_dict["pred_boxes"] = pred_boxes
        box_dict["pred_labels"] = pred_labels

        if args.merge_class:
            print("Merged class")
            kitti_pred_dict = generate_single_label_with_score(
                box_dict, calib, ["Car", "Car", "Car"]
            )
        else:
            kitti_pred_dict = generate_single_label_with_score(
                box_dict,
                calib,
                ["Car", "Truck", "Cyclist"]
                # box_dict,
                # calib,
                # ["Cyclist", "Car", "Truck", "SemiTruck", "Pickup", "Other", "Unknown"],
            )

        # print(kitti_pred_dict)
        frame_id = "{:06d}".format(i)
        # print(frame_id)
        if output_path is not None:
            cur_det_file = output_path / ("%s.txt" % frame_id)
            with open(cur_det_file, "w") as f:
                bbox = kitti_pred_dict["bbox"]
                loc = kitti_pred_dict["location"]
                dims = kitti_pred_dict["dimensions"]  # lhw -> hwl

                if args.add_score:

                    for idx in range(len(bbox)):
                        print(
                            "%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
                            % (
                                kitti_pred_dict["name"][idx],
                                kitti_pred_dict["alpha"][idx],
                                bbox[idx][0],
                                bbox[idx][1],
                                bbox[idx][2],
                                bbox[idx][3],
                                dims[idx][1],
                                dims[idx][2],
                                dims[idx][0],
                                loc[idx][0],
                                loc[idx][1],
                                loc[idx][2],
                                kitti_pred_dict["rotation_y"][idx],
                                kitti_pred_dict["score"][idx],
                            ),
                            file=f,
                        )
                else:
                    for idx in range(len(bbox)):
                        print(
                            "%s 0.0 0 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
                            % (
                                kitti_pred_dict["name"][idx],
                                kitti_pred_dict["alpha"][idx],
                                bbox[idx][0],
                                bbox[idx][1],
                                bbox[idx][2],
                                bbox[idx][3],
                                dims[idx][1],
                                dims[idx][2],
                                dims[idx][0],
                                loc[idx][0],
                                loc[idx][1],
                                loc[idx][2],
                                kitti_pred_dict["rotation_y"][idx],
                            ),
                            file=f,
                        )

        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert vector of bboxpvrcnn to kitti labels",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="input directory to the vec bboxpvrcnn labels",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="output directory to store the kitti labels",
    )
    parser.add_argument(
        "-c",
        "--calib_file_path",
        type=str,
        required=True,
        help="Path to the calibration file",
    )
    parser.add_argument(
        "--add_score", action="store_true", help="To write the score or not"
    )
    parser.add_argument(
        "--merge_class", action="store_true", help="Merge all class into car"
    )

    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("input_dir is not a directory!")

    if not os.path.isdir(args.input_dir):
        print("input_dir doesn't exists!")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)

import numpy as np
import torch
from ultralytics.utils import ops
import os


def read_pose(pose, name="pose.txt"):
    """
    Read pose data from a text file and organize it into a dictionary.

    Args:
        pose (dict): A dictionary containing pose data organized by frames.
        name (str): Name of the input text file containing pose data (default is "pose.txt").

    This function reads pose data from a text file and organizes it into a dictionary where
    each frame's data is stored as a list of object information, including object poses and bounding
    box coordinates.

    Returns:
        pose (dict): A dictionary containing pose data organized by frames.
    """
    if os.path.exists(name):
        file_pose = open(name)
        lines = file_pose.readlines()

        for line in lines:
            line = [int(x) for x in line.split()]
            if line[0] not in pose:
                pose[line[0]] = []

            # read track like {frame_id: [pose_data, box_data]}
            pose[line[0]].append([line[1:35], [line[-4], line[-3]]])
        file_pose.close()

    return pose


def correct_pose(pose, current_frame, name):
    """
    Correct object IDs in a tracking dataset based on various metrics and thresholds.

    Args:
        pose (dict): A dictionary containing pose data organized by frames.
        current_frame (int): The current frame number.
        name (str): Name of the input pose dataset.

    This function corrects pose data in a pose dataset based on subtraction of coordinates of box's centers from pose
    coordinates.

    Returns:
        pose (dict): The pose data with corrected object poses.
        current_frame (int): The current frame number.
    """

    # read pose
    pose = read_pose(pose, name)

    if current_frame in pose.keys():
        for i in range(len(pose[current_frame])):

            # subtraction of X coordinates
            for x in range(0, len(pose[current_frame][i][0]), 2):
                if pose[current_frame][i][0][x] != 0:
                    pose[current_frame][i][0][x] = pose[current_frame][i][0][x] - pose[current_frame][i][1][0]

            # subtraction of Y coordinates
            for y in range(1, len(pose[current_frame][i][0]), 2):
                if pose[current_frame][i][0][y] != 0:
                    pose[current_frame][i][0][y] = pose[current_frame][i][0][y] - pose[current_frame][i][1][1]

    return pose, current_frame


def write_pose_results(txt_path, results, frame_idx):
    """
    Write pose results to a text file.

    Args:
        txt_path (Path): The path to the text file where pose results will be written.
        results (object): Object containing pose results, including boxes and their poses.
        frame_idx (int): The index of the current frame.

    This function takes tracking results, frame index, and a path to a text file and
    writes the tracking results in the MOT (Multiple Object Tracking) format to the
    specified text file. It first prepares the data in the required format and then
    appends it to the text file.

    Returns:
        None
    """

    nr_dets = len(results.boxes)

    # create a tensor with frame indices for each detection
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    if frame_idx.shape[0] != 0:
        pose = results.keypoints.xy.to('cpu').view(frame_idx.shape[0], -1)
    else:
        pose = results.keypoints.xy.to('cpu').view(-1, )

    # concatenate frame index, object poses, and bounding boxes in the MOT format:[frame_id, box_id, box_data]
    mot = torch.cat([
        frame_idx,
        pose,
        ops.xyxy2xywh(results.boxes.xyxy).to('cpu')
    ], dim=1)

    # open the text file in append binary mode and save the tracking data as integers
    with open(str(txt_path), 'ab+') as f:
        np.savetxt(f, mot.numpy(), fmt='%d')

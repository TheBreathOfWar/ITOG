import numpy as np
from utils_track import cal_distance


def mean_pose(data):
    """
    Sets the mean pose per frame for boxes without pose.

    Args:
        data (dict): The track and pose data.

    This function corrects track and pose data in a pose dataset based mean pose per frame.

    Returns:
        data (dict): The pose data with corrected object data.
    """

    # iterate through frames
    for frame in sorted(list(data.keys())):
        count = 0

        # set mean pose
        mean = [0] * 34
        mean = np.zeros_like(mean, dtype='float32')

        # iterate through boxes per current frame
        for box in data[frame]:
            if box[2]:

                # update mean pose
                mean += np.array(box[2], dtype='float32')
                count += 1
        if count != 0:

            # get mean pose
            mean = mean / count

            # add pose to boxes without pose
            for i in range(len(data[frame])):
                if not data[frame][i][2]:
                    data[frame][i][2] = [int(x) for x in list(mean)]

    return data


def show_poses(data):
    """
    Show missed poses for each frame from previous frame.

    Args:
        data (dict): A dictionary containing pose data organized by frames.

    This function processes the pose data and add missed poses for each frame.

    Returns:
        data (dict): The updated pose data with added poses.
    """

    # get previous frame
    last_frame = sorted(list(data.keys()))[0]

    # dictionary for previous poses IDs
    already_ids = {}

    # add previous poses IDs from previous frame
    for box in data[last_frame]:
        if len(box) == 3:
            already_ids[box[0]] = box[2]

    # iterate trough frames
    for frame in sorted(list(data.keys())):

        # iterate through boxes per current frame
        for i in range(len(data[frame])):

            # update previous poses IDs
            if len(data[frame][i]) == 3:
                already_ids[data[frame][i][0]] = data[frame][i][2]

            # add missed poses
            if len(data[frame][i]) == 2 and data[frame][i][0] in already_ids:
                data[frame][i].append(already_ids[data[frame][i][0]])

            # if not found
            if len(data[frame][i]) == 2 and data[frame][i][0] not in already_ids:
                data[frame][i].append([])

    return data


def write_data(data, name="data.txt"):
    """
    Write track and pose data to a text file in a specific format.

    Args:
        data (dict): A dictionary containing pose data organized by frames.
        name (str): The name of the output text file to write the pose data (default is "data.txt").

    This function writes the tracking data to a text file in a specified format. It iterates through
    the frames and object information and formats it as "frame_id x y w h pose" for each object.

    Returns:
        None
    """
    with open(name, "w") as file:

        # iterate trough frames
        for frame in sorted(list(data.keys())):
            for box in data[frame]:

                # write track data like: 'frame_id box_data pose_data'
                file.write(str(frame) + ' ' + str(box[0]) + ' ' + ' '.join([str(x) for x in box[1]]) + ' ' + ' '.join(
                    [str(x) for x in box[2]]) + '\n')


def connection(track, pose, same_thr):
    """
    This function combines tracking results and poses using a threshold for the same objects.

    Args:
        track (dict): A dictionary containing tracking data organized by frames.
        pose (dict): A dictionary containing pose data organized by frames.
        same_thr (int): Threshold for distance between same objects.

    This function processes the tracking data and add missed boxes for each frame.

    Returns:
        track (dict): The connected track and pose data.
    """

    # iterate through frames
    for frame in sorted(list(track.keys())):

        # chose frames with pose
        if frame in sorted(list(pose.keys())):

            # iterate through boxes per current frame
            for i in range(len(track[frame])):

                # get box from track
                dot_1 = track[frame][i][1]
                pose_possible = []
                min_distance = same_thr
                for pose_box in pose[frame]:

                    # get box from pose
                    dot_2 = pose_box[1]

                    # calculate distance between boxes and compare with a threshold
                    if cal_distance(dot_1, dot_2) <= min_distance:
                        min_distance = cal_distance(dot_1, dot_2)
                        pose_possible = pose_box[0]

                # update track
                track[frame][i].append(pose_possible)

    return track

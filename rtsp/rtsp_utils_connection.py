import numpy as np
from rtsp_utils_track import cal_distance


def mean_pose(track, current_frame):
    """
    Sets the mean pose per frame for boxes without pose.

    Args:
        track (dict): The track and pose data.
        current_frame (int): The current frame number.

    This function corrects track and pose data in a pose dataset based mean pose per frame.

    Returns:
        track (dict): The pose data with corrected object data.
    """
    if current_frame in track.keys():

        # set mean pose
        count = 0
        mean = [0] * 34
        mean = np.zeros_like(mean, dtype='float32')

        # iterate through boxes per current frame
        for box in track[current_frame]:
            if box[2]:

                # update mean pose
                mean += np.array(box[2], dtype='float32')
                count += 1
        if count != 0:

            # get mean pose
            mean = mean / count

            # add pose to boxes without pose
            for i in range(len(track[current_frame])):
                if not track[current_frame][i][2]:
                    track[current_frame][i][2] = [int(x) for x in list(mean)]

    return track


def show_poses(track, current_frame, last_frame_connection, already_ids_connection):
    """
    Show missed poses for each frame from previous frame.

    Args:
        track (dict): A dictionary containing pose data organized by frames.
        current_frame (int): The current frame number.
        last_frame_connection (int): Previous frame number.
        already_ids_connection (dict): A dictionary containing past IDs.

    This function processes the pose data and add missed poses for each frame.

    Returns:
        track (dict): The updated pose data with added poses.
        current_frame (int): The current frame number.
        last_frame_connection (int): Previous frame number.
        already_ids_connection (dict): A dictionary containing past IDs.
    """

    # get previous frame
    if last_frame_connection == 0 and track != {}:
        last_frame_connection = sorted(list(track.keys()))[0]

        # add previous poses IDs from previous frame
        for box in track[last_frame_connection]:
            if box[2]:
                already_ids_connection[box[0]] = box[2]

    # iterate through boxes per current frame
    if current_frame in track.keys():
        for i in range(len(track[current_frame])):
            # update previous poses IDs
            if track[current_frame][i][2]:
                already_ids_connection[track[current_frame][i][0]] = track[current_frame][i][2]
            # add missed poses
            if track[current_frame][i][2] == [] and track[current_frame][i][0] in already_ids_connection:
                track[current_frame][i][2] = already_ids_connection[track[current_frame][i][0]]

    return track, current_frame, last_frame_connection, already_ids_connection


def connect(track, pose, current_frame, same_thr):
    """
    This function combines tracking results and poses using a threshold for the same objects.

    Args:
        track (dict): A dictionary containing tracking data organized by frames.
        pose (dict): A dictionary containing pose data organized by frames.
        current_frame (int): The current frame number.
        same_thr (int): Threshold for distance between same objects.

    This function processes the tracking data and add missed boxes for each frame.

    Returns:
        track (dict): The connected track and pose data.
    """

    # chose frames with pose
    if current_frame in track.keys():

        # iterate through boxes per current frame
        for i in range(len(track[current_frame])):

            # get box from track
            dot_1 = track[current_frame][i][1]
            pose_possible = []
            min_distance = same_thr
            if current_frame in pose.keys():
                for pose_box in pose[current_frame]:

                    # get box from pose
                    dot_2 = pose_box[1]

                    # calculate distance between boxes and compare with a threshold
                    if cal_distance(dot_1, dot_2) <= min_distance:
                        min_distance = cal_distance(dot_1, dot_2)
                        pose_possible = pose_box[0]

            # update track
            track[current_frame][i].append(pose_possible)

    return track

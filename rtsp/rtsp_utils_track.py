from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT

import numpy as np
from numpy import dot
from numpy.linalg import norm

from ultralytics.utils import ops
from ultralytics.utils.plotting import save_one_box

from pathlib import Path
import os
import cv2

import torch
from torchvision import transforms


def on_predict_start(predictor):
    """
    This function is called at the beginning of the prediction process.

    Args:
        predictor (Predictor): The Predictor object used for prediction.

    This function loads the tracking configuration from a file, creates trackers
    based on the specified tracking method and its settings, and performs an
    initial setup for the trackers.

    Returns:
        None
    """

    # load the tracking configuration from a file
    tracking_config = \
        ROOT / \
        'boxmot' / \
        'configs' / \
        (predictor.custom_args.track_method + '.yaml')

    # create a list to store trackers
    trackers = []

    # create trackers for each item in the dataset
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.track_method,
            tracking_config,
            Path(predictor.custom_args.track_re_id_model),
            predictor.device,
            False,
            False
        )

        # if the tracker has a model, perform warm-up
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    # assign the list of trackers to the predictor
    predictor.trackers = trackers


def write_track_results(txt_path, results, frame_idx):
    """
    Write tracking results to a text file in the MOT format.

    Args:
        txt_path (Path): The path to the text file where tracking results will be written.
        results (object): Object containing tracking results, including boxes and their IDs.
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

    # concatenate frame index, object IDs, and bounding boxes in the MOT format:[frame_id, box_id, box_data]
    mot = torch.cat([
        frame_idx,
        results.boxes.id.unsqueeze(1).to('cpu'),
        ops.xyxy2xywh(results.boxes.xyxy).to('cpu'),
    ], dim=1)

    # create the parent folder for the text file if it doesn't exist
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    # create or touch the text file
    txt_path.touch(exist_ok=True)

    # open the text file in append binary mode and save the tracking data as integers
    with open(str(txt_path), 'ab+') as f:
        np.savetxt(f, mot.numpy(), fmt='%d')


def change_number(track, current_frame, next_number, ids):
    """
    Update object IDs in a tracking dataset to ensure they are continuous and without gaps.

    Args:
        track (dict): A dictionary containing tracking data organized by frames.
        current_frame (int): The current frame number.
        next_number (int): The next available object ID.
        ids (dict): A dictionary to store mappings of old IDs to new IDs.

    This function iterates through the tracking data and updates object IDs to ensure
    they are continuous without gaps. It also maintains a mapping of old IDs to new IDs.

    Returns:
        track (dict): The updated tracking data with continuous object IDs.
        current_frame (int): The current frame number.
        next_number (int): The next available object ID.
        ids (dict): A dictionary to store mappings of old IDs to new IDs.
    """

    # update object IDs within the current frame
    if current_frame in track.keys():
        for i in range(len(track[current_frame])):
            if track[current_frame][i][0] in ids:
                track[current_frame][i] = [ids[track[current_frame][i][0]], track[current_frame][i][1]]

        # sort boxes within the frame based on object IDs
        for box in sorted(track[current_frame], key=lambda x: x[0]):
            if box[0] > next_number:

                # update object IDs and maintain the mapping
                for i in range(len(track[current_frame])):
                    if track[current_frame][i][0] == box[0]:
                        track[current_frame][i] = [next_number + 1, track[current_frame][i][1]]
                        ids[box[0]] = next_number + 1
                        break
                next_number += 1
    return track, current_frame, next_number, ids


def add_frame(track, current_frame):
    """
    Ensure that tracking data includes frames up to a specified maximum frame number.

    Args:
        track (dict): A dictionary containing tracking data organized by frames.
        current_frame (int): The current frame number.

    This function checks if the tracking data includes frames up to the specified
    maximum frame number. If any frames are missing, it duplicates the data from the
    previous frame to fill in the gaps and ensure that data is available for all frames.

    Returns:
        track (dict): The updated tracking data with frames extended up to the maximum frame number.
        current_frame (int): The current frame number.
    """

    # if frame is missed then add
    if current_frame not in track.keys():
        if current_frame - 1 in track.keys():
            track[current_frame] = track[current_frame - 1]
    return track, current_frame


def del_not_human(track, current_frame, conf, kernel_size, human_conv):
    """
    Remove non-human objects from the tracking data based on a confidence threshold.

    Args:
        track (dict): A dictionary containing track data organized by frames.
        current_frame (int): The current frame number.
        conf (float): Confidence threshold for determining whether an object is human or not.
        kernel_size (int): Size of the kernel for image convolution.
        human_conv (list): human convolution vector.

    This function processes the tracking data and removes objects that are not considered
    human based on a confidence threshold. It uses an image convolution technique with a
    specified kernel size and a precomputed human convolution kernel to make this determination.

    Returns:
        track (dict): The updated tracking data with non-human objects removed.
        current_frame (int): The current frame number.
        human_conv (list): human convolution vector.

    """

    if current_frame in track.keys():

        # list for delete candidates per frame
        del_candidates = []

        # get frame image for crops
        image = cv2.imread(f'images/{current_frame}.jpg')

        # iterate through boxes per frame
        for box in track[current_frame]:
            boxxy = xywh2xyxy(box[1])

            # get box's crop
            save_one_box(torch.FloatTensor(boxxy), image, BGR=True)

            # get box's vector
            crop = get_conv('im.jpg', kernel_size)

            # get cosine_similarity of human vector and box vector
            metric = cosine_similarity(human_conv, crop)
            os.remove('im.jpg')

            # comparison with threshold
            if metric < conf:
                del_candidates.append(box)

        # remove non-human candidates from the frame's object list
        for candidate in del_candidates:
            track[current_frame].remove(candidate)

    return track, current_frame, human_conv


def show_boxes(track, current_frame, last_frame_show, missed_ids_show):
    """
    Show missed boxes for each frame from previous frame.

    Args:
        track (dict): A dictionary containing track data organized by frames.
        current_frame (int): The current frame number.
        last_frame_show (int): # Previous frame number.
        missed_ids_show (dict): A dictionary containing missed IDs.

    This function processes the tracking data and add missed boxes for each frame.

    Returns:
        track (dict): The updated tracking data with added boxes.
        current_frame (int): The current frame number.
        last_frame_show (int): # Previous frame number.
        missed_ids_show (dict): A dictionary containing missed IDs.
    """

    # add missed IDs from previous frame
    if last_frame_show == 0 and track != {}:
        last_frame_show = sorted(list(track.keys()))[0]

        for box in track[last_frame_show]:
            missed_ids_show[box[0]] = box[1]

    # get IDs for current frame
    now_ids = []
    if current_frame in track.keys():
        for box in track[current_frame]:
            now_ids.append(box[0])

        # search missed IDs in current frame
        for idx in missed_ids_show:

            # if we find ID then update missed IDs dictionary
            if idx in now_ids:
                for box in track[current_frame]:
                    if box[0] == idx:
                        missed_ids_show[idx] = box[1]
                        break

            # if we not find ID then add box to current frame
            else:
                track[current_frame].append([idx, missed_ids_show[idx]])

        # update missed IDs dictionary for new IDs from current frame
        for idx in now_ids:
            if idx not in missed_ids_show:
                for box in track[current_frame]:
                    if box[0] == idx:
                        missed_ids_show[idx] = box[1]
                        break

    return track, current_frame, last_frame_show, missed_ids_show


def correct_ids(name: str,
                kernel_size: int,
                same_thr: int,
                minimum_metric: int,
                conv_importance: int,
                distance_thershold: int,
                conv_thershold: float,
                weight: int,
                close_thr: int,
                possible_thr: int,
                track: dict,
                current_frame: int,
                last_frame_track: int,
                already_ids_track: dict,
                crops_path: str,
                convs: dict) -> object:
    """
     Correct object IDs in a tracking dataset based on various metrics and thresholds.

     Args:
         name (str): Name of the input tracking dataset.
         kernel_size (int): Size of the kernel for image convolution.
         same_thr (float): Threshold for identifying objects as the same.
         min_metric (float): Minimum metric value for matching objects.
         conv_importance (float): Importance of convolution in object matching.
         distance_threshold (float): Threshold for distance between objects.
         conv_threshold (float): Threshold for convolution similarity.
         weight (float): Weight for combining convolution features.
         close_thr (float): Threshold for objects considered close.
         possible_thr (float): Threshold for considering objects as possible matches.
         track (dict): A dictionary containing track data organized by frames.
         current_frame (int): The current frame number.
         last_frame_track (int): Previous frame number.
         missed_ids_track (dict): A dictionary containing missed IDs.
         crops_path (str): A path to boxes' crops.
         convs (dict): A dictionary containing convolution vectors for boxes.

     This function corrects object IDs in a tracking dataset based on various metrics and thresholds.
     It processes the data to handle cases where objects are considered the same, find and update
     missing object IDs, and refine the object IDs based on convolution features and distances.

     Returns:
         track (dict): The tracking data with corrected object IDs.
         current_frame (int): The current frame number.
         last_frame_track (int): Previous frame number.
         missed_ids_track (dict): A dictionary containing missed IDs.
         crops_path (str): A path to boxes' crops.
         convs (dict): A dictionary containing convolution vectors for boxes.
     """

    # read track
    track = read_track(track, name)

    # get convolution vectors for crops from previous frame
    if last_frame_track == 0 and track != {}:
        last_frame_track = sorted(list(track.keys()))[0]

        for box in track[last_frame_track]:
            already_ids_track[box[0]] = [box[1], last_frame_track]
            path = crops_path + f'/{last_frame_track - 1}/{box[0]}.jpg'
            convs[box[0]] = get_conv(path, kernel_size)

    if current_frame in track.keys():

        # delete duplicate human boxes within the same frame based on a distance threshold
        track[current_frame] = del_same(current_frame, track, same_thr)

        # list if IDs for current frame
        now_convs = {}


        # get now IDs and new IDs and update convolution dictionary for boxes from current frame
        for box in track[current_frame]:
            path = crops_path + f'/{current_frame - 1}/{box[0]}.jpg'
            now_convs[box[0]] = get_conv(path, kernel_size)

        now_ids = list(now_convs.keys())


        # iterate trough new IDs
        print(now_convs.keys())
        print(convs.keys())
        print(already_ids_track)
        print(now_ids)
        for idx in now_ids:
            for i in range(len(track[current_frame])):
                if idx == track[current_frame][i][0]:
                    dot_1 = track[current_frame][i][1]

                    # total flag shows whether we have found at least one match box
                    total_flag = False

                    # skip flag shows we have found very close box
                    skip = True

                    # minimum value for general metric
                    min_metric = minimum_metric
                    new_id = idx

                    # list of possible dots with False metric and reasonable distance
                    possible = []

                    # search new ID in missed IDs
                    for already_id in already_ids_track:
                        dot_2 = already_ids_track[already_id][0]
                        count = current_frame - already_ids_track[already_id][1]

                        # get general metrics
                        flag, metric, distance, conv = get_metrics(dot_1, dot_2, now_convs[idx], convs[already_id], count,
                                                                   conv_importance, distance_thershold,
                                                                   conv_thershold)

                        # compare threshold for very close boxes
                        if distance < close_thr:
                            new_id = already_id
                            skip = False
                            break

                        # compare conditions for possible boxes
                        if not flag and distance < possible_thr + count:
                            possible.append((already_id, distance))

                        # update total flag
                        total_flag = total_flag or flag

                        # compare general metric
                        if flag and metric < min_metric:
                            min_metric = metric
                            new_id = already_id

                    # add closets possible ID
                    if new_id == idx and possible != [] and not total_flag and skip:
                        new_id = min(possible, key=lambda x: x[1])[0]

                    # update find ID, missed IDs dictionary, convolution dictionary, now IDs, new IDs
                    if new_id != idx:
                        print(new_id, idx)
                        convs[new_id] = (convs[new_id] + weight * now_convs[idx]) / (weight + 1)
                        #del already_ids_track[new_id]
                        track[current_frame][i] = [new_id, track[current_frame][i][1]]
                        already_ids_track[new_id] = [dot_1, current_frame]
                        now_ids.remove(idx)
                        #now_ids.append(new_id)
                        break

        for idx in now_ids:
            for i in range(len(track[current_frame])):
                if idx == track[current_frame][i][0]:
                    dot_1 = track[current_frame][i][1]
                    already_ids_track[idx] = [dot_1, current_frame]
                    convs[idx] = now_convs[idx]

        # delete duplicate human boxes within the same frame based on a distance threshold
        track[current_frame] = del_same(current_frame, track, same_thr)

        # update previous frame and previous frame IDs
        last_frame_track = current_frame

    return track, current_frame, last_frame_track, already_ids_track, crops_path, convs


def del_same(frame, data, same_thr):
    """
    Remove duplicate objects within the same frame based on a distance threshold.

    Args:
        frame (int): The frame number for which duplicates should be removed.
        data (dict): A dictionary containing tracking data organized by frames.
        same_thr (float): Threshold for identifying duplicate objects.

    This function processes tracking data for a specific frame and removes duplicate objects
    based on a distance threshold. It ensures that only one instance of each duplicate object
    is retained in the frame's object list.

    Returns:
        data[frame] (list): The updated object list for the specified frame.
    """

    # list for delete candidates per frame
    boxes = []

    # iterate first box trough boxes per frame
    for i in range(len(data[frame])):
        dot_1 = data[frame][i][1]
        min_distance = same_thr
        same_dot_id = i
        # iterate second box trough boxes per frame
        for j in range(i, len(data[frame])):
            dot_2 = data[frame][j][1]

            # calculate distance between centers of first and second box and compare with a threshold
            if cal_distance(dot_1, dot_2) <= min_distance and i != j:
                min_distance = cal_distance(dot_1, dot_2)
                same_dot_id = j

        # leave a smaller index
        if data[frame][i][0] < data[frame][same_dot_id][0]:
            if data[frame][same_dot_id] not in boxes:
                boxes.append(data[frame][same_dot_id])
        elif data[frame][i][0] > data[frame][same_dot_id][0]:
            if data[frame][i] not in boxes:
                boxes.append(data[frame][i])

    # remove extra boxes per frame
    for box in boxes:
        data[frame].remove(box)

    return data[frame]


def get_conv(path, kernel_size):
    """
    Compute the convolution feature of an image using a specified kernel size.

    Args:
        path (str): Path to the image file for which the convolution feature is computed.
        kernel_size (int): Size of the kernel used for convolution.

    This function reads an image, applies convolution using a specified kernel size, and
    returns the convolution feature as a numpy array.

    Returns:
        conv_feature (numpy array): The computed convolution feature of the image.
    """

    # get image
    image = cv2.imread(path)
    convert_tensor = transforms.ToTensor()
    convert_resize = transforms.Resize((64, 64))
    image = convert_resize(convert_tensor(image))
    image = image.view(64, 64, 3)

    # build convolution model
    conv = torch.nn.Conv2d(64, 1, kernel_size=kernel_size, bias=False, padding=(kernel_size - 1) // 2)
    kernel = torch.tensor([[0., 0., 1., 0., 0.],
                           [0., 1., 1., 1., 0.],
                           [1., 1., 1., 1., 1.],
                           [0., 1., 1., 1., 0.],
                           [0., 0., 1., 0., 0.]])

    # chose kernel size for convolution model
    if kernel_size == 3:
        kernel = torch.tensor([[0., 1., 0.],
                               [1., 1., 1.],
                               [0., 1., 0.]])
    elif kernel_size == 5:
        kernel = torch.tensor([[0., 0., 1., 0., 0.],
                               [0., 1., 1., 1., 0.],
                               [1., 1., 1., 1., 1.],
                               [0., 1., 1., 1., 0.],
                               [0., 0., 1., 0., 0.]])
    elif kernel_size == 7:
        kernel = torch.tensor([[0., 0., 0., 1., 0., 0., 0.],
                               [0., 0., 1., 1., 1., 0., 0.],
                               [0., 1., 1., 1., 1., 1., 0.],
                               [1., 1., 1., 1., 1., 1., 1.],
                               [0., 1., 1., 1., 1., 1., 0.],
                               [0., 0., 1., 1., 1., 0., 0.],
                               [0., 0., 0., 1., 0., 0., 0.]])
    elif kernel_size == 9:
        kernel = torch.tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 1., 1., 0., 0., 0.],
                               [0., 0., 1., 1., 1., 1., 1., 0., 0.],
                               [0., 1., 1., 1., 1., 1., 1., 1., 0.],
                               [1., 1., 1., 1., 1., 1., 1., 1., 0.],
                               [0., 1., 1., 1., 1., 1., 1., 1., 0.],
                               [0., 0., 1., 1., 1., 1., 1., 0., 0.],
                               [0., 0., 0., 1., 1., 1., 0., 0., 0.],
                               [0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    with torch.no_grad():
        conv.weight.copy_(kernel)

    # get vector of image
    out = conv(image)

    # flat vector
    out = out.view(192, 1)
    return out.detach().numpy().reshape(192, )


def cal_distance(dot_1, dot_2):
    """
    Calculate the Euclidean distance between two points in 2D space.

    Args:
        dot_1 (tuple): Coordinates of the first point (x1, y1).
        dot_2 (tuple): Coordinates of the second point (x2, y2).

    This function computes the Euclidean distance between two points in 2D space
    based on their coordinates (x1, y1) and (x2, y2).

    Returns:
        distance (float): The Euclidean distance between the two points.
    """

    return ((dot_1[0] - dot_2[0]) ** 2 + (dot_1[1] - dot_2[1]) ** 2) ** 0.5


def cosine_similarity(x1, x2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        x1 (array-like): The first vector.
        x2 (array-like): The second vector.

    This function computes the cosine similarity between two vectors, which measures
    the cosine of the angle between them and provides a value indicating their similarity.

    Returns:
        cos_sim (float): The cosine similarity between the two input vectors.
    """

    # calculate cosine similarity with a scalar dot
    cos_sim = dot(x1, x2) / (norm(x1) * norm(x2))
    return cos_sim


def get_metrics(dot_1, dot_2, conv_1, conv_2, count, conv_importance, distance_thershold, conv_thershold):
    """
    Calculate metrics for object matching based on distance and convolution features.

    Args:
        dot_1 (tuple): Coordinates of the first object.
        dot_2 (tuple): Coordinates of the second object.
        conv_1 (array-like): Convolution feature of the first object.
        conv_2 (array-like): Convolution feature of the second object.
        count (int): Count representing the time elapsed since the last observation.
        conv_importance (float): Importance factor for convolution in the metric.
        distance_threshold (float): Threshold for distance-based matching.
        conv_threshold (float): Threshold for convolution-based matching.

    This function computes various metrics for object matching, including distance, convolution
    similarity, and a combined metric. It checks if the objects are a potential match based on
    distance and convolution criteria.

    Returns:
        flag (bool): A boolean indicating whether the objects are a potential match.
        metric (float): The combined metric for object matching.
        distance (float): The distance between the two objects.
        conv_similarity (float): The cosine similarity of the convolution features.
        :param conv_importance:
        :param count:
        :param conv_2:
        :param conv_1:
        :param dot_2:
        :param dot_1:
        :param conv_thershold:
        :param distance_thershold:
    """

    # get distance between two dots
    distance = cal_distance(dot_1, dot_2)

    # get cosine similarity for two vectors
    convol = (1 - cosine_similarity(conv_1, conv_2))

    # calculate metric based on distance and cosine similarity
    metric = distance + conv_importance * convol
    return ((distance < distance_thershold * count) & (
            cosine_similarity(conv_1, conv_2) > conv_thershold)), metric, distance, cosine_similarity(conv_1,
                                                                                                      conv_2)


def read_track(track, name="track.txt"):
    """
    Read tracking data from a text file and organize it into a dictionary.

    Args:
        track (dict): A dictionary containing track data organized by frames.
        name (str): Name of the input text file containing tracking data (default is "track.txt").

    This function reads tracking data from a text file and organizes it into a dictionary where
    each frame's data is stored as a list of object information, including object IDs and bounding
    box coordinates.

    Returns:
        track (dict): A dictionary containing track data organized by frames.
    """

    if os.path.exists(name):
        file_track = open(name)
        lines = file_track.readlines()

        for line in lines:
            line = [int(x) for x in line.split()]
            if line[0] not in track:
                track[line[0]] = []

            # read track like {frame_id: [box_id, box_data]}
            track[line[0]].append([line[1], [line[2], line[3], line[4], line[5]]])
        file_track.close()

    return track


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) to (x1, y1, x2, y2) format.

    Args:
        x (list): List representing bounding box coordinates in (x, y, width, height) format.

    This function takes a list of bounding box coordinates in (x, y, width, height) format and
    converts it to (x1, y1, x2, y2) format, where (x1, y1) is the top-left corner and (x2, y2) is
    the bottom-right corner of the bounding box.

    Returns:
        y (list): List representing bounding box coordinates in (x1, y1, x2, y2) format.
    """

    y = [0] * 4

    # half-width
    dw = x[2] / 2

    # half-height
    dh = x[3] / 2

    # top left x
    y[0] = x[0] - dw

    # top left y
    y[1] = x[1] - dh

    # bottom right x
    y[2] = x[0] + dw

    # bottom right y
    y[3] = x[1] + dh

    return y

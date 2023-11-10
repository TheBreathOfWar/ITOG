import cv2
import numpy as np
import copy as cp

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)
MSGCOLOR = (128, 128, 128)
THICKNESS = 1
LINETYPE = 1
plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)


# convert HEX color to RGB color
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = [hex2color(h) for h in plate_green]


def visualize(frames, annotations, show_ids, show_actions, show_boxes, plate=None, max_num=1):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[tuple]): The predicted results.
        show_ids (bool): Show IDs on frames
        show_actions (bool): Show actions on frames
        show_boxes (bool): Show boxes on frames
        plate (tuple): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """

    # set plate
    if plate is None:
        plate = plate_blue
    plate = [x[::-1] for x in plate]
    frames_out = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    nfpa = len(frames) // len(annotations)
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])

    # iterate through results
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_out[ind]

            # iterate through results per current frames
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue

                # show box
                idx = ann[3]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                if show_boxes:
                    cv2.rectangle(frame, st, ed, plate[0], 2)

                # show action
                if show_actions:
                    for k, lb in enumerate(label):
                        if k >= max_num:
                            break
                        text = abbrev(lb)

                        location = (0 + st[0], 18 + (k + 1) * (-18) + st[1])
                        textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                                   THICKNESS)[0]
                        textwidth = textsize[0]
                        diag0 = (location[0] + textwidth, location[1] - 14)
                        diag1 = (location[0], location[1] + 2)
                        cv2.rectangle(frame, diag0, diag1, plate[k], -1)
                        cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                    FONTCOLOR, THICKNESS, LINETYPE)

                # show IDs
                if show_ids:
                    text = 'id: ' + str(idx)
                    location = (0 + st[0], -18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[0], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_out


def load_label_map(file_path):
    """Load Label Map.
    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, idx, img_h, img_w):
    """Short summary.
    Args:
        human_detection (list): Human detection result.
        result (lis): The predicted label of each human proposal.
        idx (list): The predicted IDs of each human proposal
        img_h (int): The image height.
        img_w (int): The image width.
    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None

    # iterate per boxes, actions and IDs
    for prop, res, i in zip(human_detection, result, idx):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1] for x in res], int(i)))
    return results

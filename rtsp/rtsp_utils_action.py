

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

import os
import shutil

# files to remove before start
FILES = ['track.txt',
         'pose.txt',
         'action_logs.json',
         'pose_logs.json']

# directories to remove before start
DIRS = ['images',
        'crops']


def del_past(track, pose, missed_ids_correct, action_logs, pose_logs, current_frame, window_size):
    """
    Delete previous logs

    Args:
        track (dict): A dictionary containing track data organized by frames.
        pose (dict): A dictionary containing pose data organized by frames.
        missed_ids_correct (dict): A dictionary containing missed IDs for correcting IDs organized by frames.
        action_logs (dict): A dictionary containing action logs organized by frames.
        pose_logs (dict): A dictionary containing pose logs organized by frames.
        current_frame (int): The current frame number.
        window_size (int): The size of window for action prediction.

    This function delete past logs to clean memory.

    Returns:
        track (dict): Cleaned dictionary containing track data organized by frames.
        pose (dict): Cleaned dictionary containing pose data organized by frames.
        missed_ids_correct (dict): Cleaned dictionary containing missed IDs for correcting IDs organized by frames.
        action_logs (dict): Cleaned dictionary containing action logs organized by frames.
        pose_logs (dict): Cleaned dictionary containing pose logs organized by frames.
        current_frame (int): The current frame number.
        window_size (int): The size of window for action prediction.
    """
    if os.path.exists('track.txt'):
        os.remove('track.txt')
    if os.path.exists('pose.txt'):
        os.remove('pose.txt')
    if os.path.exists("images/" + str(current_frame - window_size*5) + ".jpg"):
        os.remove("images/" + str(current_frame - window_size*5) + ".jpg")
    if os.path.exists("crops/" + str(current_frame - window_size*5)):  # add check crops path
        shutil.rmtree("crops/" + str(current_frame - window_size*5))
    if current_frame - window_size*5 in track.keys():
        del track[current_frame - window_size*5]
    if current_frame - window_size*5 in pose.keys():
        del pose[current_frame - window_size*5]
    if current_frame - window_size*5 in action_logs.keys():
        del action_logs[current_frame - window_size*5]
    if current_frame - window_size*5 in pose_logs.keys():
        del pose_logs[current_frame - window_size*5]
    for idx in missed_ids_correct:
        if missed_ids_correct[idx][1] == current_frame - window_size*5:
            del missed_ids_correct[idx]

    return track, pose, missed_ids_correct, action_logs, pose_logs


def del_logs():
    """Delete previous logs"""
    for FILE in FILES:
        if os.path.exists(FILE):
            os.remove(FILE)
    for DIR in DIRS:
        if os.path.exists(DIR):
            shutil.rmtree(DIR)

import os
import shutil
import numpy as np

# files to remove before start
FILES = ['track.txt',
         'pose.txt',
         'output_track.txt',
         'output_pose.txt',
         'data',
         'action_logs.json',
         'pose_logs.json',
         'output.mp4']

# directories to remove before start
DIRS = ['images',
        'crops']


def dense_timestamps(times, n):
    """Make it nx frames"""
    old_frame_interval = (times[1] - times[0])
    start = times[0] - old_frame_interval / n * (n - 1) / 2
    new_frame_inds = np.arange(
        len(times) * n) * old_frame_interval / n + start
    return new_frame_inds.astype(np.int64)


def del_logs():
    """Delete previous logs"""
    for FILE in FILES:
        if os.path.exists(FILE):
            os.remove(FILE)
    for DIR in DIRS:
        if os.path.exists(DIR):
            shutil.rmtree(DIR)

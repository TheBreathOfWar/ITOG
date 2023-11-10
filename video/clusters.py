from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
import pandas as pd
import argparse
import os


def parse_args():
    """Parsing arguments"""

    parser = argparse.ArgumentParser(description='SOKOL')

    # common arguments
    parser.add_argument('--logs',
                        type=str,
                        default='pose_logs.json',
                        help='file with pose logs')
    parser.add_argument('--output',
                        type=str,
                        default='result.jpg',
                        help='output filename')
    parser.add_argument('--start_frame',
                        type=int,
                        default=0,
                        help='start frame for clusters fit')
    parser.add_argument('--end_frame',
                        type=int,
                        default=630,
                        help='end frame for clusters fit')
    parser.add_argument('--step_frame',
                        type=int,
                        default=1,
                        help='step frame for clusters fit')
    parser.add_argument('--perplexity',
                        type=int,
                        default=None,
                        help='number of nearest neighbors')
    parser.add_argument('--iter',
                        type=int,
                        default=5000,
                        help='maximum number of iterations')

    args = parser.parse_args()
    return args


def main():
    """Start cluster processing"""
    args = parse_args()

    # delete past files
    if os.path.exists(args.output):
        os.remove(args.output)

    # write pose logs
    with open(args.logs, "r") as my_file:
        pose_data = my_file.read()

    pose_data = json.loads(pose_data)

    # create data for chosen period of time
    df = []

    # iterate through frames
    for i in range(args.start_frame, args.end_frame, args.step_frame):
        if str(i + 1) in pose_data.keys():
            for pose in pose_data[str(i + 1)].keys():
                df.append(pose_data[str(i + 1)][pose])

    df = pd.DataFrame(df)

    # set nearest neighbors
    if args.perplexity is None:
        perplexity = int(df.shape[0] // 2)
    else:
        perplexity = args.perplexity

    # load model
    model = TSNE(n_components=2, perplexity=perplexity, n_iter=5000)

    # fir model
    transformed = model.fit_transform(df)
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    # draw result
    plt.scatter(x_axis, y_axis)

    # save result
    plt.savefig(args.output, dpi=900, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()

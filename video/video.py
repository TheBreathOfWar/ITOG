import mmcv
import mmengine
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.utils import frame_extract, get_str_type

from boxmot.utils.checks import TestRequirements

import argparse
from argparse import Namespace
import tempfile
import json
import moviepy.editor as mpy
from functools import partial
from ultralytics import YOLO
from PIL import Image

from utils_track import *
from utils_pose import *
from utils_connection import *
from utils_action import *
from utils import *

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git',))


def parse_args():
    """Parsing arguments"""

    parser = argparse.ArgumentParser(description='SOKOL')

    # common arguments
    parser.add_argument('--source',
                        type=str,
                        default='input.mp4',
                        help='video file/url')
    parser.add_argument('--output',
                        type=str,
                        default='output.mp4',
                        help='output filename')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='CPU/CUDA device option')
    parser.add_argument('--short-side',
                        type=int,
                        default=256,
                        help='specify the short-side length of the image')
    parser.add_argument('--predict-stepsize',
                        type=int,
                        default=5,
                        help='give out a prediction per n frames')
    parser.add_argument('--output-stepsize',
                        type=int,
                        default=1,
                        help=(
                            'show one frame per n frames in the video, we should have: predict_stepsize % '
                            'output_stepsize == 0'))
    parser.add_argument('--output-fps',
                        type=int,
                        default=21,
                        help='the fps of video output')
    parser.add_argument('--detection-category-id',
                        type=int,
                        default=0,
                        help='the category id for human detection')
    parser.add_argument('--visualize',
                        type=bool,
                        default=True,
                        help='visualize results on video')
    parser.add_argument('--logs',
                        type=bool,
                        default=True,
                        help='write action and pose logs')
    parser.add_argument('--show-ids',
                        type=bool,
                        default=True,
                        help='show ids on output video')
    parser.add_argument('--show-actions',
                        type=bool,
                        default=True,
                        help='show actions on output video')
    parser.add_argument('--show-boxes',
                        type=bool,
                        default=True,
                        help='show boxes on output video')

    # track arguments
    parser.add_argument('--track-detection-config',
                        type=str,
                        default='yolov8x.pt',
                        help='human track detection config file path')
    parser.add_argument('--track-method',
                        type=str,
                        default='deepocsort',
                        help='human track method')
    parser.add_argument('--track-re_id-model',
                        type=str,
                        default='clip_market1501.pt',
                        help='re_id model file path')
    parser.add_argument('--track-detection-score-thr',
                        type=float,
                        default=0.15,
                        help='the threshold of human track detection score')
    parser.add_argument('--write-track',
                        type=bool,
                        default=True,
                        help='write track postprocess result')

    # track postprocess arguments
    parser.add_argument('--track-postprocess-conv-kernel-size',
                        type=int,
                        default=5,
                        help='kernel size for convolution')
    parser.add_argument('--track-postprocess-same-thr',
                        type=int,
                        default=25,
                        help='threshold for same boxes')
    parser.add_argument('--track-postprocess-min-metric',
                        type=int,
                        default=1000,
                        help='minimum value for track metric')
    parser.add_argument('--track-postprocess-convolution-importance',
                        type=int,
                        default=10000,
                        help='importance of convolution similarity for track metric')
    parser.add_argument('--track-postprocess-distance-thr',
                        type=int,
                        default=8,
                        help='threshold for distance for track metric')
    parser.add_argument('--track-postprocess-convolution-thr',
                        type=float,
                        default=0.9985,
                        help='threshold for convolution for track metric')
    parser.add_argument('--track-postprocess-frame-weight',
                        type=int,
                        default=1,
                        help='weight of frame for convolution')
    parser.add_argument('--track-postprocess-close-thr',
                        type=int,
                        default=5,
                        help='threshold for close boxes')
    parser.add_argument('--track-postprocess-possible-thr',
                        type=int,
                        default=50,
                        help='threshold for possible boxes')
    parser.add_argument('--track-postprocess-del-thr-1',
                        type=float,
                        default=0.985,
                        help='human detection threshold for first stage delete')
    parser.add_argument('--track-postprocess-del-thr-2',
                        type=float,
                        default=0.980,
                        help='human detection threshold for second stage delete')

    # pose arguments
    parser.add_argument('--pose-detection-config',
                        type=str,
                        default='yolov8x-pose.pt',
                        help='human pose detection config file path')
    parser.add_argument('--pose-detection-score-thr',
                        type=float,
                        default=0.02,
                        help='the threshold of human pose detection score')
    parser.add_argument('--write-pose',
                        type=bool,
                        default=True,
                        help='write pose postprocess result')

    # connection arguments
    parser.add_argument('--connection-same-thr',
                        type=int,
                        default=20,
                        help='threshold for same boxes')
    parser.add_argument('--write-connection',
                        type=bool,
                        default=True,
                        help='write connection result')

    # action arguments

    parser.add_argument('--action-config',
                        type=str,
                        default=(
                            'configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21'
                            '-rgb.py'),
                        help='action model config file path')
    parser.add_argument('--action-checkpoint',
                        type=str,
                        default=(
                            'https://download.openmmlab.com/mmaction/detection/ava'
                            '/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                            '/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth'),
                        help='action model checkpoint file/url')
    parser.add_argument('--action-score-thr',
                        type=float,
                        default=0.2,
                        help='the threshold of human action score')
    parser.add_argument('--label-map',
                        type=str,
                        default='label_map.txt',
                        help='label map file')

    args = parser.parse_args()
    return args


def main():
    """Start video processing"""

    # delete past files
    del_logs()

    # make directories for images and crops
    os.mkdir('images')
    os.mkdir('crops')

    # parsing arguments
    args: Namespace = parse_args()

    # track part
    print('Track...')

    # load track model
    model: YOLO = YOLO(args.track_detection_config)

    # get track results
    results = model.track(
        source=args.source,
        conf=args.track_detection_score_thr,
        iou=0.7,
        show=False,
        stream=True,
        device=args.device,
        show_conf=False,
        save_txt=False,
        show_labels=False,
        save=False,
        verbose=False,
        exist_ok=True,
        project=ROOT / 'runs' / 'track',
        name='exp',
        classes=args.detection_category_id,
        imgsz=640,
        vid_stride=1,
        line_width=None
    )

    model.add_callback('on_predict_start', partial(on_predict_start))
    model.predictor.custom_args = args

    # save track, images and crops
    max_frame = 0
    for frame_idx, r in enumerate(results):

        # save images
        max_frame += 1
        img = Image.fromarray(r.orig_img[:, :, ::-1], 'RGB')
        img.save(f"images/{frame_idx + 1}.jpg")

        if r.boxes.data.shape[1] == 7:

            # save track
            model.predictor.mot_txt_path = Path('track.txt')
            write_track_results(model.predictor.mot_txt_path, r, frame_idx)

            # save crops
            for d in r.boxes:
                save_one_box(
                    d.xyxy,
                    r.orig_img.copy(),
                    file=(Path('crops') /
                          str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                          ),
                    BGR=True
                )

    print('Track done successfully')

    # track postprocess part
    print('Postprocess...')

    # read track and correct human ids
    track = correct_ids('track.txt',
                        args.track_postprocess_conv_kernel_size,
                        args.track_postprocess_same_thr,
                        args.track_postprocess_min_metric,
                        args.track_postprocess_convolution_importance,
                        args.track_postprocess_distance_thr,
                        args.track_postprocess_convolution_thr,
                        args.track_postprocess_frame_weight,
                        args.track_postprocess_close_thr,
                        args.track_postprocess_possible_thr,
                        )

    # delete not human detections
    track = del_not_human(track, args.track_postprocess_del_thr_1, args.track_postprocess_conv_kernel_size)

    # show missed human detections
    track = show_boxes(track)

    # add missed frames
    track = add_frames(track, max_frame)

    # delete not human detections with a stricter threshold
    track = del_not_human(track, args.track_postprocess_del_thr_2, args.track_postprocess_conv_kernel_size)

    # change ids numbers
    track = change_numbers(track)

    # write track results
    if args.write_track:
        write_track(track)

    print('Track postprocess done successfully')

    # pose part
    print('Pose...')

    # load pose model
    model = YOLO(args.pose_detection_config)

    # get pose results
    results = model(
        source=args.source,
        conf=args.pose_detection_score_thr,
        show=False,
        stream=True,
        show_conf=False,
        save_txt=False,
        verbose=False,
        exist_ok=True,
        project=ROOT / 'runs' / 'pose',
        name='exp',
        show_labels=False,
        save=False,
        classes=args.detection_category_id,
        iou=0.1
    )

    # save pose
    for frame_idx, r in enumerate(results):
        write_pose_results('pose.txt', r, frame_idx)

    print('Pose done successfully')

    # pose postprocess part
    print('Postprocess...')

    # get pose coordinates relative to the center of the box
    pose = correct_pose('pose.txt')

    # write pose results
    if args.write_pose:
        write_pose(pose)

    print('Pose postprocess done successfully')

    # connection part
    print('Connection...')

    # connect track with pose
    data = connection(track, pose, args.connection_same_thr)

    # show missed human poses
    data = show_poses(data)

    # set mean pose for missed poses
    data = mean_pose(data)

    # write connection results
    if args.write_connection:
        write_data(data)

    print('Connect done successfully')

    # action part
    print('Action...')

    # get video frames
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames = frame_extract(
        args.source, out_dir=tmp_dir.name)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # resize frames to shortside
    new_w, new_h = mmcv.rescale_size((w, h), (args.short_side, np.Inf))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h

    # get clip_len, frame_interval and calculate center index of each clip
    config = mmengine.Config.fromfile(args.action_config)
    val_pipeline = config.val_pipeline

    # get timestamps for prediction
    sampler = [
        x for x in val_pipeline if get_str_type(x['type']) == 'SampleAVAFrames'
    ][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    # load label map
    label_map = load_label_map(args.label_map)

    # load human detections
    human_detections = []
    for frame in timestamps:
        boxes = []
        if frame in data.keys():
            for box in data[frame]:
                boxes.append(np.array(xywh2xyxy(box[1]), dtype='float32'))
        else:
            boxes.append(np.array([], dtype='float32'))
        human_detections.append(np.array(boxes, dtype='float32'))

    torch.cuda.empty_cache()
    for i in range(len(human_detections)):
        if len(human_detections[i].shape) == 1 or human_detections[i].shape == (1, 0):
            det = human_detections[i].reshape((0, 4))
        else:
            det = human_detections[i]
        print(det, det.shape, type(det))
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

    # build STDET model
    try:
        config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
    except KeyError:
        pass

    # load action model
    config.model.backbone.pretrained = None
    model = MODELS.build(config.model)

    load_checkpoint(model, args.action_checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    # get action predictions
    predictions = []
    img_norm_cfg = dict(
        mean=np.array(config.model.data_preprocessor.mean),
        std=np.array(config.model.data_preprocessor.std),
        to_rgb=False)

    # performing action for each clip
    prog_bar = mmengine.ProgressBar(len(timestamps))
    for timestamp, proposal in zip(timestamps, human_detections):
        if proposal.shape[0] == 0:
            predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(args.device)
        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal)
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
        with torch.no_grad():
            result = model(input_tensor, [datasample], mode='predict')
            scores = result[0].pred_instances.scores
            prediction = []
            for i in range(proposal.shape[0]):
                prediction.append([])
            for i in range(scores.shape[1]):
                if i not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if scores[j, i] > args.action_score_thr:
                        prediction[j].append((label_map[i], scores[j, i].item()))
            predictions.append(prediction)
        prog_bar.update()

    # get track ids
    ids = []
    for frame in timestamps:
        boxes = []
        if frame in data.keys():
            for box in data[frame]:
                boxes.append(box[0])
        ids.append(np.array(boxes, dtype='float32'))

    # connection results
    results = []
    for human_detection, prediction, i in zip(human_detections, predictions, ids):
        results.append(pack_result(human_detection, prediction, i, new_h, new_w))

    print('Action done successfully')

    # visualize part
    if args.visualize:
        print('Visualization...')
        dense_n = int(args.predict_stepsize / args.output_stepsize)
        frames = [
            cv2.imread(frame_paths[i - 1])
            for i in dense_timestamps(timestamps, dense_n)
        ]
        vis_frames = visualize(frames, results, args.show_ids, args.show_actions, args.show_boxes)
        vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                    fps=args.output_fps)
        vid.write_videofile(args.output)

        tmp_dir.cleanup()

        print('Visualization done successfully')

    # logging part
    if args.logs:
        print('Logging...')

        # action logs
        action_logs = {}
        for i in range(len(timestamps)):
            if results[i] is not None:
                action_logs[int(timestamps[i])] = {}
                for result in results[i]:
                    if not result[1]:
                        action_logs[int(timestamps[i])][int(result[3])] = 'stand'
                    else:
                        action_logs[int(timestamps[i])][int(result[3])] = result[1][0]
        json_string = json.dumps(action_logs)

        # write action logs
        with open("action_logs.json", "w") as json_file:
            json_file.write(json_string)

        # pose logs
        pose_logs = {}
        for i in range(len(timestamps)):
            if int(timestamps[i]) in data.keys():
                pose_logs[int(timestamps[i])] = {}
                for box in data[int(timestamps[i])]:
                    pose_logs[int(timestamps[i])][int(box[0])] = list(box[2])
        json_string = json.dumps(pose_logs)

        # write pose logs
        with open("pose_logs.json", "w") as json_file:
            json_file.write(json_string)

        print('Logging done successfully')


if __name__ == '__main__':
    main()


'''
5. Протестировать github
5*. Избавиться от ненужных файлов
6. Сделать коммит-историю
9. Протестировать на других видео

assets
demo
docker
docs
examples
detection
projects
resources
tests
utils.py
run.py
setup.py
setup.cgf
README
yolo_tracking
'''

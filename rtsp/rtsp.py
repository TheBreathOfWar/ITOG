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
from functools import partial
from ultralytics import YOLO

from rtsp_utils_track import *
from rtsp_utils_pose import *
from rtsp_utils_connection import *
from rtsp_utils_action import *
from rtsp_utils import *

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git',))


def parse_args():
    """Parsing arguments"""

    parser = argparse.ArgumentParser(description='SOKOL')

    # common arguments
    parser.add_argument('--camera',
                        type=int,
                        default=17,
                        help='camera for rtsp stream')
    parser.add_argument('--stream',
                        type=str,
                        default='main',
                        help='rtsp stream: "main" or "sub"')
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
    parser.add_argument('--detection-category-id',
                        type=int,
                        default=0,
                        help='the category id for human detection')
    parser.add_argument('--logs',
                        type=bool,
                        default=True,
                        help='write action and pose logs')
    parser.add_argument('--stop',
                        type=bool,
                        default=True,
                        help='stop stream')
    parser.add_argument('--stop-frame',
                        type=int,
                        default=400,
                        help='frame to stop stream')

    # track arguments
    parser.add_argument('--track-detection-config',
                        type=str,
                        default='yolov8l.pt',
                        help='human track detection config file path')
    parser.add_argument('--track-method',
                        type=str,
                        default='deepocsort',
                        help='human track method')
    parser.add_argument('--track-re_id-model',
                        type=str,
                        default='osnet_x0_25_market1501.pt',
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
                        default=40,
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
                        default='yolov8l-pose.pt',
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
                        default=0.1,
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

    # parsing arguments
    args: Namespace = parse_args()

    # make directories for images and crops
    os.mkdir('images')
    os.mkdir('crops')

    # capture rtsp stream
    url = f"rtsp://web:merkatorweb@195.112.117.250:20211/cameras/{args.camera}/streaming/{args.stream}"
    video = '17.mp4'
    cap = cv2.VideoCapture(video)

    # frame counter
    current_frame = 1

    # prepare track
    track = {}

    # prepare track postprocess

    # get human vector
    file_convolution = open('convolutions/convolution.txt', "r")

    human_conv = [float(x) for x in file_convolution.readline().split()]
    human_conv = torch.FloatTensor(human_conv)

    # path of boxes' crops
    crops_path = 'crops'

    # dictionary for vectors of convolutions of boxes' crops
    convs = {}

    # correct IDs

    # previous frame
    last_frame_correct = 0

    # dictionary for missed human IDs
    missed_ids_correct = {}

    # show missed human detections

    # previous frame
    last_frame_show = 0

    # dictionary for missed IDs
    missed_ids_show = {}

    # change number

    # initialize the next available object ID
    next_number = 0

    # create a dictionary to store mappings of old IDs to new IDs
    ids = {}

    # prepare pose
    pose = {}

    # prepare pose postprocess

    # prepare connection

    # previous frame
    last_frame_connection = 0

    # dictionary for previous poses IDs
    already_ids_connection = {}

    # prepare action

    # get clip_len, frame_interval and calculate center index of each clip
    config = mmengine.Config.fromfile(args.action_config)
    val_pipeline = config.val_pipeline

    sampler = [
        x for x in val_pipeline if get_str_type(x['type']) == 'SampleAVAFrames'
    ][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval

    # load label_map
    label_map = load_label_map(args.label_map)

    config.model.backbone.pretrained = None
    action_model = MODELS.build(config.model)

    load_checkpoint(action_model, args.action_checkpoint, map_location='cpu')
    action_model.to(args.device)
    action_model.eval()

    # prepare logging
    action_logs = {}
    pose_logs = {}

    while cap.isOpened():
        ret, frame = cap.read()
        print(current_frame)

        track, pose, missed_ids_correct, action_logs, pose_logs = del_past(track,
                                                                           pose,
                                                                           missed_ids_correct,
                                                                           action_logs,
                                                                           pose_logs,
                                                                           current_frame,
                                                                           window_size)

        # save current frame
        name = str(current_frame) + ".jpg"
        cv2.imwrite("images/" + name, frame)

        if current_frame >= window_size // 2:

            print(track)

            # track part

            # load track model
            model = YOLO(args.track_detection_config)

            # get track results
            results = model.track(
                source="images/" + name,
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
                project=ROOT / 'rtsp' / 'track',
                name='exp',
                classes=args.detection_category_id,
                imgsz=640,
                vid_stride=1,
                line_width=None
            )

            model.add_callback('on_predict_start', partial(on_predict_start))
            model.predictor.custom_args = args

            # save track, images and crops
            for r in results:
                if r.boxes.data.shape[1] == 7:

                    # save track
                    model.predictor.mot_txt_path = Path('track.txt')
                    write_track_results(model.predictor.mot_txt_path, r, current_frame - 1)

                    # save crops
                    for d in r.boxes:
                        save_one_box(
                            d.xyxy,
                            r.orig_img.copy(),
                            file=(Path('crops') /
                                  f'{current_frame - 1}' / Path(str(int(d.id.cpu().numpy().item())) + '.jpg')
                                  ),
                            BGR=True

                        )

            # track postprocess part

            # read track and correct human IDs

            track, current_frame, last_frame_correct, missed_ids_correct, crops_path, convs = correct_ids(
                'track.txt',
                args.track_postprocess_conv_kernel_size,
                args.track_postprocess_same_thr,
                args.track_postprocess_min_metric,
                args.track_postprocess_convolution_importance,
                args.track_postprocess_distance_thr,
                args.track_postprocess_convolution_thr,
                args.track_postprocess_frame_weight,
                args.track_postprocess_close_thr,
                args.track_postprocess_possible_thr,
                track,
                current_frame,
                last_frame_correct,
                missed_ids_correct,
                crops_path,
                convs)

            # delete not human detections
            track, current_frame, human_conv = del_not_human(track,
                                                             current_frame,
                                                             args.track_postprocess_del_thr_1,
                                                             args.track_postprocess_conv_kernel_size,
                                                             human_conv)

            # show missed human detections
            track, current_frame, last_frame_show, missed_ids_show = show_boxes(track,
                                                                                current_frame,
                                                                                last_frame_show,
                                                                                missed_ids_show)

            # add missed frames
            track, current_frame = add_frame(track, current_frame)

            # delete not human detections
            track, current_frame, human_conv = del_not_human(track,
                                                             current_frame,
                                                             args.track_postprocess_del_thr_2,
                                                             args.track_postprocess_conv_kernel_size,
                                                             human_conv)

            # change IDs numbers
            track, current_frame, next_number, ids = change_number(track,
                                                                   current_frame,
                                                                   next_number,
                                                                   ids)

            # pose part

            # load pose model
            model = YOLO(args.pose_detection_config)

            # get pose results
            results = model(
                source="images/" + name,
                conf=args.pose_detection_score_thr,
                show=False,
                stream=True,
                show_conf=False,
                save_txt=False,
                verbose=False,
                exist_ok=True,
                project=ROOT / 'rtsp' / 'pose',
                name='exp',
                show_labels=False,
                save=False,
                classes=args.detection_category_id,
                iou=0.1
            )

            # save pose
            for r in results:
                write_pose_results('pose.txt', r, current_frame - 1)

            # pose postprocess part

            # get pose coordinates relative to the center of the bo
            pose, current_frame = correct_pose(pose, current_frame, 'pose.txt')

            # connection part

            # connect track with pose
            track = connect(track, pose, current_frame, args.connection_same_thr)

            # show missed human poses
            track, current_frame, last_frame_connection, already_ids_connection = show_poses(track,
                                                                                             current_frame,
                                                                                             last_frame_connection,
                                                                                             already_ids_connection)

            # set mean pose for missed poses
            track = mean_pose(track, current_frame)

        # get timestamps for prediction
        timestamp = current_frame - window_size // 2

        # action part
        if current_frame >= window_size and (
                current_frame - window_size) % args.predict_stepsize == 0 and timestamp in track.keys():

            tmp_dir = tempfile.TemporaryDirectory()

            frame_paths = []
            original_frames = []
            start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
            frame_inds = start_frame + np.arange(0, window_size, frame_interval)
            frame_inds = list(frame_inds - 1)

            for f in frame_inds:
                frame_path, original_frame = frame_extract(f'images/{f}.jpg', out_dir=tmp_dir.name)
                frame_paths.append(frame_path[0])
                original_frames.append(original_frame[0])

            # resize frames to shortside
            h, w, _ = original_frames[0].shape
            new_w, new_h = mmcv.rescale_size((w, h), (args.short_side, np.Inf))
            frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
            w_ratio, h_ratio = new_w / w, new_h / h

            # load human_detections
            boxes = []
            for box in track[timestamp]:
                boxes.append(np.array(xywh2xyxy(box[1]), dtype='float32'))

            torch.cuda.empty_cache()

            if len(np.array(boxes, dtype='float32').shape) == 1 or np.array(boxes, dtype='float32').shape == (1, 0):
                det = np.array(boxes, dtype='float32').reshape((0, 4))
            else:
                det = np.array(boxes, dtype='float32')
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            boxes = torch.from_numpy(det[:, :4]).to(args.device)

            # Build STDET model
            try:
                config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
            except KeyError:
                pass

            # get predictions
            img_norm_cfg = dict(
                mean=np.array(config.model.data_preprocessor.mean),
                std=np.array(config.model.data_preprocessor.std),
                to_rgb=False)

            imgs = [frame.astype(np.float32) for frame in frames]
            _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
            input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(args.device)
            datasample = ActionDataSample()
            datasample.proposals = InstanceData(bboxes=boxes)
            datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
            with torch.no_grad():
                result = action_model(input_tensor, [datasample], mode='predict')
                scores = result[0].pred_instances.scores
                prediction = []
                for i in range(boxes.shape[0]):
                    prediction.append([])
                for i in range(scores.shape[1]):
                    if i not in label_map:
                        continue
                    for j in range(boxes.shape[0]):
                        if scores[j, i] > args.action_score_thr:
                            prediction[j].append((label_map[i], scores[j, i].item()))

            # get track ids
            inds = []
            for box in track[timestamp]:
                inds.append(box[0])

            # connection results
            result = pack_result(boxes, prediction, inds, new_h, new_w)

            tmp_dir.cleanup()
            # logging part

            # action logs
            if result is not None:
                action_logs[int(timestamp)] = {}
                for r in result:
                    if not r[1]:
                        action_logs[int(timestamp)][int(r[3])] = 'stand'
                    else:
                        action_logs[int(timestamp)][int(r[3])] = r[1][0]
            json_string = json.dumps(action_logs)

            # write action log
            with open("action_logs.json", "w") as json_file:
                json_file.write(json_string)

            # pose logs
            pose_logs[int(timestamp)] = {}
            for box in track[int(timestamp)]:
                pose_logs[int(timestamp)][int(box[0])] = list(box[2])
            json_string = json.dumps(pose_logs)

            # write pose logs
            with open("pose_logs.json", "w") as json_file:
                json_file.write(json_string)

        current_frame += 1
        if args.stop and current_frame >= args.stop_frame + window_size // 2:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

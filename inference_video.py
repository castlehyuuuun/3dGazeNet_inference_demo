import tqdm
import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
import cv2
from utils import config as cfg, update_config, get_logger, Timer, VideoLoader, VideoSaver, show_result, draw_results
from models import FaceDetectorIF as FaceDetector
from models import GazePredictorHandler as GazePredictor
from matplotlib.path import Path as MplPath
from shapely.geometry import LineString, Polygon
from sungjoon import gaze_on_bbox_with_multiple_scales

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

right_monitor_keywords = ["000_000741294612", "000_000036592912"]
left_monitor_keywords = ["001_000137300112", "001_000336294412"]

def get_monitor_coords(video_path):
    filename = os.path.basename(video_path)
    if filename.startswith(("processed_A1", "A1", "processed_A2", "A2")):
        if "001_000336294412" in filename:
            return np.array([(1, 60), (102, 119), (101, 220), (0, 208)]), "left"
        elif "000_000036592912" in filename:
            return np.array([(559, 106), (638, 60), (639, 206), (559, 215)]), "right"
    else:
        if "000_000741294612" in video_path:
            return np.array([(565, 84), (639, 45), (639, 195), (560, 203)]), "right"
        elif "001_000137300112" in video_path:
            return np.array([(1, 25), (139, 112), (137, 211), (1, 189)]), "left"
    return None, None

def project_gaze_to_image_2d(eye_center, gaze_vec, scale=250):
    dx, dy = float(gaze_vec[0]), float(gaze_vec[1])
    dx *= -1
    norm = np.linalg.norm([dx, dy])
    if norm < 1e-5:
        return eye_center
    dx /= norm
    dy /= norm
    return (eye_center[0] + dx * scale, eye_center[1] + dy * scale)

def is_inside_monitor(gaze_vec, eye_center, monitor_coords):
    dx, dy = -float(gaze_vec[0]), -float(gaze_vec[1])  # negate for coordinate flip
    norm = np.linalg.norm([dx, dy])
    if norm < 1e-5:
        return False

    dx /= norm
    dy /= norm
    end_point = (eye_center[0] + dx * 10000, eye_center[1] + dy * 10000)

    gaze_line = LineString([eye_center, end_point])
    monitor_poly = Polygon(monitor_coords)

    inside = gaze_line.intersects(monitor_poly)
    # distance = monitor_poly.distance(gaze_line)  # 거리 측정
    #
    # print(f"[DEBUG] end_point=({end_point[0]:.2f}, {end_point[1]:.2f})")
    # print(f"[DEBUG] intersects monitor: {inside}, distance: {distance:.2f}")

    return inside


def average_output(out_dict, prev_dict):
    out_dict['gaze_out'] += prev_dict['gaze_out']
    norm = np.linalg.norm(out_dict['gaze_out'])
    out_dict['gaze_out'] = np.array([0.0, 0.0, 0.0]) if norm < 1e-6 or np.isnan(norm) else out_dict['gaze_out'] / norm
    return out_dict

@Timer(name='Forward', fps=True, pprint=False)
def infer_once(img, detector, predictor, draw, prev_dict=None, monitor_coords=None, video_path=None):
    out_img = None
    out_dict = None
    lms5_out = None
    bboxes, lms5, faces = detector.run(img)
    if bboxes is not None:
        idxs_sorted = sorted(range(len(bboxes)), key=lambda k: bboxes[k][2])
        lms5 = lms5[idxs_sorted[-1]]
        bboxes = bboxes[idxs_sorted[-1]]
        lms5_out = lms5
        out_dict = predictor(img, lms5, undo_roll=True)
        if prev_dict is not None:
            out_dict = average_output(out_dict, prev_dict)
        if draw and out_dict is not None:
            out_img = draw_results(img, lms5, out_dict, video_path=video_path)
            if monitor_coords is not None:
                for i in range(4):
                    pt1 = tuple(monitor_coords[i])
                    pt2 = tuple(monitor_coords[(i + 1) % 4])
                    cv2.line(out_img, pt1, pt2, (255, 255, 0), 2)
    return out_img, out_dict, lms5_out

def inference(cfg, video_path, draw, smooth):
    detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
    predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)
    loader = VideoLoader(video_path, cfg.DETECTOR.IMAGE_SIZE, use_letterbox=False)
    save_dir = video_path[:video_path.rfind('.')] + f'_out_{cfg.PREDICTOR.BACKBONE_TYPE}_x{cfg.PREDICTOR.IMAGE_SIZE[0]}_{cfg.PREDICTOR.MODE}'
    os.makedirs(save_dir.strip(), exist_ok=True)
    monitor_coords, monitor_type = get_monitor_coords(video_path)
    fps = loader.fps or 30.0
    saver = VideoSaver(output_dir=save_dir, fps=fps, img_size=loader.vid_size, vid_size=640, save_images=False)
    tq = tqdm.tqdm(loader)
    prev_dict = None
    pred_gaze_all, is_inside_all = [], []
    for frame_idx, input in tq:
        if input is None:
            break
        out_img, out_dict, lms5 = infer_once(input, detector, predictor, draw, prev_dict, monitor_coords, video_path)
        if out_img is None or out_dict is None or lms5 is None:
            out_img = input.copy()
            pred_gaze_all.append((0.0, 0.0, 0.0))
            is_inside_all.append(0)
            prev_dict = None
        else:
            gaze = out_dict['gaze_out']
            eye_center = (lms5[2] + lms5[0]) / 2
            # inside = is_inside_monitor(gaze, eye_center, monitor_coords)
            xs = monitor_coords[:, 0]
            ys = monitor_coords[:, 1]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            cx, cy = eye_center
            dx, dy = -gaze[0], -gaze[1]
            inside = gaze_on_bbox_with_multiple_scales(cx, cy, dx, dy, bbox, num_steps=10)

            is_inside_all.append(1 if inside else 0)
            pred_gaze_all.append(gaze)
            prev_dict = out_dict.copy() if smooth else None
        saver(out_img, frame_idx)
    with open(f'{save_dir}/predicted_gaze_vectors.txt', 'w') as f:
        f.writelines([f"{x:.4f}, {y:.4f}, {z:.4f}\n" for (x, y, z) in pred_gaze_all])
    with open(f'{save_dir}/gaze_inside_monitor.txt', 'w') as f:
        f.writelines([f"{v}\n" for v in is_inside_all])

    ratio = sum(is_inside_all) / len(is_inside_all) if is_inside_all else 0
    label = " 정반응" if ratio >= 0.5 else " 오반응"
    print(f"{video_path} → {label} ({ratio:.2%})")

    thresholds = [30, 60, 90, 120]
    total_len = len(is_inside_all)
    total_positive = sum(is_inside_all)

    with open(f'{save_dir}/gaze_inside_monitor_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"전체 프레임 수: {total_len}, 정반응 수: {total_positive}\n\n")
        for threshold in thresholds:
            label = "정반응" if total_positive >= threshold else "오반응"
            f.write(f"{threshold}프레임 기준 → {label} (정반응 수: {total_positive} / 기준: {threshold})\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Inference Gaze')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    known_args, rest = parser.parse_known_args()
    update_config(known_args.cfg)
    parser.add_argument('--video_path', help='Video file to run', required=True, type=str)
    parser.add_argument('--gpu_id', help='id of the gpu to utilize', default=0, type=int)
    parser.add_argument('--no_draw', help='Draw and save the results', action='store_true')
    parser.add_argument('--smooth_predictions', help='Average predictions between consecutive frames', action='store_true')
    return parser.parse_args()

# if __name__ == '__main__':
#     args = parse_args()
#     exp_save_path = f'log/{cfg.EXP_NAME}'; os.makedirs(exp_save_path, exist_ok=True)
#     logger = get_logger(exp_save_path, save=True, use_tqdm=True)
#     Timer.save_path = exp_save_path
#     video_path = args.video_path
#     is_dir = os.path.isdir(video_path)
#     with torch.no_grad():
#         if is_dir:
#             video_files = [
#                 f for f in os.listdir(video_path)
#                 if f.endswith('.mp4') and '02_rec' in f and any(k in f for k in right_monitor_keywords + left_monitor_keywords)
#             ]
#             for vid in video_files:
#                 full_path = os.path.join(video_path, vid)
#                 print(f"[INFO] Processing: {full_path}")
#                 try:
#                     inference(cfg=cfg, video_path=full_path, draw=not args.no_draw, smooth=args.smooth_predictions)
#                 except Exception as e:
#                     print(f"[ERROR] Failed on {vid}: {e}")
#         else:
#             inference(cfg=cfg, video_path=video_path, draw=not args.no_draw, smooth=args.smooth_predictions)
if __name__ == '__main__':
    args = parse_args()
    exp_save_path = f'log/{cfg.EXP_NAME}'
    os.makedirs(exp_save_path, exist_ok=True)
    logger = get_logger(exp_save_path, save=True, use_tqdm=True)
    Timer.save_path = exp_save_path
    video_path = args.video_path
    is_dir = os.path.isdir(video_path)

    act_1_keywords = ["000_000741294612", "000_000036592912"]
    act_2_keywords = ["001_000137300112", "001_000336294412"]
    summary_bins_act1 = {0.2: [], 0.4: [], 0.6: [], 0.8: []}
    summary_bins_act2 = {0.2: [], 0.4: [], 0.6: [], 0.8: []}


    def write_summary(file_path, is_inside_list, tag):
        thresholds = [30, 60, 90, 120]
        total_len = len(is_inside_list)
        total_positive = sum(is_inside_list)

        with open(file_path, 'w', encoding='utf-8') as f:
            if total_len == 0:
                f.write(f"[{tag}] 처리된 프레임이 없습니다.\n")
                return

            f.write(f"[{tag}] 전체 프레임 수: {total_len}, 정반응 수: {total_positive}\n")

            for threshold in thresholds:
                if total_len < threshold:
                    f.write(f"[{tag}] {threshold}프레임 이상 → 프레임 수 부족 (총 {total_len}프레임)\n")
                    continue

                # ✅ 전체 기준으로 정반응 개수만 비교
                label = "정반응" if total_positive >= threshold else "오반응"
                f.write(
                    f"[{tag}] {threshold}프레임 이상 정반응 필요 → {label} "
                    f"(정반응 수: {total_positive}, 기준: {threshold}프레임)\n"
                )

    with torch.no_grad():
        if is_dir:
            video_files = [
                f for f in os.listdir(video_path)
                if f.endswith('.mp4') and '02_rec' in f and any(
                    k in f for k in right_monitor_keywords + left_monitor_keywords)
            ]
            for vid in video_files:
                full_path = os.path.join(video_path, vid)
                print(f"[INFO] Processing: {full_path}")
                try:
                    inference(cfg=cfg, video_path=full_path, draw=not args.no_draw, smooth=args.smooth_predictions)
                except Exception as e:
                    print(f"[ERROR] Failed on {vid}: {e}")
        else:
            print("[ERROR] 디렉터리가 아닙니다. --video_path는 폴더여야 전체 집계가 가능합니다.")
            exit()

    # with torch.no_grad():
    #     if is_dir:
    #         video_files = [
    #             f for f in os.listdir(video_path)
    #             if f.endswith('.mp4') and '02_rec' in f and any(k in f for k in right_monitor_keywords + left_monitor_keywords)
    #         ]
    #         for vid in video_files:
    #             full_path = os.path.join(video_path, vid)
    #             print(f"[INFO] Processing: {full_path}")
    #             try:
    #                 # inference 후 결과 받아서 누적
    #                 detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
    #                 predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)
    #                 loader = VideoLoader(full_path, cfg.DETECTOR.IMAGE_SIZE, use_letterbox=False)
    #                 monitor_coords, monitor_type = get_monitor_coords(full_path)
    #                 save_dir = full_path[:full_path.rfind('.')] + f'_out_{cfg.PREDICTOR.BACKBONE_TYPE}_x{cfg.PREDICTOR.IMAGE_SIZE[0]}_{cfg.PREDICTOR.MODE}'
    #                 os.makedirs(save_dir.strip(), exist_ok=True)
    #                 fps = loader.fps or 30.0
    #                 saver = VideoSaver(output_dir=save_dir, fps=fps, img_size=loader.vid_size, vid_size=640, save_images=False)
    #                 tq = tqdm.tqdm(loader)
    #                 prev_dict = None
    #                 pred_gaze_all, is_inside_all = [], []
    #                 for frame_idx, input in tq:
    #                     if input is None:
    #                         break
    #                     out_img, out_dict, lms5 = infer_once(input, detector, predictor, not args.no_draw, prev_dict, monitor_coords, full_path)
    #                     if out_img is None or out_dict is None or lms5 is None:
    #                         out_img = input.copy()
    #                         pred_gaze_all.append((0.0, 0.0, 0.0))
    #                         is_inside_all.append(0)
    #                         prev_dict = None
    #                     else:
    #                         gaze = out_dict['gaze_out']
    #                         # gaze[0] *= -1 # dx 뒤집어야할 것 같음
    #                         # gaze[1] *= -1  # y축도 뒤집어야 하면 주석 해제
    #                         eye_center = (lms5[2] + lms5[0]) / 2
    #                         inside = is_inside_monitor(gaze, eye_center, monitor_coords)
    #                         is_inside_all.append(1 if inside else 0)
    #                         pred_gaze_all.append(gaze)
    #                         prev_dict = out_dict.copy() if args.smooth_predictions else None
    #                     saver(out_img, frame_idx)
    #
    #                 with open(f'{save_dir}/predicted_gaze_vectors.txt', 'w') as f:
    #                     f.writelines([f"{x:.4f}, {y:.4f}, {z:.4f}\n" for (x, y, z) in pred_gaze_all])
    #                 # 정반응 요약 텍스트 출력 (절대 프레임 기준)
    #                 with open(f'{save_dir}/gaze_inside_monitor_summary.txt', 'w', encoding='utf-8') as f:
    #                     total_len = len(is_inside_all)
    #                     total_positive = sum(is_inside_all)
    #                     total_ratio = total_positive / total_len if total_len > 0 else 0
    #
    #                     f.write(f"전체 프레임 수: {total_len}, 정반응 수: {total_positive}, 비율: {total_ratio:.2%}\n")
    #
    #                     thresholds = [30, 60, 90, 120]
    #                     for threshold in thresholds:
    #                         if total_positive >= threshold:
    #                             label = "정반응"
    #                         else:
    #                             label = "오반응"
    #                         f.write(f"{threshold}프레임 기준 → {label} (정반응 수: {total_positive}, 기준: {threshold})\n")
    #
    #
    #             except Exception as e:
    #                 print(f"[ERROR] Failed on {vid}: {e}")
    #     else:
    #         print("[ERROR] 디렉터리가 아닙니다. --video_path는 폴더여야 전체 집계가 가능합니다.")
    #         exit()

